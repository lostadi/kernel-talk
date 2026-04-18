"""
core/mirror/store.py
─────────────────────
The Mirror — Stage 4: Unified Storage Layer

Two retrieval paradigms, one interface:

  1. Vector search   → semantic similarity (ChromaDB)
  2. Graph traversal → structural context  (KernelGraph)

Neither is sufficient alone. Vector search finds the most semantically
relevant code. Graph traversal expands that seed into its architectural
neighborhood — the callers, callees, and data structures that give
the LLM the full structural picture it needs to synthesize a real answer.

The HybridResult combines both. The caller (retriever.py) decides how to
weight them. By default: vector results are seeds, graph results are context.

Persistence:
  ChromaDB uses a local disk collection (no server needed).
  The KernelGraph is serialized to GraphML.
  Both live in the same storage_dir, so the whole Mirror is one directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .parser import CodeNode
from .graph import KernelGraph, GraphContext
from .embedder import CodeEmbedder


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class VectorResult:
    node: CodeNode
    score: float        # cosine similarity ∈ [-1, 1], higher is more similar

@dataclass
class HybridResult:
    """
    The synthesizer's input: a ranked list of primary hits (vector) plus
    structural context (graph neighbors), deduplicated.

    primary:   Direct vector search results, ranked by similarity.
    context:   Graph-expanded neighbors — not necessarily high-similarity
               but structurally related. Callers, callees, referenced structs.
    graph_ctx: The raw GraphContext for inspection / visualization.
    """
    primary: list[VectorResult]
    context: list[CodeNode]
    graph_ctx: GraphContext

    def all_nodes(self) -> list[CodeNode]:
        """All nodes, primary first, then context (deduped)."""
        seen = {r.node.id for r in self.primary}
        result = [r.node for r in self.primary]
        for node in self.context:
            if node.id not in seen:
                result.append(node)
                seen.add(node.id)
        return result

    def to_context_text(self, max_nodes: int = 15) -> str:
        """
        Render nodes into the text block the synthesizer will receive.
        We format each node with its file path, line range, and code —
        enough for the LLM to reason about structure and cite precisely.
        """
        nodes = self.all_nodes()[:max_nodes]
        blocks = []
        for node in nodes:
            header = (
                f"=== {node.node_type.upper()}: {node.symbol_name} ===\n"
                f"File: {node.file_path}  Lines: {node.line_start}–{node.line_end}\n"
            )
            if node.docstring:
                header += f"Doc: {node.docstring[:300]}\n"
            blocks.append(header + node.code[:1000])  # cap code length per node
        return "\n\n".join(blocks)


# ─── Unified Store ────────────────────────────────────────────────────────────

class KernelStore:
    """
    The single access point for all Mirror data.

    Wraps ChromaDB (vector index) and KernelGraph (structural graph)
    behind a unified interface. Ingestion, retrieval, and persistence
    all go through here.

    Usage:
        store = KernelStore.create("/path/to/storage")
        store.index_nodes(nodes)          # ingest CodeNodes
        results = store.hybrid_search(    # retrieve
            "why does schedule() block?",
            top_k=8, hops=2
        )
    """

    CHROMA_COLLECTION = "kernel_code"
    GRAPH_FILENAME    = "kernel_graph.graphml"
    EMBEDDER_MODEL    = "microsoft/codebert-base"

    def __init__(
        self,
        storage_dir: str | Path,
        embedder: CodeEmbedder | None = None,
        graph: KernelGraph | None = None,
    ):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = embedder or CodeEmbedder(self.EMBEDDER_MODEL)
        self.graph    = graph    or KernelGraph()

        # Lazy ChromaDB init
        self._chroma_client = None
        self._collection    = None

    @classmethod
    def create(cls, storage_dir: str | Path, **kwargs) -> "KernelStore":
        """Create a fresh store at storage_dir."""
        return cls(storage_dir, **kwargs)

    @classmethod
    def load(cls, storage_dir: str | Path, **kwargs) -> "KernelStore":
        """Load an existing store from disk."""
        storage_dir = Path(storage_dir)
        store = cls(storage_dir, **kwargs)

        graph_path = storage_dir / cls.GRAPH_FILENAME
        if graph_path.exists():
            store.graph = KernelGraph.load(str(graph_path))
            print(f"[store] Loaded graph: {store.graph.stats()}")
        else:
            print("[store] No existing graph found — starting fresh.")

        return store

    # ── ChromaDB ──────────────────────────────────────────────────────────────

    def _get_collection(self):
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed.\n"
                "Run: pip install chromadb"
            )

        self._chroma_client = chromadb.PersistentClient(
            path=str(self.storage_dir / "chroma")
        )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.CHROMA_COLLECTION,
            # COSINE distance: works with our normalized embeddings
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def index_nodes(
        self,
        nodes: list[CodeNode],
        batch_size: int = 128,
        resolve_edges: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Add CodeNodes to both the vector index and the knowledge graph.

        resolve_edges=True wires graph edges after all nodes are added.
        Set to False during incremental streaming — call store.graph.resolve_edges()
        manually at the end.
        """
        collection = self._get_collection()
        total = len(nodes)

        if verbose:
            print(f"[store] Indexing {total} nodes...")

        # Add to graph (fast — just Python dicts)
        self.graph.add_nodes(nodes)

        # Embed and add to ChromaDB in batches
        for i in range(0, total, batch_size):
            batch = nodes[i : i + batch_size]

            embeddings = self.embedder.embed_nodes(batch)

            collection.upsert(
                ids=[n.id for n in batch],
                embeddings=embeddings.tolist(),
                documents=[n.embedding_text() for n in batch],
                metadatas=[n.to_metadata() for n in batch],
            )

            if verbose:
                done = min(i + batch_size, total)
                print(f"  [{done}/{total}] embedded and indexed")

        if resolve_edges:
            if verbose:
                print("[store] Resolving graph edges...")
            edge_counts = self.graph.resolve_edges(verbose=verbose)
            if verbose:
                print(f"[store] Edge resolution complete: {edge_counts}")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        where: dict | None = None,
    ) -> list[VectorResult]:
        """
        Pure vector similarity search.

        where: ChromaDB metadata filter, e.g. {"node_type": "function"}
        or {"file_path": {"$contains": "kernel/sched"}}
        """
        collection = self._get_collection()
        q_embedding = self.embedder.embed_query(query)

        kwargs: dict[str, Any] = {
            "query_embeddings": [q_embedding.tolist()],
            "n_results": top_k,
            "include": ["metadatas", "distances", "documents"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)

        output = []
        for i, node_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # ChromaDB cosine distance ∈ [0, 2], convert to similarity ∈ [-1, 1]
            score = 1.0 - distance

            # Reconstruct CodeNode from metadata.
            # F-11: docstring is now stored in metadata and restored here.
            # The document field holds embedding_text() (docstring + header + code);
            # we use the raw metadata fields for each structured attribute.
            def _split(val: str) -> list[str]:
                return [x for x in val.split(",") if x] if val else []

            node = CodeNode(
                id=node_id,
                node_type=meta.get("node_type", ""),
                symbol_name=meta.get("symbol_name", ""),
                file_path=meta.get("file_path", ""),
                line_start=int(meta.get("line_start", 0)),
                line_end=int(meta.get("line_end", 0)),
                code=results["documents"][0][i],  # embedding_text() stored as document
                docstring=meta.get("docstring", ""),  # F-11: was missing
                calls=_split(meta.get("calls", "")),
                uses_structs=_split(meta.get("uses_structs", "")),
                includes=_split(meta.get("includes", "")),
            )
            output.append(VectorResult(node=node, score=score))

        return output

    def hybrid_search(
        self,
        query: str,
        top_k: int = 8,
        hops: int = 2,
        node_type_filter: str | None = None,
        subsystem_filter: str | None = None,
        max_context_nodes: int = 30,
    ) -> HybridResult:
        """
        The primary retrieval method.

        1. Vector search → top_k semantically similar CodeNodes (seeds)
        2. Graph expansion → structural neighborhood of seeds (context)
        3. Combine → HybridResult with primary hits + context

        node_type_filter: "function" | "struct" | "macro" | etc.
        subsystem_filter: e.g. "kernel/sched" to restrict by file path prefix
        """
        # Build ChromaDB filter
        where: dict | None = None
        filters = []
        if node_type_filter:
            filters.append({"node_type": {"$eq": node_type_filter}})
        if subsystem_filter:
            filters.append({"file_path": {"$contains": subsystem_filter}})

        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        # Step 1: Vector search
        vector_results = self.vector_search(query, top_k=top_k, where=where)

        if not vector_results:
            return HybridResult(
                primary=[],
                context=[],
                graph_ctx=GraphContext(seed_ids=[], neighbor_ids=[], subgraph=self.graph._g.subgraph([])),
            )

        # Step 2: Graph expansion from vector seeds
        seed_ids = [r.node.id for r in vector_results]
        graph_ctx = self.graph.neighborhood(
            seed_ids,
            hops=hops,
            max_nodes=max_context_nodes,
        )

        # Retrieve full CodeNodes for graph neighbors.
        # F-12: deduplicate against primary seed IDs so the `context` field
        # never contains a node that's already in `primary`.  The caller
        # (and HybridResult.all_nodes()) no longer has to guess whether a
        # context node duplicates a primary hit.
        primary_ids: set[str] = {r.node.id for r in vector_results}
        context_nodes: list[CodeNode] = []
        seen_context: set[str] = set()
        for node_id in graph_ctx.neighbor_ids:
            if node_id in primary_ids or node_id in seen_context:
                continue
            node = self.graph.get_node(node_id)
            if node:
                context_nodes.append(node)
                seen_context.add(node_id)

        return HybridResult(
            primary=vector_results,
            context=context_nodes,
            graph_ctx=graph_ctx,
        )

    # ── Filesystem X-Ray Support ───────────────────────────────────────────────

    def find_sysfs_handlers(self, sysfs_path: str) -> list[VectorResult]:
        """
        Given a /sys or /proc path, find the kernel code responsible for it.

        Strategy: decompose the path into tokens, embed a synthetic query,
        and filter to show_* / read_* / store_* functions which are the
        typical sysfs attribute handlers.
        """
        # Synthetic query: describe what we're looking for in natural language
        parts = sysfs_path.strip("/").split("/")
        attr_name = parts[-1].replace("_", " ")
        query = (
            f"sysfs attribute handler for {attr_name} "
            f"in {' '.join(parts[:-1])} subsystem, "
            f"show store read write function"
        )
        return self.vector_search(query, top_k=5, where={"node_type": {"$eq": "function"}})

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_graph(self) -> None:
        """Persist the knowledge graph to disk."""
        path = self.storage_dir / self.GRAPH_FILENAME
        self.graph.save(str(path))
        print(f"[store] Graph saved to {path}")

    def stats(self) -> dict:
        collection = self._get_collection()
        return {
            "vector_index": {
                "collection": self.CHROMA_COLLECTION,
                "count": collection.count(),
            },
            "graph": self.graph.stats(),
            "storage_dir": str(self.storage_dir),
            "embedder_model": self.embedder.model_name,
        }
