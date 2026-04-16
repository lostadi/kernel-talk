"""
core/mirror/graph.py
─────────────────────
The Mirror — Stage 2: Knowledge Graph Construction

A flat vector index is *retrieval*, not *understanding*.
Vector similarity tells you "this function looks like what you asked about."
Graph traversal tells you "and here are the 12 things it depends on."

We need both. The graph encodes the *structural topology* of the kernel —
who calls whom, which functions touch which data structures, which files
include which headers. This is the relational skeleton that vector search
can't capture.

The graph spans ALL FOUR LAYERS of the Digital Twin stack:

  Layer 1  Source   node_type: function | struct | union | enum | macro | file
  Layer 2  Binary   node_type: binary_symbol
  Layer 3  Symbol   node_type: kallsym (live address from /proc/kallsyms)
  Layer 4  Memory   (live snapshots are not graph nodes — they're query results)

Edge types across layers:
  Within source:    CALLS | USES_STRUCT | DEFINED_IN | INCLUDES
  Source→Binary:    SOURCE_TO_BINARY  (C function → compiled address range)
  Binary→Symbol:    BINARY_TO_LIVE    (DWARF addr → KASLR-adjusted live addr)
  Source→Offset:    FIELD_TO_OFFSET   (struct field → byte offset in memory)

This gives us the full traversal path for any query:
  Vector hit (source) → SOURCE_TO_BINARY → binary addr
                      → BINARY_TO_LIVE   → live virtual addr
                      → /proc/kcore read → actual bytes
                      → FIELD_TO_OFFSET  → decoded field values

The graph is a MultiDiGraph (directed, multiple edges allowed between same nodes).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterator

import networkx as nx

from .parser import CodeNode


# ─── Edge Types (typed constants, not magic strings) ──────────────────────────

class EdgeType:
    # ── Layer 1: Source graph ──────────────────────────────────────────────
    CALLS        = "CALLS"           # function → function
    USES_STRUCT  = "USES_STRUCT"     # function → struct/union
    DEFINED_IN   = "DEFINED_IN"      # symbol → file
    INCLUDES     = "INCLUDES"        # file → file (header graph)
    RELATED_TO   = "RELATED_TO"      # soft co-occurrence edge (at query time)

    # ── Layer 1→2: Source to Binary ───────────────────────────────────────
    SOURCE_TO_BINARY = "SOURCE_TO_BINARY"
    # C source node → compiled BinarySymbol node
    # Edge carries: dwarf_addr_start, dwarf_addr_end, section
    # Created by: link_dwarf(dwarf_bridge)

    # ── Layer 2→3: Binary to Live ─────────────────────────────────────────
    BINARY_TO_LIVE = "BINARY_TO_LIVE"
    # BinarySymbol node → KallsymNode (live address)
    # Edge carries: kaslr_slide, live_addr
    # Created by: link_kallsyms(kallsyms_bridge, dwarf_bridge)

    # ── Layer 1: Struct layout ────────────────────────────────────────────
    FIELD_TO_OFFSET = "FIELD_TO_OFFSET"
    # struct CodeNode → field CodeNode (or synthetic field node)
    # Edge carries: byte_offset, byte_size, c_type
    # Created by: link_struct_layouts(dwarf_bridge)


# ─── Query Result ─────────────────────────────────────────────────────────────

@dataclass
class GraphContext:
    """
    What the graph retriever hands to the synthesizer.
    seed_ids:    The nodes we started from (vector search results)
    neighbor_ids: Nodes reachable within k hops — the structural context
    subgraph:    The actual induced subgraph for further analysis
    """
    seed_ids: list[str]
    neighbor_ids: list[str]
    subgraph: nx.MultiDiGraph

    @property
    def all_ids(self) -> list[str]:
        return list(dict.fromkeys(self.seed_ids + self.neighbor_ids))


# ─── Knowledge Graph ──────────────────────────────────────────────────────────

class KernelGraph:
    """
    The structural backbone of Kernel-Talk.

    Internally a NetworkX MultiDiGraph where each node is a CodeNode.id
    and carries the full CodeNode as a 'data' attribute.

    The graph is built incrementally — you can add nodes one at a time
    (during streaming ingestion) or in bulk (after a full parse run).
    Edges are resolved lazily after all nodes are loaded, because a function
    in sched/core.c might call something defined in mm/slab.c — we need to
    see both sides before we can wire the edge.
    """

    def __init__(self):
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

        # Index structures for fast edge resolution
        # symbol_name → list of node IDs that define it
        self._symbol_index: dict[str, list[str]] = defaultdict(list)
        # file_path → node ID of the file node
        self._file_index: dict[str, str] = {}
        # Track which nodes have been edge-resolved
        self._resolved: set[str] = set()

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_node(self, node: CodeNode) -> None:
        """Add a single CodeNode. Edges are NOT resolved here — call resolve() after bulk load."""
        self._g.add_node(node.id, data=node)

        # Update symbol index (multiple definitions can share a name — e.g., static funcs)
        self._symbol_index[node.symbol_name].append(node.id)

        if node.node_type == "file":
            self._file_index[node.file_path] = node.id

    def add_nodes(self, nodes: list[CodeNode]) -> None:
        """Bulk add CodeNodes."""
        for node in nodes:
            self.add_node(node)

    def resolve_edges(self, verbose: bool = False) -> dict[str, int]:
        """
        Wire all edges from node metadata.

        This is a second pass over all nodes — we need it to be separate
        from add_node() because cross-file references can't be resolved
        until both endpoints are in the graph.

        Returns a counter of how many edges were created per type.
        """
        edge_counts: dict[str, int] = defaultdict(int)

        for node_id, attrs in self._g.nodes(data=True):
            if node_id in self._resolved:
                continue

            node: CodeNode = attrs["data"]

            # DEFINED_IN: symbol → file
            if node.node_type != "file" and node.file_path in self._file_index:
                file_id = self._file_index[node.file_path]
                self._g.add_edge(node_id, file_id, type=EdgeType.DEFINED_IN)
                edge_counts[EdgeType.DEFINED_IN] += 1

            # CALLS: function → function
            for callee_name in node.calls:
                for callee_id in self._symbol_index.get(callee_name, []):
                    if callee_id != node_id:
                        self._g.add_edge(node_id, callee_id, type=EdgeType.CALLS)
                        edge_counts[EdgeType.CALLS] += 1

            # USES_STRUCT: function → struct/union
            for struct_name in node.uses_structs:
                for struct_id in self._symbol_index.get(struct_name, []):
                    if struct_id != node_id:
                        self._g.add_edge(node_id, struct_id, type=EdgeType.USES_STRUCT)
                        edge_counts[EdgeType.USES_STRUCT] += 1

            # INCLUDES: file → file (header dependency graph)
            if node.node_type == "file":
                for included_path in node.includes:
                    # Try to find the included file in the graph
                    # Linux headers can be referenced as "linux/sched.h"
                    for candidate_path in self._file_index:
                        if candidate_path.endswith(included_path):
                            included_id = self._file_index[candidate_path]
                            self._g.add_edge(node_id, included_id, type=EdgeType.INCLUDES)
                            edge_counts[EdgeType.INCLUDES] += 1
                            break

            self._resolved.add(node_id)

        if verbose:
            for etype, count in edge_counts.items():
                print(f"  {etype}: {count} edges")

        return dict(edge_counts)

    # ── Graph Traversal ───────────────────────────────────────────────────────

    def neighborhood(
        self,
        node_ids: list[str],
        hops: int = 2,
        edge_types: list[str] | None = None,
        max_nodes: int = 50,
    ) -> GraphContext:
        """
        Expand a set of seed nodes into their structural neighborhood.

        hops=1: direct neighbors only (callers, callees, used structs)
        hops=2: 2nd-order neighborhood (gives much richer context)

        edge_types: filter to specific edge types. None = all types.
        max_nodes: cap expansion to avoid context explosion.

        This is the graph retrieval step — vector search gives seeds,
        graph traversal gives the structural context around them.
        """
        # BFS expansion
        visited: set[str] = set()
        frontier: set[str] = set(n for n in node_ids if n in self._g)
        all_nodes: list[str] = []

        for hop in range(hops):
            next_frontier: set[str] = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)
                all_nodes.append(node_id)

                if len(all_nodes) >= max_nodes:
                    break

                # Explore both directions: successors AND predecessors
                # Why both? A function's callers are as relevant as its callees
                for neighbor in list(self._g.successors(node_id)) + list(self._g.predecessors(node_id)):
                    if neighbor not in visited:
                        # Filter by edge type if requested
                        if edge_types is not None:
                            edges = self._g.get_edge_data(node_id, neighbor) or {}
                            edges_rev = self._g.get_edge_data(neighbor, node_id) or {}
                            all_edges = {**edges, **edges_rev}
                            if not any(e.get("type") in edge_types for e in all_edges.values()):
                                continue
                        next_frontier.add(neighbor)

            if len(all_nodes) >= max_nodes:
                break
            frontier = next_frontier

        # Separate seeds from neighbors for the caller to know which are primary
        seed_set = set(node_ids)
        neighbor_ids = [n for n in all_nodes if n not in seed_set]
        valid_seeds = [n for n in node_ids if n in self._g]

        # Build induced subgraph
        subgraph = self._g.subgraph(all_nodes).copy()

        return GraphContext(
            seed_ids=valid_seeds,
            neighbor_ids=neighbor_ids,
            subgraph=subgraph,
        )

    def callers_of(self, symbol_name: str) -> list[CodeNode]:
        """Who calls this function? (inverse of CALLS edge)"""
        results = []
        for node_id in self._symbol_index.get(symbol_name, []):
            for caller_id in self._g.predecessors(node_id):
                edges = self._g.get_edge_data(caller_id, node_id)
                if edges and any(e.get("type") == EdgeType.CALLS for e in edges.values()):
                    results.append(self._g.nodes[caller_id]["data"])
        return results

    def callees_of(self, symbol_name: str) -> list[CodeNode]:
        """What does this function call? (CALLS edges)"""
        results = []
        for node_id in self._symbol_index.get(symbol_name, []):
            for callee_id in self._g.successors(node_id):
                edges = self._g.get_edge_data(node_id, callee_id)
                if edges and any(e.get("type") == EdgeType.CALLS for e in edges.values()):
                    results.append(self._g.nodes[callee_id]["data"])
        return results

    def struct_users(self, struct_name: str) -> list[CodeNode]:
        """Which functions use this struct?"""
        results = []
        for struct_id in self._symbol_index.get(struct_name, []):
            for user_id in self._g.predecessors(struct_id):
                edges = self._g.get_edge_data(user_id, struct_id)
                if edges and any(e.get("type") == EdgeType.USES_STRUCT for e in edges.values()):
                    results.append(self._g.nodes[user_id]["data"])
        return results

    def get_node(self, node_id: str) -> CodeNode | None:
        """Retrieve a CodeNode by its ID."""
        if node_id in self._g:
            return self._g.nodes[node_id].get("data")
        return None

    def find_by_symbol(self, symbol_name: str) -> list[CodeNode]:
        """Find all nodes with a given symbol name (may have multiple definitions)."""
        return [
            self._g.nodes[nid]["data"]
            for nid in self._symbol_index.get(symbol_name, [])
            if nid in self._g
        ]

    # ── Subsystem Analysis ────────────────────────────────────────────────────

    def subsystem_summary(self) -> dict[str, dict[str, int]]:
        """
        Group nodes by top-level kernel subsystem (kernel/, mm/, net/, fs/, etc.)
        and return counts per type.

        Useful for building the "Mandala Kernel" visualization.
        """
        summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for _, attrs in self._g.nodes(data=True):
            node: CodeNode = attrs["data"]
            parts = node.file_path.split("/")
            subsystem = parts[0] if parts else "root"
            summary[subsystem][node.node_type] += 1
        return {k: dict(v) for k, v in summary.items()}

    def most_connected(self, top_k: int = 20) -> list[tuple[str, int]]:
        """
        Return the top_k most connected nodes by total degree.
        These are the architectural hubs of the kernel — typically core
        scheduler functions, memory management routines, and core structs.
        """
        degrees = [(nid, self._g.degree(nid)) for nid in self._g.nodes]
        degrees.sort(key=lambda x: x[1], reverse=True)
        return [(self._g.nodes[nid]["data"].symbol_name, deg) for nid, deg in degrees[:top_k]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialize the graph to a GraphML file."""
        # GraphML can't handle complex objects, so we serialize CodeNode to metadata
        g_copy = nx.MultiDiGraph()
        for node_id, attrs in self._g.nodes(data=True):
            node: CodeNode = attrs["data"]
            g_copy.add_node(node_id, **node.to_metadata())
        for u, v, edge_attrs in self._g.edges(data=True):
            g_copy.add_edge(u, v, **edge_attrs)
        nx.write_graphml(g_copy, path)

    @classmethod
    def load(cls, path: str) -> "KernelGraph":
        """Load a previously saved graph. Note: CodeNode objects are reconstructed from metadata."""
        g = cls()
        loaded = nx.read_graphml(path)
        for node_id, attrs in loaded.nodes(data=True):
            node = CodeNode(
                id=node_id,
                node_type=attrs.get("node_type", "function"),
                symbol_name=attrs.get("symbol_name", ""),
                file_path=attrs.get("file_path", ""),
                line_start=int(attrs.get("line_start", 0)),
                line_end=int(attrs.get("line_end", 0)),
                code=attrs.get("code", ""),
                docstring=attrs.get("docstring", ""),
                calls=attrs.get("calls", "").split(",") if attrs.get("calls") else [],
                uses_structs=attrs.get("uses_structs", "").split(",") if attrs.get("uses_structs") else [],
                includes=attrs.get("includes", "").split(",") if attrs.get("includes") else [],
            )
            g._g.add_node(node_id, data=node)
            g._symbol_index[node.symbol_name].append(node_id)
            if node.node_type == "file":
                g._file_index[node.file_path] = node_id
        for u, v, edge_attrs in loaded.edges(data=True):
            g._g.add_edge(u, v, **edge_attrs)
        return g

    # ── Digital Twin Layer Linking ────────────────────────────────────────────
    #
    # These three methods build the cross-layer edges that make this a
    # true Digital Twin rather than a flat code search engine.
    # Call them in order after the source graph is built:
    #   1. link_dwarf()      — connects source nodes to binary addresses
    #   2. link_kallsyms()   — connects binary addresses to live addresses
    #   3. link_struct_layouts() — annotates struct edges with byte offsets

    def link_dwarf(self, dwarf_bridge, verbose: bool = True) -> dict[str, int]:
        """
        Layer 1→2: Connect source CodeNodes to their compiled BinarySymbols.

        For each function/struct/macro CodeNode in the graph, we look up
        its symbol name in the DWARF bridge and create a SOURCE_TO_BINARY edge
        pointing to a new BinarySymbol node.

        The BinarySymbol node carries: addr_start, addr_end, section,
        source_file (from DWARF, may differ from our AST parse), source_line.

        Returns count of edges created.
        """
        from ..dwarf.bridge import BinarySymbol

        if not dwarf_bridge._loaded:
            dwarf_bridge.load(verbose=verbose)

        edges_created = 0
        symbols_not_found = 0

        for node_id, attrs in list(self._g.nodes(data=True)):
            node: CodeNode = attrs["data"]
            if node.node_type not in ("function", "struct", "union", "enum", "macro"):
                continue

            # Look up in DWARF
            binary_syms = dwarf_bridge.symbol_to_addrs(node.symbol_name)
            if not binary_syms:
                symbols_not_found += 1
                continue

            for bsym in binary_syms:
                # Create a BinarySymbol node in the graph
                bin_node_id = f"bin::{bsym.name}::{bsym.addr_start_hex}"
                if bin_node_id not in self._g:
                    self._g.add_node(bin_node_id, data=bsym, layer=2)
                    self._symbol_index.setdefault(
                        f"bin::{bsym.name}", []
                    ).append(bin_node_id)

                # Create SOURCE_TO_BINARY edge
                self._g.add_edge(
                    node_id, bin_node_id,
                    type=EdgeType.SOURCE_TO_BINARY,
                    dwarf_addr_start=bsym.addr_start,
                    dwarf_addr_end=bsym.addr_end,
                    section=bsym.section,
                )
                edges_created += 1

        if verbose:
            print(f"[graph] DWARF linking: {edges_created} SOURCE_TO_BINARY edges, "
                  f"{symbols_not_found} symbols not found in DWARF")

        return {"SOURCE_TO_BINARY": edges_created, "not_found": symbols_not_found}

    def link_kallsyms(self, kallsyms_bridge, dwarf_bridge, verbose: bool = True) -> dict[str, int]:
        """
        Layer 2→3: Connect BinarySymbol nodes to live /proc/kallsyms addresses.

        Computes the KASLR slide (dwarf_addr → live_addr offset) and creates
        BINARY_TO_LIVE edges from each BinarySymbol node to a synthetic
        KallsymNode carrying the live virtual address.

        After this, every source CodeNode has a path:
          source → SOURCE_TO_BINARY → binary → BINARY_TO_LIVE → live_addr

        Returns counts of edges and the computed KASLR slide.
        """
        if not kallsyms_bridge._loaded:
            kallsyms_bridge.load(verbose=verbose)

        # Compute KASLR slide
        slide = kallsyms_bridge.kaslr_slide(dwarf_bridge)
        if slide is None:
            if verbose:
                print("[graph] WARNING: Could not compute KASLR slide "
                      "(need root + DWARF loaded). Skipping live address linking.")
            return {"BINARY_TO_LIVE": 0, "kaslr_slide": None}

        if verbose:
            print(f"[graph] KASLR slide: 0x{slide:016x} ({slide:+d})")

        edges_created = 0

        # Walk all binary nodes and create BINARY_TO_LIVE edges
        for node_id, attrs in list(self._g.nodes(data=True)):
            if attrs.get("layer") != 2:
                continue

            bsym = attrs["data"]
            if not hasattr(bsym, "addr_start"):
                continue

            live_addr = bsym.addr_start + slide

            # Verify against kallsyms
            entry = kallsyms_bridge.symbol_entries(bsym.name)
            verified = any(e.address == live_addr for e in entry)

            live_node_id = f"live::{bsym.name}::0x{live_addr:016x}"
            if live_node_id not in self._g:
                self._g.add_node(live_node_id, layer=3, data={
                    "name": bsym.name,
                    "live_addr": live_addr,
                    "live_addr_hex": f"0x{live_addr:016x}",
                    "verified": verified,
                })

            self._g.add_edge(
                node_id, live_node_id,
                type=EdgeType.BINARY_TO_LIVE,
                kaslr_slide=slide,
                live_addr=live_addr,
                verified=verified,
            )
            edges_created += 1

        if verbose:
            print(f"[graph] Kallsyms linking: {edges_created} BINARY_TO_LIVE edges")

        return {"BINARY_TO_LIVE": edges_created, "kaslr_slide": slide}

    def link_struct_layouts(self, dwarf_bridge, verbose: bool = True) -> dict[str, int]:
        """
        Layer 1 enrichment: Annotate struct nodes with DWARF field offsets.

        For each struct/union CodeNode, looks up the StructLayout in DWARF
        and creates FIELD_TO_OFFSET edges from the struct node to synthetic
        field nodes. Each edge carries: byte_offset, byte_size, c_type.

        This enables raw memory decoding: given a pointer to any struct,
        we can read each field without needing drgn's type system.
        """
        if not dwarf_bridge._loaded:
            dwarf_bridge.load(verbose=verbose)

        edges_created = 0
        structs_linked = 0

        for node_id, attrs in list(self._g.nodes(data=True)):
            node: CodeNode = attrs["data"]
            if node.node_type not in ("struct", "union"):
                continue

            layout = dwarf_bridge.struct_layout(node.symbol_name)
            if not layout:
                continue

            structs_linked += 1

            # Store total size on the struct node
            self._g.nodes[node_id]["sizeof"] = layout.total_size

            for field_name, finfo in layout.fields.items():
                # Create a synthetic field node
                field_node_id = f"field::{node.symbol_name}::{field_name}"
                if field_node_id not in self._g:
                    self._g.add_node(field_node_id, layer=1, data={
                        "struct": node.symbol_name,
                        "field": field_name,
                        "byte_offset": finfo.byte_offset,
                        "byte_size": finfo.byte_size,
                        "c_type": finfo.c_type,
                    })

                self._g.add_edge(
                    node_id, field_node_id,
                    type=EdgeType.FIELD_TO_OFFSET,
                    byte_offset=finfo.byte_offset,
                    byte_size=finfo.byte_size,
                    c_type=finfo.c_type,
                )
                edges_created += 1

        if verbose:
            print(f"[graph] Struct layout linking: {structs_linked} structs, "
                  f"{edges_created} FIELD_TO_OFFSET edges")

        return {"FIELD_TO_OFFSET": edges_created, "structs_linked": structs_linked}

    def live_address_for(self, symbol_name: str) -> list[int]:
        """
        Get all live virtual addresses for a symbol name.
        Requires link_kallsyms() to have been called.

        Traversal: source_node → SOURCE_TO_BINARY → binary_node
                                                   → BINARY_TO_LIVE → live_addr
        """
        live_addrs = []
        for source_id in self._symbol_index.get(symbol_name, []):
            if source_id not in self._g:
                continue
            for bin_id in self._g.successors(source_id):
                edges = self._g.get_edge_data(source_id, bin_id) or {}
                if not any(e.get("type") == EdgeType.SOURCE_TO_BINARY
                           for e in edges.values()):
                    continue
                for live_id in self._g.successors(bin_id):
                    live_edges = self._g.get_edge_data(bin_id, live_id) or {}
                    for e in live_edges.values():
                        if e.get("type") == EdgeType.BINARY_TO_LIVE:
                            live_addrs.append(e["live_addr"])
        return live_addrs

    def struct_field_offset(self, struct_name: str, field_name: str) -> int | None:
        """
        Get the byte offset of a field within a struct.
        Requires link_struct_layouts() to have been called.
        Returns None if not found.
        """
        field_node_id = f"field::{struct_name}::{field_name}"
        if field_node_id in self._g:
            node_data = self._g.nodes[field_node_id].get("data", {})
            return node_data.get("byte_offset")
        return None

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        type_counts: dict[str, int] = defaultdict(int)
        for _, attrs in self._g.nodes(data=True):
            type_counts[attrs["data"].node_type] += 1

        return {
            "total_nodes": self._g.number_of_nodes(),
            "total_edges": self._g.number_of_edges(),
            "node_types":  dict(type_counts),
            "unique_symbols": len(self._symbol_index),
        }
