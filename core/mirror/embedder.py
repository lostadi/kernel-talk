"""
core/mirror/embedder.py
───────────────────────
The Mirror — Stage 3: Code Embeddings

We need a model that *understands C*, not just English prose.
General-purpose text embedders (sentence-transformers, ada-002) will work,
but they've never seen a `struct task_struct` before. Code-specific models
have been pretrained on GitHub repositories and understand the semantic
relationship between, say, `kmalloc` and memory management.

Model choice rationale:
  microsoft/codebert-base   — Trained on code+docstring pairs from GitHub.
                              Good at matching natural language queries to code.
                              512 token limit, 768-dim output.
  Salesforce/codet5p-110m-embedding — Lighter, faster. 256-dim output.
  nomic-ai/nomic-embed-text-v1 — Strong general embedder, no code specialization.

Default: codebert-base. It was trained on exactly our use case: "given a docstring,
find the code" — which is structurally identical to "given a question, find the function."

Batching: kernel indexing means hundreds of thousands of nodes. We batch
to saturate GPU throughput and avoid OOM. Default batch_size=32.

Output: normalized embeddings (unit sphere). This makes cosine similarity
equivalent to dot product — ChromaDB defaults to L2, so we either normalize
or configure it to use cosine. We normalize.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .parser import CodeNode

# Lazy imports — don't pay the HuggingFace loading cost until first use
_sentence_transformers_loaded = False
_SentenceTransformer = None


def _get_sentence_transformer():
    global _sentence_transformers_loaded, _SentenceTransformer
    if not _sentence_transformers_loaded:
        try:
            from sentence_transformers import SentenceTransformer
            _SentenceTransformer = SentenceTransformer
            _sentence_transformers_loaded = True
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            )
    return _SentenceTransformer


class CodeEmbedder:
    """
    Wraps a HuggingFace sentence-transformer model for CodeNode embedding.

    The key design decision: we embed the *embedding_text()* of each CodeNode,
    which concatenates the docstring + a type/path prefix + raw code.
    This is richer than embedding raw code alone because:

    1. Kernel docstrings capture *intent* — the 'why' behind the implementation.
    2. The type/path prefix helps the model localize: "function schedule in kernel/sched/core.c"
       encodes subsystem-level context that disambiguates common names.

    At query time, we embed the user's natural language question with the
    same model. The semantic bridge between "why is my CPU usage spiking"
    and `void scheduler_tick(void)` is what CodeBERT was trained to build.
    """

    DEFAULT_MODEL = "microsoft/codebert-base"
    # Alternative: "Salesforce/codet5p-110m-embedding" (faster, smaller)
    # Alternative: "nomic-ai/nomic-embed-text-v1.5" (general, slower)

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str | None = None,  # None = auto-detect (CUDA > MPS > CPU)
        normalize: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.progress_callback = progress_callback

        # Lazy load — only instantiate when first called
        self._model = None
        self._device = device
        self._dim: int | None = None

    def _load(self):
        if self._model is not None:
            return

        SentenceTransformer = _get_sentence_transformer()
        device = self._device or _auto_device()
        self._model = SentenceTransformer(self.model_name, device=device)
        # Get embedding dimension from a dummy forward pass
        self._dim = len(self._model.encode(["dim_probe"], show_progress_bar=False)[0])

    @property
    def dim(self) -> int:
        """Embedding dimensionality. Triggers model load on first access."""
        self._load()
        return self._dim

    def embed_nodes(self, nodes: list[CodeNode]) -> np.ndarray:
        """
        Embed a list of CodeNodes.

        Returns shape (N, D) float32 array.
        If normalize=True, each row has L2 norm = 1.0.

        F-9 / F-2: CodeBERT has a 512-token context window (~2 KB of code).
        For nodes whose embedding_text() exceeds that limit, we:
          1. Split into overlapping chunks via chunk_text()
          2. Embed each chunk separately
          3. Mean-pool the chunk embeddings back to a single vector
          4. Renormalize if normalize=True

        This ensures that even 500-line kernel functions get faithful
        embeddings rather than having their bodies silently truncated.
        Mean-pooling is the standard approach for long documents in the
        information retrieval literature (Dense Passage Retrieval, etc.).
        """
        self._load()

        # Rough threshold: CodeBERT 512 tokens ≈ 1800 chars for C code
        _CHUNK_THRESHOLD_CHARS = 1800

        per_node_texts: list[list[str]] = []
        for node in nodes:
            text = node.embedding_text()
            if len(text) <= _CHUNK_THRESHOLD_CHARS:
                per_node_texts.append([text])
            else:
                chunks = chunk_text(text, max_tokens=450, overlap=50)
                per_node_texts.append(chunks if chunks else [text])

        # Flatten all chunks into one batch for efficient GPU utilization
        flat_texts: list[str] = []
        node_chunk_counts: list[int] = []
        for chunks in per_node_texts:
            flat_texts.extend(chunks)
            node_chunk_counts.append(len(chunks))

        flat_embeddings = self._embed_texts(flat_texts)   # shape (total_chunks, D)

        # Reassemble: for each node, mean-pool its chunk embeddings
        result_embeddings = []
        offset = 0
        for count in node_chunk_counts:
            chunk_block = flat_embeddings[offset : offset + count]   # (count, D)
            offset += count
            if count == 1:
                result_embeddings.append(chunk_block[0])
            else:
                mean_emb = chunk_block.mean(axis=0)
                if self.normalize:
                    norm = np.linalg.norm(mean_emb)
                    if norm > 0.0:
                        mean_emb = mean_emb / norm
                result_embeddings.append(mean_emb)

        return np.stack(result_embeddings).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a natural language query for retrieval.

        Returns shape (D,) float32 array (1D, already normalized if normalize=True).

        Note: we prepend "query: " — some models (e.g. E5, nomic) are
        trained with task prefixes. CodeBERT doesn't need it, but it
        doesn't hurt to have a consistent convention.
        """
        self._load()
        text = f"query: {query}"
        result = self._embed_texts([text])
        return result[0]

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Core batched embedding loop."""
        all_embeddings = []
        n_batches = math.ceil(len(texts) / self.batch_size)

        for i in range(n_batches):
            batch = texts[i * self.batch_size : (i + 1) * self.batch_size]
            embeddings = self._model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
            all_embeddings.append(embeddings)

            if self.progress_callback:
                done = min((i + 1) * self.batch_size, len(texts))
                self.progress_callback(done, len(texts))

        return np.vstack(all_embeddings).astype(np.float32)

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two embedding vectors.
        If normalize=True, this is equivalent to dot product.
        """
        if self.normalize:
            return float(np.dot(a, b))
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def top_k_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Brute-force top-k similarity search over an in-memory corpus.
        Only use this for small corpora (< 10k). For large corpora, use ChromaDB.

        Returns list of (index, score) sorted by descending score.
        """
        scores = corpus_embeddings @ query_embedding  # (N,) dot products
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices]


# ─── Utilities ────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    """Pick the best available device for inference."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def chunk_text(text: str, max_tokens: int = 450, overlap: int = 50) -> list[str]:
    """
    Fallback chunker for very long code blocks (> 512 tokens for CodeBERT).

    We split on line boundaries rather than arbitrary token positions
    to avoid cutting function signatures from their bodies.

    Overlap ensures continuity across chunk boundaries — the synthesizer
    won't lose context at chunk edges.
    """
    lines = text.split("\n")
    # Rough token estimate: 1 token ≈ 4 chars for code
    chars_per_chunk = max_tokens * 4

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chars_per_chunk, len(text))
        # Walk back to a newline if we're mid-line
        if end < len(text):
            newline_pos = text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos
        chunks.append(text[start:end])
        if end >= len(text):
            break  # reached end — stop before the overlap steps backward
        next_start = end - (overlap * 4)  # overlap in characters
        if next_start <= start:
            next_start = end  # guard: always move forward
        start = next_start

    return chunks
