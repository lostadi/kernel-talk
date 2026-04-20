"""
tests/test_store.py
────────────────────
Phase 1 regression tests for core/mirror/store.py and core/mirror/parser.py

Covers:
  F-11  — docstring round-trip: index a CodeNode with a non-empty docstring,
           call vector_search, assert docstring is present on the result.
  F-12  — dedup: hybrid_search context field must not contain any node
           whose id already appears in the primary field.

These tests use a real ChromaDB PersistentClient in a tmp directory, but
stub out the embedding model so we don't need GPU / network access.

Run with:
    pytest tests/test_store.py -v
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock, patch
from core.mirror.parser import CodeNode
from core.mirror.store import KernelStore, HybridResult


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_node(name: str, path: str, docstring: str = "",
               calls: list[str] | None = None) -> CodeNode:
    return CodeNode(
        id=f"{path}::{name}",
        node_type="function",
        symbol_name=name,
        file_path=path,
        line_start=1,
        line_end=20,
        code=f"void {name}(void) {{ /* body */ }}",
        docstring=docstring,
        calls=calls or [],
    )


class _StubEmbedder:
    """
    Deterministic embedder that maps node_id → a fixed unit vector.
    Node 0 → e_0, Node 1 → e_1, etc. (orthogonal basis vectors).
    Query → e_0 so vector_search always returns Node 0 as the top hit.
    Dimension 64 is plenty for ChromaDB.
    """

    model_name = "stub"
    DIM = 64

    def __init__(self):
        self._id_to_idx: dict[str, int] = {}

    def _vec(self, idx: int) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        v[idx % self.DIM] = 1.0
        return v

    def embed_nodes(self, nodes) -> np.ndarray:
        out = []
        for node in nodes:
            if node.id not in self._id_to_idx:
                self._id_to_idx[node.id] = len(self._id_to_idx)
            out.append(self._vec(self._id_to_idx[node.id]))
        return np.stack(out)

    def embed_query(self, text: str) -> np.ndarray:
        # Query always aligns with index 0, so the first indexed node ranks #1
        return self._vec(0)

    def chunk_text(self, text: str, chunk_size: int = 512,
                   overlap: int = 64) -> list[str]:
        return [text]


def _make_store(tmp_path: str, nodes: list[CodeNode]) -> KernelStore:
    embedder = _StubEmbedder()
    store = KernelStore.create(tmp_path, embedder=embedder)
    store.index_nodes(nodes, verbose=False)
    return store


# ─── F-11: docstring round-trip ───────────────────────────────────────────────

class TestDocstringRoundTrip:
    """
    Index three nodes.  At least one has a non-empty docstring.
    After vector_search, the returned CodeNode must carry that docstring.
    """

    def test_docstring_survives_vector_search(self, tmp_path):
        nodes = [
            _make_node("schedule",       "kernel/sched/core.c",
                       docstring="/** schedule: put the current task to sleep */"),
            _make_node("do_fork",        "kernel/fork.c",
                       docstring="/** do_fork: create a child process */"),
            _make_node("kmalloc",        "mm/slab.c",
                       docstring=""),          # intentionally empty
        ]
        store = _make_store(str(tmp_path), nodes)

        results = store.vector_search("task scheduling", top_k=3)
        assert len(results) >= 1

        # Collect returned docstrings keyed by symbol_name
        by_name = {r.node.symbol_name: r.node.docstring for r in results}

        assert "/** schedule: put the current task to sleep */" in by_name.get("schedule", ""), \
            f"schedule docstring not round-tripped; got: {by_name.get('schedule')!r}"
        assert "/** do_fork: create a child process */" in by_name.get("do_fork", ""), \
            f"do_fork docstring not round-tripped; got: {by_name.get('do_fork')!r}"

    def test_empty_docstring_stays_empty(self, tmp_path):
        nodes = [
            _make_node("kmalloc", "mm/slab.c", docstring=""),
        ]
        store = _make_store(str(tmp_path), nodes)
        results = store.vector_search("memory allocation", top_k=1)
        assert results[0].node.docstring == ""

    def test_to_metadata_includes_docstring(self):
        """Unit test: to_metadata() must include the docstring key."""
        node = _make_node("schedule", "kernel/sched/core.c",
                          docstring="/** the scheduler */")
        meta = node.to_metadata()
        assert "docstring" in meta
        assert meta["docstring"] == "/** the scheduler */"

    def test_to_metadata_empty_docstring_is_string(self):
        """to_metadata() docstring must be a string (not None) when empty."""
        node = _make_node("kmalloc", "mm/slab.c", docstring="")
        meta = node.to_metadata()
        assert isinstance(meta["docstring"], str)


# ─── F-12: context deduplication ─────────────────────────────────────────────

class TestContextDedup:
    """
    hybrid_search must return a HybridResult where:
      1. context contains no node whose id is already in primary
      2. context has no duplicates within itself
    """

    def test_context_does_not_overlap_primary(self, tmp_path):
        # schedule calls do_nanosleep which calls schedule (cycle) — ensures
        # the graph neighborhood of "schedule" will include itself.
        nodes = [
            _make_node("schedule",     "kernel/sched/core.c",
                       calls=["do_nanosleep"],
                       docstring="/** schedule */"),
            _make_node("do_nanosleep", "kernel/time/hrtimer.c",
                       calls=["schedule"]),
            _make_node("pick_next_task", "kernel/sched/core.c",
                       calls=["schedule"]),
        ]
        store = _make_store(str(tmp_path), nodes)

        result = store.hybrid_search("task scheduling", top_k=2, hops=2)

        primary_ids = {r.node.id for r in result.primary}
        context_ids = [n.id for n in result.context]

        # No context node should duplicate a primary node
        overlap = primary_ids & set(context_ids)
        assert not overlap, \
            f"Context duplicates primary ids: {overlap}"

    def test_context_has_no_internal_duplicates(self, tmp_path):
        nodes = [
            _make_node("a", "kernel/a.c", calls=["b", "c"]),
            _make_node("b", "kernel/b.c", calls=["c"]),     # b and a both call c
            _make_node("c", "kernel/c.c"),
        ]
        store = _make_store(str(tmp_path), nodes)

        result = store.hybrid_search("function a", top_k=1, hops=2)
        context_ids = [n.id for n in result.context]

        assert len(context_ids) == len(set(context_ids)), \
            f"Duplicate context ids: {context_ids}"

    def test_all_nodes_deduped(self, tmp_path):
        """HybridResult.all_nodes() must also be free of duplicates."""
        nodes = [
            _make_node("schedule",     "kernel/sched/core.c",
                       calls=["pick_next_task"], docstring="/** schedule */"),
            _make_node("pick_next_task", "kernel/sched/core.c"),
        ]
        store = _make_store(str(tmp_path), nodes)

        result = store.hybrid_search("scheduling", top_k=2, hops=1)
        all_nodes = result.all_nodes()
        all_ids = [n.id for n in all_nodes]

        assert len(all_ids) == len(set(all_ids)), \
            f"all_nodes() contains duplicates: {all_ids}"
