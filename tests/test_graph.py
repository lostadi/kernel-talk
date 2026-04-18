"""
tests/test_graph.py
────────────────────
Phase 1 regression tests for core/mirror/graph.py

Covers:
  F-5 / F-8  — O(1) INCLUDES suffix index: resolve_edges must wire
               INCLUDES edges correctly and the suffix index must be
               populated on add_node.

Run with:
    pytest tests/test_graph.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from core.mirror.graph import KernelGraph, EdgeType
from core.mirror.parser import CodeNode


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _file_node(path: str, includes: list[str] | None = None) -> CodeNode:
    """Minimal file-type CodeNode."""
    return CodeNode(
        id=f"file::{path}",
        node_type="file",
        symbol_name=path,
        file_path=path,
        line_start=1,
        line_end=1,
        code="",
        includes=includes or [],
    )


def _func_node(name: str, path: str, calls: list[str] | None = None,
               uses_structs: list[str] | None = None) -> CodeNode:
    """Minimal function-type CodeNode."""
    return CodeNode(
        id=f"{path}::{name}",
        node_type="function",
        symbol_name=name,
        file_path=path,
        line_start=10,
        line_end=50,
        code=f"void {name}(void) {{}}",
        docstring=f"/** {name} docstring */",
        calls=calls or [],
        uses_structs=uses_structs or [],
    )


# ─── F-5 / F-8: Suffix index & INCLUDES resolution ───────────────────────────

class TestIncludesSuffixIndex:
    """
    Verify that _includes_suffix_index is populated correctly and that
    resolve_edges uses it to wire INCLUDES edges in O(1).
    """

    def test_suffix_index_populated_on_add_node(self):
        """All trailing suffixes of a file path must appear in the suffix index."""
        g = KernelGraph()
        node = _file_node("include/linux/sched.h")
        g.add_node(node)

        # All three suffix granularities must map to the node id
        assert node.id in g._includes_suffix_index["sched.h"]
        assert node.id in g._includes_suffix_index["linux/sched.h"]
        assert node.id in g._includes_suffix_index["include/linux/sched.h"]

    def test_includes_edge_wired_by_partial_path(self):
        """
        A file that includes "linux/sched.h" should get an INCLUDES edge to
        the file node at "include/linux/sched.h" — even though the stored
        path is longer than the include reference.
        """
        g = KernelGraph()
        header = _file_node("include/linux/sched.h")
        includer = _file_node("kernel/sched/core.c", includes=["linux/sched.h"])

        g.add_nodes([header, includer])
        counts = g.resolve_edges()

        assert counts.get(EdgeType.INCLUDES, 0) >= 1

        # Verify the actual edge exists in the graph
        assert g._g.has_edge(includer.id, header.id)
        edges = g._g.get_edge_data(includer.id, header.id)
        assert any(e.get("type") == EdgeType.INCLUDES for e in edges.values())

    def test_includes_edge_exact_path(self):
        """An include using the full relative path should also resolve."""
        g = KernelGraph()
        header = _file_node("include/linux/types.h")
        includer = _file_node("lib/string.c",
                               includes=["include/linux/types.h"])
        g.add_nodes([header, includer])
        g.resolve_edges()

        assert g._g.has_edge(includer.id, header.id)

    def test_no_self_loop_on_includes(self):
        """A file must never get an INCLUDES edge pointing to itself."""
        g = KernelGraph()
        node = _file_node("include/linux/sched.h",
                           includes=["linux/sched.h"])
        g.add_node(node)
        g.resolve_edges()

        # No self-loop
        assert not g._g.has_edge(node.id, node.id)

    def test_suffix_index_collision_both_wired(self):
        """
        Two different files sharing the same basename (e.g. config.h in
        different subdirs) must both appear in the suffix index under
        their unique full paths; only the basename entry is ambiguous.
        """
        g = KernelGraph()
        h1 = _file_node("arch/x86/include/asm/config.h")
        h2 = _file_node("include/linux/config.h")
        g.add_nodes([h1, h2])

        # Basename is ambiguous — both IDs must be present
        assert h1.id in g._includes_suffix_index["config.h"]
        assert h2.id in g._includes_suffix_index["config.h"]

        # Full paths are unambiguous
        assert g._includes_suffix_index["arch/x86/include/asm/config.h"] == [h1.id]
        assert g._includes_suffix_index["include/linux/config.h"] == [h2.id]

    def test_resolve_edges_no_spurious_includes(self):
        """Files with empty includes list must produce zero INCLUDES edges."""
        g = KernelGraph()
        g.add_node(_file_node("kernel/fork.c"))    # no includes
        g.add_node(_file_node("include/linux/sched.h"))
        counts = g.resolve_edges()

        assert counts.get(EdgeType.INCLUDES, 0) == 0

    def test_large_graph_includes_performance(self, benchmark=None):
        """
        Performance sanity check: 1000 file nodes each including 10 headers
        should resolve in well under 1 second (O(N) not O(N²)).
        """
        import time

        g = KernelGraph()
        headers = [_file_node(f"include/linux/h{i}.h") for i in range(200)]
        sources = [
            _file_node(
                f"drivers/net/d{i}.c",
                includes=[f"linux/h{j}.h" for j in range(i % 10, i % 10 + 10)]
            )
            for i in range(200)
        ]
        g.add_nodes(headers + sources)

        t0 = time.perf_counter()
        g.resolve_edges()
        elapsed = time.perf_counter() - t0

        # 400 nodes, 2000 include references — should be << 0.5 s
        assert elapsed < 0.5, f"resolve_edges took {elapsed:.3f}s — too slow"


# ─── CALLS / USES_STRUCT / DEFINED_IN edges (regression, not changed) ─────────

class TestSourceEdges:
    def test_calls_edge(self):
        g = KernelGraph()
        callee = _func_node("schedule", "kernel/sched/core.c")
        caller = _func_node("do_nanosleep", "kernel/time/hrtimer.c",
                             calls=["schedule"])
        g.add_nodes([callee, caller])
        counts = g.resolve_edges()

        assert counts.get(EdgeType.CALLS, 0) >= 1
        assert g._g.has_edge(caller.id, callee.id)

    def test_defined_in_edge(self):
        g = KernelGraph()
        f = _file_node("kernel/fork.c")
        fn = _func_node("copy_process", "kernel/fork.c")
        g.add_nodes([f, fn])
        counts = g.resolve_edges()

        assert counts.get(EdgeType.DEFINED_IN, 0) >= 1
        assert g._g.has_edge(fn.id, f.id)


# ─── stats() robustness ───────────────────────────────────────────────────────

class TestStats:
    def test_stats_with_layer2_node(self):
        """stats() must not crash when layer-2 (BinarySymbol-like) nodes exist."""
        g = KernelGraph()
        g.add_node(_func_node("schedule", "kernel/sched/core.c"))

        # Inject a fake layer-2 node whose 'data' is NOT a CodeNode
        from dataclasses import dataclass

        @dataclass
        class FakeBinSym:
            name: str = "schedule"
            addr_start: int = 0xffffffff81234000

        g._g.add_node("bin::schedule::0xffffffff81234000",
                       data=FakeBinSym(), layer=2)

        # Must not raise AttributeError
        s = g.stats()
        assert s["total_nodes"] == 2
        assert "FakeBinSym" in s["node_types"]

    def test_stats_with_layer3_dict_node(self):
        """stats() must handle layer-3 nodes whose data is a plain dict."""
        g = KernelGraph()
        g.add_node(_func_node("schedule", "kernel/sched/core.c"))
        g._g.add_node("live::schedule::0xffffffff81334000",
                       data={"name": "schedule", "live_addr": 0xffffffff81334000},
                       layer=3)

        s = g.stats()
        assert s["total_nodes"] == 2
        # Dict nodes fall back to "layer3_dict"
        assert "layer3_dict" in s["node_types"]
