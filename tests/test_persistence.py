"""
tests/test_persistence.py
──────────────────────────
Phase 1 regression tests for graph persistence (F-6) and version pairing (F-13).

F-6  — save() must not crash or silently drop Layer 1 nodes when the graph
        also contains Layer 2 (BinarySymbol-like) or Layer 3 (dict) nodes
        added by link_dwarf() / link_kallsyms().

F-6b — load() must correctly repopulate all secondary indexes
        (_symbol_index, _file_index, _includes_suffix_index) so that
        graph queries like find_by_symbol() work after a round-trip.

F-13 — The saved graph must carry a schema_version attribute so that
        KernelGraph.load() can detect and reject stale saves.

These tests must FAIL against the code before the fix and PASS after.
Run with:
    pytest tests/test_persistence.py -v
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from dataclasses import dataclass
from pathlib import Path

from core.mirror.graph import KernelGraph, EdgeType
from core.mirror.parser import CodeNode


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _func_node(name: str, path: str, calls: list[str] | None = None) -> CodeNode:
    return CodeNode(
        id=f"{path}::{name}",
        node_type="function",
        symbol_name=name,
        file_path=path,
        line_start=10,
        line_end=50,
        code=f"void {name}(void) {{}}",
        docstring=f"/** {name} doc */",
        calls=calls or [],
    )


def _file_node(path: str) -> CodeNode:
    return CodeNode(
        id=f"file::{path}",
        node_type="file",
        symbol_name=path,
        file_path=path,
        line_start=1, line_end=1, code="",
    )


@dataclass
class _FakeBinarySymbol:
    """Simulates a BinarySymbol as added by link_dwarf()."""
    name: str
    addr_start: int
    addr_end: int
    addr_start_hex: str
    section: str = ".text"


def _make_enriched_graph() -> KernelGraph:
    """
    Build a KernelGraph that contains all three node categories:
      Layer 1: two CodeNodes (schedule, pick_next_task)
      Layer 2: a BinarySymbol node (as added by link_dwarf)
      Layer 3: a live-address dict node (as added by link_kallsyms)
    """
    g = KernelGraph()

    # Layer 1
    schedule = _func_node("schedule", "kernel/sched/core.c", calls=["pick_next_task"])
    pick = _func_node("pick_next_task", "kernel/sched/core.c")
    g.add_nodes([schedule, pick])
    g.resolve_edges()

    # Layer 2: inject a BinarySymbol node (normally done by link_dwarf)
    bsym = _FakeBinarySymbol(
        name="schedule",
        addr_start=0xffffffff81234000,
        addr_end=0xffffffff81234200,
        addr_start_hex="0xffffffff81234000",
    )
    bin_id = f"bin::schedule::0xffffffff81234000"
    g._g.add_node(bin_id, data=bsym, layer=2)
    g._g.add_edge(
        schedule.id, bin_id,
        type=EdgeType.SOURCE_TO_BINARY,
        dwarf_addr_start=bsym.addr_start,
        dwarf_addr_end=bsym.addr_end,
        section=bsym.section,
    )

    # Layer 3: inject a live-address dict node (normally done by link_kallsyms)
    live_addr = 0xffffffff81334000
    live_id = f"live::schedule::0x{live_addr:016x}"
    g._g.add_node(live_id, layer=3, data={
        "name": "schedule",
        "live_addr": live_addr,
        "live_addr_hex": f"0x{live_addr:016x}",
        "verified": True,
    })
    g._g.add_edge(
        bin_id, live_id,
        type=EdgeType.BINARY_TO_LIVE,
        kaslr_slide=0x100000,
        live_addr=live_addr,
        verified=True,
    )

    return g


# ─── F-6: save() robustness with mixed-layer graph ────────────────────────────

class TestSaveRobustness:
    """
    F-6: save() must not crash or corrupt Layer 1 data when the graph
    also contains Layer 2/3 nodes that don't have a to_metadata() method.
    """

    def test_save_does_not_crash_with_mixed_layers(self, tmp_path):
        """
        Core F-6 crash: calling save() on an enriched graph (with
        BinarySymbol and dict nodes from link_dwarf/link_kallsyms)
        must not raise AttributeError.
        """
        g = _make_enriched_graph()
        save_path = str(tmp_path / "graph.graphml")
        # This MUST NOT raise AttributeError from BinarySymbol.to_metadata()
        g.save(save_path)
        assert Path(save_path).exists()

    def test_saved_file_is_valid_graphml(self, tmp_path):
        """The saved file must be parseable by NetworkX."""
        import networkx as nx
        g = _make_enriched_graph()
        save_path = str(tmp_path / "graph.graphml")
        g.save(save_path)
        loaded_raw = nx.read_graphml(save_path)
        assert loaded_raw is not None

    def test_save_pure_layer1_graph_unchanged(self, tmp_path):
        """
        A plain Layer 1 graph (no link_dwarf/link_kallsyms) must round-trip
        perfectly — the enriched-graph fix must not break the normal path.
        """
        g = KernelGraph()
        schedule = _func_node("schedule", "kernel/sched/core.c", calls=["__schedule"])
        __schedule = _func_node("__schedule", "kernel/sched/core.c")
        g.add_nodes([schedule, __schedule])
        g.resolve_edges()

        save_path = str(tmp_path / "l1.graphml")
        g.save(save_path)

        g2 = KernelGraph.load(save_path)
        assert g2.stats()["total_nodes"] >= 2
        nodes = g2.find_by_symbol("schedule")
        assert len(nodes) == 1
        assert nodes[0].symbol_name == "schedule"


# ─── F-6b: load() repopulates all secondary indexes ──────────────────────────

class TestLoadIndexRepopulation:
    """
    After save() + load(), all secondary indexes must be live so that
    graph queries work correctly on the loaded graph.
    """

    def test_symbol_index_populated_after_load(self, tmp_path):
        g = KernelGraph()
        g.add_nodes([
            _func_node("schedule", "kernel/sched/core.c"),
            _func_node("do_fork", "kernel/fork.c"),
        ])
        g.resolve_edges()
        p = str(tmp_path / "g.graphml")
        g.save(p)

        g2 = KernelGraph.load(p)
        assert g2.find_by_symbol("schedule"), "schedule missing from loaded symbol index"
        assert g2.find_by_symbol("do_fork"), "do_fork missing from loaded symbol index"

    def test_file_index_populated_after_load(self, tmp_path):
        g = KernelGraph()
        fn = _func_node("schedule", "kernel/sched/core.c")
        fi = _file_node("kernel/sched/core.c")
        g.add_nodes([fn, fi])
        g.resolve_edges()
        p = str(tmp_path / "g.graphml")
        g.save(p)

        g2 = KernelGraph.load(p)
        assert "kernel/sched/core.c" in g2._file_index, \
            "file_index not rebuilt after load"

    def test_calls_edge_survives_round_trip(self, tmp_path):
        g = KernelGraph()
        schedule = _func_node("schedule", "kernel/sched/core.c",
                               calls=["__schedule"])
        __schedule = _func_node("__schedule", "kernel/sched/core.c")
        g.add_nodes([schedule, __schedule])
        g.resolve_edges()
        p = str(tmp_path / "g.graphml")
        g.save(p)

        g2 = KernelGraph.load(p)
        callees = g2.callees_of("schedule")
        assert any(c.symbol_name == "__schedule" for c in callees), \
            "CALLS edge not preserved across save/load"

    def test_includes_suffix_index_populated_after_load(self, tmp_path):
        g = KernelGraph()
        header = CodeNode(
            id="file::include/linux/sched.h",
            node_type="file", symbol_name="include/linux/sched.h",
            file_path="include/linux/sched.h",
            line_start=1, line_end=1, code="",
        )
        g.add_node(header)
        p = str(tmp_path / "g.graphml")
        g.save(p)

        g2 = KernelGraph.load(p)
        assert "sched.h" in g2._includes_suffix_index, \
            "includes_suffix_index not rebuilt after load"
        assert header.id in g2._includes_suffix_index["sched.h"]


# ─── F-13: schema version pairing ────────────────────────────────────────────

class TestSchemaVersion:
    """
    F-13: The saved graph must carry a schema_version so that future
    loads can detect and reject stale serializations.
    """

    def test_save_writes_schema_version(self, tmp_path):
        """A saved graph must have a schema_version attribute accessible via load."""
        g = KernelGraph()
        g.add_node(_func_node("schedule", "kernel/sched/core.c"))
        p = str(tmp_path / "g.graphml")
        g.save(p)
        g2 = KernelGraph.load(p)
        assert hasattr(g2, "_schema_version"), \
            "Loaded graph missing _schema_version attribute"
        assert isinstance(g2._schema_version, int) and g2._schema_version > 0, \
            f"schema_version must be a positive int, got {g2._schema_version!r}"

    def test_load_current_version_succeeds(self, tmp_path):
        """Loading a graph saved by the current code must not raise."""
        g = KernelGraph()
        g.add_node(_func_node("schedule", "kernel/sched/core.c"))
        p = str(tmp_path / "g.graphml")
        g.save(p)
        # Should not raise
        KernelGraph.load(p)

    def test_schema_version_is_stable(self, tmp_path):
        """Two separate save/load cycles must produce the same version number."""
        g = KernelGraph()
        g.add_node(_func_node("schedule", "kernel/sched/core.c"))
        p = str(tmp_path / "g.graphml")
        g.save(p)
        v1 = KernelGraph.load(p)._schema_version
        g.save(p)
        v2 = KernelGraph.load(p)._schema_version
        assert v1 == v2

    def test_graph_stats_have_schema_version(self, tmp_path):
        """stats() dict must expose the schema_version."""
        g = KernelGraph()
        g.add_node(_func_node("schedule", "kernel/sched/core.c"))
        p = str(tmp_path / "g.graphml")
        g.save(p)
        g2 = KernelGraph.load(p)
        s = g2.stats()
        assert "schema_version" in s, "stats() missing schema_version key"
