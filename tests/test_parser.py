"""
tests/test_parser.py
────────────────────
Direct unit tests for KernelParser — the tree-sitter AST walker.

These tests verify that the tree-walking implementation (which replaced
the Query.matches() API removed in tree-sitter 0.25) extracts the correct
CodeNode shapes for every entity type and edge case.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.mirror.parser import KernelParser, CodeNode


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse(c_source: str) -> list[CodeNode]:
    """Parse a string of C source and return all extracted CodeNodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = Path(tmpdir) / "test.c"
        fpath.write_text(c_source)
        parser = KernelParser(tmpdir)
        return parser.parse_file(fpath)


def _by_type(nodes: list[CodeNode], node_type: str) -> list[CodeNode]:
    return [n for n in nodes if n.node_type == node_type]


# ─── Function extraction ──────────────────────────────────────────────────────

class TestFunctionExtraction:
    def test_plain_function(self):
        nodes = _parse("int schedule(void) { return 0; }")
        fns = _by_type(nodes, "function")
        assert len(fns) == 1
        assert fns[0].symbol_name == "schedule"
        assert fns[0].node_type == "function"

    def test_static_function(self):
        nodes = _parse("static int do_fork(int flags) { return flags; }")
        fns = _by_type(nodes, "function")
        assert any(f.symbol_name == "do_fork" for f in fns)

    def test_pointer_return_single(self):
        """int *kmalloc(size_t sz) — single-pointer return."""
        nodes = _parse("void *kmalloc(int sz) { return 0; }")
        fns = _by_type(nodes, "function")
        assert any(f.symbol_name == "kmalloc" for f in fns)

    def test_pointer_return_double(self):
        """void **get_ptrs(int n) — double-pointer return."""
        nodes = _parse("void **get_ptrs(int n) { return 0; }")
        fns = _by_type(nodes, "function")
        assert any(f.symbol_name == "get_ptrs" for f in fns)

    def test_function_calls_extracted(self):
        nodes = _parse(
            "void foo(void) { bar(); baz(1); }"
        )
        fns = _by_type(nodes, "function")
        assert len(fns) == 1
        assert "bar" in fns[0].calls
        assert "baz" in fns[0].calls

    def test_indirect_calls_not_extracted(self):
        """ops->fn() should not appear as a direct call."""
        nodes = _parse(
            "struct ops { void (*fn)(void); };\n"
            "void caller(struct ops *o) { o->fn(); }"
        )
        fns = _by_type(nodes, "function")
        caller = next(f for f in fns if f.symbol_name == "caller")
        # Indirect field calls should not show up as identifier calls
        assert "fn" not in caller.calls

    def test_line_numbers(self):
        src = "// line 1\nint foo(void) {\n    return 0;\n}"
        nodes = _parse(src)
        fns = _by_type(nodes, "function")
        assert fns[0].line_start == 2
        assert fns[0].line_end == 4

    def test_docstring_captured(self):
        src = (
            "/**\n"
            " * schedule - yield the CPU\n"
            " */\n"
            "void schedule(void) {}\n"
        )
        nodes = _parse(src)
        fns = _by_type(nodes, "function")
        assert "schedule" in fns[0].docstring

    def test_multiple_functions(self):
        src = "void a(void) {}\nvoid b(void) {}\nvoid c(void) {}"
        nodes = _parse(src)
        syms = {f.symbol_name for f in _by_type(nodes, "function")}
        assert syms == {"a", "b", "c"}


# ─── Struct extraction ────────────────────────────────────────────────────────

class TestStructExtraction:
    def test_simple_struct(self):
        nodes = _parse("struct task_struct { int pid; };")
        structs = _by_type(nodes, "struct")
        assert len(structs) == 1
        assert structs[0].symbol_name == "task_struct"

    def test_forward_declaration_skipped(self):
        """struct foo; — forward declaration has no body, must be skipped."""
        nodes = _parse("struct foo;\nvoid bar(struct foo *p) {}")
        structs = _by_type(nodes, "struct")
        assert len(structs) == 0

    def test_struct_param_usage_skipped(self):
        """struct task_struct *ptr — usage in param, no body."""
        nodes = _parse("void f(struct task_struct *p) { return; }")
        structs = _by_type(nodes, "struct")
        assert len(structs) == 0

    def test_struct_registers_for_ref(self):
        """A struct defined in the same file should appear in function struct refs."""
        src = (
            "struct my_custom { int x; };\n"
            "void user(struct my_custom *p) { p->x = 1; }\n"
        )
        nodes = _parse(src)
        fns = _by_type(nodes, "function")
        assert len(fns) == 1
        assert "my_custom" in fns[0].uses_structs

    def test_struct_defined_after_function(self):
        """
        Struct defined AFTER the function in the file — because parse_file()
        extracts structs first (pre-pass), the function's struct refs should
        still find it.
        """
        src = (
            "void user(struct late_struct *p) { p->x = 1; }\n"
            "struct late_struct { int x; };\n"
        )
        nodes = _parse(src)
        fns = _by_type(nodes, "function")
        assert len(fns) == 1
        assert "late_struct" in fns[0].uses_structs

    def test_core_struct_ref(self):
        """Functions referencing built-in CORE_STRUCTS are detected."""
        src = "void f(struct task_struct *t) { return; }"
        nodes = _parse(src)
        fns = _by_type(nodes, "function")
        assert "task_struct" in fns[0].uses_structs


# ─── Union extraction ─────────────────────────────────────────────────────────

class TestUnionExtraction:
    def test_simple_union(self):
        nodes = _parse("union ktime { long tv64; };")
        unions = _by_type(nodes, "union")
        assert len(unions) == 1
        assert unions[0].symbol_name == "ktime"

    def test_union_no_body_skipped(self):
        nodes = _parse("void f(union ktime *k) {}")
        unions = _by_type(nodes, "union")
        assert len(unions) == 0


# ─── Enum extraction ──────────────────────────────────────────────────────────

class TestEnumExtraction:
    def test_simple_enum(self):
        nodes = _parse("enum task_state { RUNNING = 0, SLEEPING = 1 };")
        enums = _by_type(nodes, "enum")
        assert len(enums) == 1
        assert enums[0].symbol_name == "task_state"

    def test_enum_without_body_skipped(self):
        """Enum usage in code without a definition body."""
        nodes = _parse("void f(enum my_state s) {}")
        enums = _by_type(nodes, "enum")
        assert len(enums) == 0


# ─── Macro extraction ────────────────────────────────────────────────────────

class TestMacroExtraction:
    def test_trivial_numeric_macro_skipped(self):
        """#define FOO 5 — too short, pure number."""
        nodes = _parse("#define FOO 5")
        macros = _by_type(nodes, "macro")
        assert len(macros) == 0

    def test_nontrivial_macro_extracted(self):
        nodes = _parse("#define KERN_ERR  \"<3>\"")
        macros = _by_type(nodes, "macro")
        assert any(m.symbol_name == "KERN_ERR" for m in macros)

    def test_function_macro_extracted(self):
        nodes = _parse("#define NICE_TO_PRIO(nice) (MAX_PRIO - 20 - (nice))")
        macros = _by_type(nodes, "macro")
        assert any(m.symbol_name == "NICE_TO_PRIO" for m in macros)

    def test_multiword_macro_extracted(self):
        nodes = _parse("#define TASK_RUNNING_MASK (0x01 | 0x02)")
        macros = _by_type(nodes, "macro")
        assert any(m.symbol_name == "TASK_RUNNING_MASK" for m in macros)


# ─── Include extraction ───────────────────────────────────────────────────────

class TestIncludeExtraction:
    def test_angle_bracket_include(self):
        nodes = _parse('#include <linux/sched.h>')
        file_nodes = _by_type(nodes, "file")
        assert any("linux/sched.h" in n.includes for n in file_nodes)

    def test_quoted_include(self):
        nodes = _parse('#include "local.h"')
        file_nodes = _by_type(nodes, "file")
        assert any("local.h" in n.includes for n in file_nodes)

    def test_no_includes_no_file_node(self):
        nodes = _parse("int x = 0;")
        file_nodes = _by_type(nodes, "file")
        assert len(file_nodes) == 0


# ─── Node ID uniqueness ───────────────────────────────────────────────────────

class TestNodeIDs:
    def test_ids_are_unique_within_file(self):
        src = (
            "void a(void) {}\n"
            "void b(void) {}\n"
            "struct s { int x; };\n"
            "enum e { V = 100 };\n"
        )
        nodes = _parse(src)
        ids = [n.id for n in nodes]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_id_format(self):
        nodes = _parse("void my_func(void) {}")
        fns = _by_type(nodes, "function")
        assert fns[0].id.endswith("::my_func")


# ─── CORE_STRUCTS isolation ───────────────────────────────────────────────────

class TestCoreStructsIsolation:
    def test_class_frozenset_not_mutated(self):
        """Parsing a file that defines a new struct must NOT mutate the class-level frozenset."""
        original = frozenset(KernelParser.CORE_STRUCTS)
        _parse("struct my_brand_new_struct { int x; };")
        assert KernelParser.CORE_STRUCTS == original, (
            "CORE_STRUCTS class constant was mutated by parse_file()!"
        )

    def test_instances_isolated(self):
        """Two parser instances must not share discovered struct names."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.c"
            p.write_text("struct only_in_a { int x; };")
            parser1 = KernelParser(d)
            parser1.parse_file(p)

            parser2 = KernelParser(d)
            assert "only_in_a" not in parser2._known_structs
