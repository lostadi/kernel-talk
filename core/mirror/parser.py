"""
core/mirror/parser.py
─────────────────────
The Mirror — Stage 1: Static AST Parsing

Naive text chunking destroys kernel semantics. A function like `schedule()`
spans maybe 80 lines, but its *meaning* is entangled with `struct task_struct`,
`struct rq`, and a dozen `#define` flags spread across a dozen headers.
We need AST-level understanding, not sliding windows over bytes.

tree-sitter gives us a full parse tree of C. We walk it to extract semantic
*units* — functions, structs, enums, macros — each as a self-contained CodeNode
with metadata that will become graph edges later.

Every CodeNode is the atomic unit of this system: a node in the knowledge graph
AND a document in the vector index. The same object carries both roles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# tree-sitter >= 0.25 API — Query.matches() was removed; we use direct tree-walking
try:
    from tree_sitter import Language, Parser, Node
    import tree_sitter_c as tsc
    C_LANGUAGE = Language(tsc.language())
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False
    C_LANGUAGE = None


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class CodeNode:
    """
    The atomic unit of the knowledge graph.

    id:          Globally unique — "{rel_path}::{symbol}" or "{rel_path}::L{line}"
    node_type:   One of VALID_TYPES
    symbol_name: The bare C identifier (e.g. "schedule", "task_struct")
    file_path:   Path relative to kernel source root (e.g. "kernel/sched/core.c")
    line_start:  1-indexed start line in the source file
    line_end:    1-indexed end line
    code:        Raw C source text of this entity
    docstring:   Preceding block comment (/** ... */) if present, else ""
    calls:       Function names called inside this node's body (for graph edges)
    uses_structs: Struct/union names referenced in this node (for graph edges)
    includes:    Files included by this file — populated for node_type="file"
    """
    id: str
    node_type: str          # "function" | "struct" | "union" | "enum" | "macro" | "file"
    symbol_name: str
    file_path: str
    line_start: int
    line_end: int
    code: str
    docstring: str = ""
    calls: list[str] = field(default_factory=list)
    uses_structs: list[str] = field(default_factory=list)
    includes: list[str] = field(default_factory=list)

    VALID_TYPES = {"function", "struct", "union", "enum", "macro", "file"}

    def embedding_text(self) -> str:
        """
        The text we actually embed. We prepend the docstring because
        kernel devs write really good block comments that capture intent
        (the 'why') while the code captures mechanism (the 'what').
        Fusing both gives the embedding far richer semantic signal.
        """
        parts = []
        if self.docstring:
            parts.append(self.docstring)
        parts.append(f"// {self.node_type}: {self.symbol_name} in {self.file_path}")
        parts.append(self.code)
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        """Flat dict for ChromaDB metadata (no nested structures).

        Note: docstring is stored here so vector_search can reconstruct it.
        ChromaDB metadata values must be str/int/float/bool — no lists or None.
        """
        return {
            "node_type":    self.node_type,
            "symbol_name":  self.symbol_name,
            "file_path":    self.file_path,
            "line_start":   self.line_start,
            "line_end":     self.line_end,
            "docstring":    self.docstring,          # F-11: was missing
            "calls":        ",".join(self.calls),
            "uses_structs": ",".join(self.uses_structs),
            "includes":     ",".join(self.includes),
        }


# ─── Parser ───────────────────────────────────────────────────────────────────

class KernelParser:
    """
    Parses Linux kernel C source files into CodeNode streams.

    We use direct AST tree-walking rather than tree-sitter Query objects.
    The Query.matches() API was removed in tree-sitter 0.25; tree-walking
    via Node.children and Node.child_by_field_name() is the stable interface.

    Usage:
        parser = KernelParser(kernel_root="/usr/src/linux")
        for node in parser.parse_directory("kernel/sched"):
            print(node.symbol_name, node.node_type)
    """

    # We skip generated files, stubs, and test fixtures — they add noise
    SKIP_PATTERNS = {
        ".mod.c", ".tmp.", "compat_ioctl", "/test/", "/tests/",
        "/.git/", "/tools/testing/",
    }

    # Class-level frozenset of well-known kernel struct names (immutable seed).
    # Each instance extends this into a mutable _known_structs set so that
    # struct names discovered while parsing a file are registered for later
    # cross-reference — without mutating the shared class constant.
    CORE_STRUCTS: frozenset[str] = frozenset({
        "task_struct", "mm_struct", "vm_area_struct", "file", "inode",
        "socket", "sk_buff", "net_device", "super_block", "dentry",
        "page", "address_space", "bio", "request", "wait_queue_head",
        "mutex", "spinlock_t", "rcu_head", "kobject", "device",
        "pci_dev", "platform_device", "irq_desc", "timer_list",
        "work_struct", "delayed_work", "completion", "semaphore",
        # sched subsystem
        "rq", "cfs_rq", "rt_rq", "sched_entity", "sched_class",
        # memory
        "pglist_data", "zone", "page_pool", "kmem_cache",
        # networking
        "sock", "tcp_sock", "inet_sock", "net", "net_proto_family",
        # block / storage
        "gendisk", "block_device", "elevator_queue",
        # locking
        "rwlock_t", "rwsem",
    })

    def __init__(self, kernel_root: str | Path, max_file_size_mb: float = 2.0):
        if not _TS_AVAILABLE:
            raise RuntimeError(
                "tree-sitter is not installed. Run: pip install tree-sitter tree-sitter-c"
            )
        self.kernel_root = Path(kernel_root)
        self.max_bytes = int(max_file_size_mb * 1024 * 1024)
        self._parser = Parser(C_LANGUAGE)
        # Instance-level mutable copy — grows as we discover struct names.
        # F-1 fix: was a class-level frozenset with .add() calls (AttributeError).
        self._known_structs: set[str] = set(self.CORE_STRUCTS)

    # ── Public Interface ───────────────────────────────────────────────────────

    def parse_file(self, path: Path) -> list[CodeNode]:
        """Parse a single .c or .h file into CodeNodes."""
        rel = str(path.relative_to(self.kernel_root))

        if path.stat().st_size > self.max_bytes:
            return []
        if any(skip in rel for skip in self.SKIP_PATTERNS):
            return []

        try:
            source_bytes = path.read_bytes()
        except (PermissionError, OSError):
            return []

        tree = self._parser.parse(source_bytes)
        source_text = source_bytes.decode("utf-8", errors="replace")
        lines = source_text.splitlines()

        nodes: list[CodeNode] = []

        # Extract includes first — they populate file-level metadata
        includes = self._extract_includes(tree.root_node, source_bytes, rel)

        # Structs/unions FIRST: registers names in _known_structs so that
        # functions parsed later in the same file see the correct struct refs.
        nodes.extend(self._extract_structs(tree.root_node, source_bytes, lines, rel, "struct"))
        nodes.extend(self._extract_structs(tree.root_node, source_bytes, lines, rel, "union"))

        # Functions, enums, macros
        nodes.extend(self._extract_functions(tree.root_node, source_bytes, lines, rel))
        nodes.extend(self._extract_enums(tree.root_node, source_bytes, lines, rel))
        nodes.extend(self._extract_macros(tree.root_node, source_bytes, lines, rel))

        # Emit a file-level node that carries the include graph edges
        if includes:
            file_node = CodeNode(
                id=f"{rel}::__file__",
                node_type="file",
                symbol_name=path.name,
                file_path=rel,
                line_start=1,
                line_end=len(lines),
                code=source_text[:512],  # first 512 chars as a summary
                includes=includes,
            )
            nodes.append(file_node)

        return nodes

    def parse_directory(
        self,
        subdir: str = "",
        extensions: tuple[str, ...] = (".c", ".h"),
    ) -> Iterator[CodeNode]:
        """
        Walk a subdirectory of the kernel source and yield CodeNodes.
        subdir="" means the whole tree (slow but complete).
        subdir="kernel/sched" means just the scheduler subsystem.
        """
        root = self.kernel_root / subdir if subdir else self.kernel_root
        for path in root.rglob("*"):
            if path.suffix in extensions and path.is_file():
                yield from self.parse_file(path)

    # ── Private Tree-Walking Helpers ───────────────────────────────────────────

    def _find_all(self, root: Node, node_type: str) -> list[Node]:
        """
        Iteratively find all descendant nodes of the given type.
        Uses an explicit stack to avoid Python recursion limits on deep trees.
        Returns nodes in document order (left-to-right, depth-first).
        """
        results: list[Node] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.type == node_type:
                results.append(node)
            # Push children in reverse so left-most child is processed first
            for child in reversed(node.children):
                stack.append(child)
        return results

    def _function_name_from_decl(
        self, decl: Node
    ) -> tuple["Node | None", "Node | None"]:
        """
        Navigate the declarator subtree to find (function_declarator, identifier).

        Handles multi-level pointer return types:
          int  schedule(void)    → function_declarator → identifier
          int *kmalloc(size_t)   → pointer_declarator  → function_declarator → identifier
          int **alloc(void)      → pointer_declarator  → pointer_declarator  → function_declarator → identifier
        """
        node = decl
        # Unwrap any pointer_declarator layers (multiple for **foo())
        while node is not None and node.type == "pointer_declarator":
            node = node.child_by_field_name("declarator")
        if node is None or node.type != "function_declarator":
            return None, None
        name_node = node.child_by_field_name("declarator")
        if name_node is None or name_node.type != "identifier":
            return None, None
        return node, name_node

    def _node_text(self, node: Node, source: bytes) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _preceding_comment(self, node: Node, lines: list[str]) -> str:
        """
        Grab the block comment immediately above a node, if any.
        Kernel devs write rich /** ... */ comments — we want them in the embedding.
        """
        start_line = node.start_point[0]  # 0-indexed
        comment_lines: list[str] = []
        i = start_line - 1
        while i >= 0:
            stripped = lines[i].strip()
            if stripped.startswith("*") or stripped.startswith("/*") or stripped == "*/":
                comment_lines.insert(0, lines[i])
                i -= 1
            elif stripped == "" and comment_lines:
                # Allow one blank line gap
                i -= 1
            else:
                break
        return "\n".join(comment_lines).strip()

    # ── Private Extraction Methods ─────────────────────────────────────────────

    def _extract_calls(self, body_node: Node, source: bytes) -> list[str]:
        """Extract direct function call identifiers from a body node."""
        calls: set[str] = set()
        for call_node in self._find_all(body_node, "call_expression"):
            fn_child = call_node.child_by_field_name("function")
            if fn_child and fn_child.type == "identifier":
                name = self._node_text(fn_child, source)
                if name and not name.startswith("__"):
                    calls.add(name)
        return sorted(calls)

    def _extract_struct_refs(self, node: Node, source: bytes) -> list[str]:
        """Extract struct/union type names referenced within a node."""
        refs: set[str] = set()
        for type_node in self._find_all(node, "type_identifier"):
            name = self._node_text(type_node, source)
            if name in self._known_structs:
                refs.add(name)
        return sorted(refs)

    def _extract_functions(
        self, root: Node, source: bytes, lines: list[str], rel_path: str
    ) -> list[CodeNode]:
        nodes = []
        for fn_node in self._find_all(root, "function_definition"):
            decl = fn_node.child_by_field_name("declarator")
            if decl is None:
                continue
            _fn_decl, name_node = self._function_name_from_decl(decl)
            if name_node is None:
                continue

            body_node = fn_node.child_by_field_name("body")
            symbol      = self._node_text(name_node, source)
            code        = self._node_text(fn_node, source)
            doc         = self._preceding_comment(fn_node, lines)
            calls       = self._extract_calls(body_node, source) if body_node else []
            struct_refs = self._extract_struct_refs(fn_node, source)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type="function",
                symbol_name=symbol,
                file_path=rel_path,
                line_start=fn_node.start_point[0] + 1,
                line_end=fn_node.end_point[0] + 1,
                code=code,
                docstring=doc,
                calls=calls,
                uses_structs=struct_refs,
            ))
        return nodes

    def _extract_structs(
        self, root: Node, source: bytes, lines: list[str], rel_path: str, kind: str
    ) -> list[CodeNode]:
        ts_type = "struct_specifier" if kind == "struct" else "union_specifier"
        nodes = []
        for spec_node in self._find_all(root, ts_type):
            name_node = spec_node.child_by_field_name("name")
            body_node = spec_node.child_by_field_name("body")
            # Skip forward declarations and parameter usages (no body)
            if name_node is None or body_node is None:
                continue

            symbol = self._node_text(name_node, source)
            code   = self._node_text(spec_node, source)
            doc    = self._preceding_comment(spec_node, lines)

            # Register this struct name for cross-reference in later functions
            self._known_structs.add(symbol)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type=kind,
                symbol_name=symbol,
                file_path=rel_path,
                line_start=spec_node.start_point[0] + 1,
                line_end=spec_node.end_point[0] + 1,
                code=code,
                docstring=doc,
            ))
        return nodes

    def _extract_enums(
        self, root: Node, source: bytes, lines: list[str], rel_path: str
    ) -> list[CodeNode]:
        nodes = []
        for enum_node in self._find_all(root, "enum_specifier"):
            name_node = enum_node.child_by_field_name("name")
            body_node = enum_node.child_by_field_name("body")
            if name_node is None or body_node is None:
                continue

            symbol = self._node_text(name_node, source)
            code   = self._node_text(enum_node, source)
            doc    = self._preceding_comment(enum_node, lines)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type="enum",
                symbol_name=symbol,
                file_path=rel_path,
                line_start=enum_node.start_point[0] + 1,
                line_end=enum_node.end_point[0] + 1,
                code=code,
                docstring=doc,
            ))
        return nodes

    def _extract_macros(
        self, root: Node, source: bytes, lines: list[str], rel_path: str
    ) -> list[CodeNode]:
        """Only extract non-trivial macros (multi-token values)."""
        nodes = []
        # preproc_def: simple macros (#define FOO value)
        # preproc_function_def: function-like macros (#define FOO(x) expr)
        for ts_type in ("preproc_def", "preproc_function_def"):
          for macro_node in self._find_all(root, ts_type):
            name_node  = macro_node.child_by_field_name("name")
            value_node = macro_node.child_by_field_name("value")
            if name_node is None:
                continue

            symbol     = self._node_text(name_node, source)
            value_text = self._node_text(value_node, source) if value_node else ""

            # Skip trivial macros — numeric constants with short values
            if len(value_text) < 4 and re.match(r"^\d+$", value_text.strip()):
                continue

            code = self._node_text(macro_node, source)
            doc  = self._preceding_comment(macro_node, lines)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type="macro",
                symbol_name=symbol,
                file_path=rel_path,
                line_start=macro_node.start_point[0] + 1,
                line_end=macro_node.end_point[0] + 1,
                code=code,
                docstring=doc,
            ))
        return nodes

    def _extract_includes(
        self, root: Node, source: bytes, rel_path: str
    ) -> list[str]:
        includes = []
        for inc_node in self._find_all(root, "preproc_include"):
            path_node = inc_node.child_by_field_name("path")
            if path_node:
                raw = self._node_text(path_node, source).strip('"<>')
                includes.append(raw)
        return includes
