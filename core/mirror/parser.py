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

# tree-sitter >= 0.22 API
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


# ─── tree-sitter Queries ───────────────────────────────────────────────────────
# We define queries once and reuse them. The pattern language is S-expression
# based — each capture name (@something) becomes a key in the match dict.

_Q_FUNCTIONS = """
(function_definition
  declarator: (function_declarator
    declarator: (identifier) @name)
  body: (compound_statement) @body) @function
"""

_Q_STRUCT_DECL = """
(struct_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @struct
"""

_Q_UNION_DECL = """
(union_specifier
  name: (type_identifier) @name
  body: (field_declaration_list) @body) @union
"""

_Q_ENUM_DECL = """
(enum_specifier
  name: (type_identifier) @name
  body: (enumerator_list) @body) @enum
"""

_Q_INCLUDES = """
(preproc_include
  path: _ @path) @include
"""

_Q_MACROS = """
(preproc_def
  name: (identifier) @name
  value: _ @value) @macro
"""

# For extracting function *calls* inside a body — used to build CALLS edges
_Q_CALLS = """
(call_expression
  function: (identifier) @callee)
"""

# For extracting struct references inside a body — used to build USES_STRUCT edges
_Q_STRUCT_REFS = """
(type_identifier) @type_name
"""


# ─── Parser ───────────────────────────────────────────────────────────────────

class KernelParser:
    """
    Parses Linux kernel C source files into CodeNode streams.

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

    # F-1: frozenset (immutable) — was a mutable set, which is a shared
    # mutable class attribute that races under parallel parsing.  Any thread
    # adding to it would mutate the class-level object seen by all instances.
    # frozenset is hashable, thread-safe, and slightly faster for `in` tests.
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

        # Pre-compile queries once — they're expensive to construct
        self._q_functions   = C_LANGUAGE.query(_Q_FUNCTIONS)
        self._q_structs     = C_LANGUAGE.query(_Q_STRUCT_DECL)
        self._q_unions      = C_LANGUAGE.query(_Q_UNION_DECL)
        self._q_enums       = C_LANGUAGE.query(_Q_ENUM_DECL)
        self._q_includes    = C_LANGUAGE.query(_Q_INCLUDES)
        self._q_macros      = C_LANGUAGE.query(_Q_MACROS)
        self._q_calls       = C_LANGUAGE.query(_Q_CALLS)
        self._q_struct_refs = C_LANGUAGE.query(_Q_STRUCT_REFS)

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

        # Extract each entity type
        nodes.extend(self._extract_functions(tree.root_node, source_bytes, lines, rel))
        nodes.extend(self._extract_structs(tree.root_node, source_bytes, lines, rel, "struct"))
        nodes.extend(self._extract_structs(tree.root_node, source_bytes, lines, rel, "union"))
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

    # ── Private Extraction Methods ─────────────────────────────────────────────

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

    def _extract_calls(self, body_node: Node, source: bytes) -> list[str]:
        """Extract all function calls from a body node."""
        matches = self._q_calls.matches(body_node)
        calls = set()
        for _, capture in matches:
            for nodes in capture.values():
                for n in (nodes if isinstance(nodes, list) else [nodes]):
                    name = self._node_text(n, source)
                    if name and not name.startswith("__"):  # skip internal builtins
                        calls.add(name)
        return sorted(calls)

    def _extract_struct_refs(self, body_node: Node, source: bytes) -> list[str]:
        """Extract struct type references from a node body."""
        matches = self._q_struct_refs.matches(body_node)
        refs = set()
        for _, capture in matches:
            for nodes in capture.values():
                for n in (nodes if isinstance(nodes, list) else [nodes]):
                    name = self._node_text(n, source)
                    if name in self.CORE_STRUCTS:
                        refs.add(name)
        return sorted(refs)

    def _extract_functions(
        self, root: Node, source: bytes, lines: list[str], rel_path: str
    ) -> list[CodeNode]:
        nodes = []
        matches = self._q_functions.matches(root)
        for _, capture in matches:
            fn_nodes  = capture.get("function", [])
            name_nodes = capture.get("name", [])
            body_nodes = capture.get("body", [])

            fn_node   = fn_nodes[0]   if isinstance(fn_nodes, list)   else fn_nodes
            name_node = name_nodes[0] if isinstance(name_nodes, list) else name_nodes
            body_node = body_nodes[0] if isinstance(body_nodes, list) else body_nodes

            if fn_node is None or name_node is None:
                continue

            symbol = self._node_text(name_node, source)
            code   = self._node_text(fn_node, source)
            doc    = self._preceding_comment(fn_node, lines)
            calls  = self._extract_calls(body_node, source) if body_node else []
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
        query = self._q_structs if kind == "struct" else self._q_unions
        nodes = []
        matches = query.matches(root)
        for _, capture in matches:
            parent_key = kind  # "struct" or "union"
            parent_nodes = capture.get(parent_key, [])
            name_nodes   = capture.get("name", [])

            parent = parent_nodes[0] if isinstance(parent_nodes, list) else parent_nodes
            name_n = name_nodes[0]   if isinstance(name_nodes, list)   else name_nodes

            if parent is None or name_n is None:
                continue

            symbol = self._node_text(name_n, source)
            code   = self._node_text(parent, source)
            doc    = self._preceding_comment(parent, lines)

            # Register new struct names for future cross-referencing
            self.CORE_STRUCTS.add(symbol)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type=kind,
                symbol_name=symbol,
                file_path=rel_path,
                line_start=parent.start_point[0] + 1,
                line_end=parent.end_point[0] + 1,
                code=code,
                docstring=doc,
            ))
        return nodes

    def _extract_enums(
        self, root: Node, source: bytes, lines: list[str], rel_path: str
    ) -> list[CodeNode]:
        nodes = []
        matches = self._q_enums.matches(root)
        for _, capture in matches:
            enum_nodes = capture.get("enum", [])
            name_nodes = capture.get("name", [])

            enum_n = enum_nodes[0] if isinstance(enum_nodes, list) else enum_nodes
            name_n = name_nodes[0] if isinstance(name_nodes, list) else name_nodes

            if enum_n is None or name_n is None:
                continue

            symbol = self._node_text(name_n, source)
            code   = self._node_text(enum_n, source)
            doc    = self._preceding_comment(enum_n, lines)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type="enum",
                symbol_name=symbol,
                file_path=rel_path,
                line_start=enum_n.start_point[0] + 1,
                line_end=enum_n.end_point[0] + 1,
                code=code,
                docstring=doc,
            ))
        return nodes

    def _extract_macros(
        self, root: Node, source: bytes, lines: list[str], rel_path: str
    ) -> list[CodeNode]:
        """Only extract non-trivial macros (multi-token values)."""
        nodes = []
        matches = self._q_macros.matches(root)
        for _, capture in matches:
            macro_nodes = capture.get("macro", [])
            name_nodes  = capture.get("name", [])
            value_nodes = capture.get("value", [])

            macro_n = macro_nodes[0] if isinstance(macro_nodes, list) else macro_nodes
            name_n  = name_nodes[0]  if isinstance(name_nodes, list)  else name_nodes
            value_n = value_nodes[0] if isinstance(value_nodes, list) else value_nodes

            if macro_n is None or name_n is None:
                continue

            symbol = self._node_text(name_n, source)
            value_text = self._node_text(value_n, source) if value_n else ""

            # Skip trivial macros — numeric constants with short names
            if len(value_text) < 4 and re.match(r"^\d+$", value_text.strip()):
                continue

            code = self._node_text(macro_n, source)
            doc  = self._preceding_comment(macro_n, lines)

            nodes.append(CodeNode(
                id=f"{rel_path}::{symbol}",
                node_type="macro",
                symbol_name=symbol,
                file_path=rel_path,
                line_start=macro_n.start_point[0] + 1,
                line_end=macro_n.end_point[0] + 1,
                code=code,
                docstring=doc,
            ))
        return nodes

    def _extract_includes(
        self, root: Node, source: bytes, rel_path: str
    ) -> list[str]:
        includes = []
        matches = self._q_includes.matches(root)
        for _, capture in matches:
            path_nodes = capture.get("path", [])
            path_n = path_nodes[0] if isinstance(path_nodes, list) else path_nodes
            if path_n:
                raw = self._node_text(path_n, source).strip('"<>')
                includes.append(raw)
        return includes
