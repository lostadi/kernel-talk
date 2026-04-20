"""
core/dwarf/bridge.py
─────────────────────
The DWARF Bridge — Layer 2 of the Digital Twin Stack

This is the piece that elevates Kernel-Talk from a code search engine
into a true Digital Twin. DWARF debug info inside vmlinux is the
*compiled truth* — it maps:

  C source line  ↔  machine instruction address
  C struct field ↔  byte offset in memory
  C function     ↔  [start_addr, end_addr) in .text section
  C variable     ↔  stack frame offset or register

The full 4-layer stack is:

  Layer 1  C Source        (tree-sitter, KernelParser)
  Layer 2  Compiled Binary (pyelftools DWARF, THIS FILE)
  Layer 3  Live Symbols    (kallsyms, KallsymsBridge)
  Layer 4  Live Memory     (drgn, DrgnBridge)

Layer 2 is the linchpin. Without it, layers 1 and 4 are disconnected —
you can find the code and you can read memory, but you can't prove they
correspond to each other. With DWARF, every live memory address has a
canonical path back to a specific C source line, and every C struct
field has a known offset so we can decode raw bytes without guessing.

Key capabilities provided here:
  - Function address ranges   (name → [lo_pc, hi_pc))
  - Line number tables        (PC address → source file + line)
  - Struct field offsets      (struct name → {field: (offset, size, type)})
  - Inline function chains    (address → call stack of inlined functions)
  - Source ↔ binary linking   (CodeNode.id → BinarySymbol)

Requires: pip install pyelftools
vmlinux must have DWARF info (CONFIG_DEBUG_INFO=y, not stripped).
"""

from __future__ import annotations

import bisect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class BinarySymbol:
    """
    A kernel symbol as it exists in the compiled binary.

    This is the bridge object between a CodeNode (source) and a live
    memory address (runtime). It represents a single compiled function,
    variable, or type as the linker placed it in vmlinux.

    addr_start / addr_end: virtual address range in .text (for functions)
                           For variables: addr_start = static address, addr_end = addr_start + size
    size:                  bytes
    section:               ".text", ".data", ".rodata", etc.
    source_file:           e.g. "kernel/sched/core.c"  (from DWARF DW_AT_decl_file)
    source_line:           line number in source_file
    symbol_type:           "function" | "variable" | "type" | "inline"
    inlined_at:            if this is an inlined function, the call-site address
    compilation_unit:      CU filename (highest-level source file for this CU)
    """
    name: str
    addr_start: int
    addr_end: int
    size: int
    section: str
    source_file: str
    source_line: int
    symbol_type: str                          # "function" | "variable" | "inline"
    inlined_at: int | None = None
    compilation_unit: str = ""
    inline_depth: int = 0

    @property
    def addr_start_hex(self) -> str:
        return f"0x{self.addr_start:016x}"

    @property
    def addr_end_hex(self) -> str:
        return f"0x{self.addr_end:016x}"

    def contains(self, addr: int) -> bool:
        return self.addr_start <= addr < self.addr_end

    def to_source_ref(self) -> str:
        return f"{self.source_file}:{self.source_line}"


@dataclass
class StructLayout:
    """
    The decoded memory layout of a C struct — every field's offset and size.

    This is what lets us decode raw memory bytes without guessing.
    Given a struct task_struct* at address 0xffff888100a58000, we can
    read field 'pid' at offset +0x528 (4 bytes) without needing drgn's
    type system — we derived it directly from DWARF.

    struct_name:  e.g. "task_struct"
    total_size:   sizeof(struct task_struct) from DWARF DW_AT_byte_size
    fields:       name → FieldInfo
    """
    struct_name: str
    total_size: int
    source_file: str
    fields: dict[str, "FieldInfo"] = field(default_factory=dict)

    def field_at_offset(self, offset: int) -> "FieldInfo | None":
        """Which field lives at this byte offset?"""
        for f in self.fields.values():
            if f.byte_offset == offset:
                return f
        return None

    def decode_bytes(self, raw: bytes, endian: str = "little") -> dict[str, int | str]:
        """
        Decode raw struct bytes into {field_name: value} using layout info.
        Only works for integer/pointer fields — nested structs return their offset.
        """
        result = {}
        for fname, finfo in self.fields.items():
            end = finfo.byte_offset + finfo.byte_size
            if end > len(raw):
                continue
            chunk = raw[finfo.byte_offset:end]
            if finfo.byte_size in (1, 2, 4, 8):
                val = int.from_bytes(chunk, endian)
                result[fname] = val
        return result


@dataclass
class FieldInfo:
    name: str
    byte_offset: int
    byte_size: int
    c_type: str          # human-readable type name
    dwarf_type_offset: int = 0  # offset to the DW_TAG_*_type DIE


@dataclass
class LineEntry:
    """A single entry from the DWARF line number program."""
    address: int
    file_path: str
    line: int
    column: int
    is_stmt: bool          # Is this address a statement start? (good breakpoint)
    end_sequence: bool     # Does this terminate a sequence?


# ─── DWARF Bridge ─────────────────────────────────────────────────────────────

class DwarfBridge:
    """
    Parses vmlinux DWARF debug info and provides bidirectional mapping
    between C source (file:line) and machine code (virtual addresses).

    This is the most expensive initialization in the system — parsing
    full DWARF for a kernel vmlinux takes 30–120 seconds and uses ~4GB
    of RAM. We cache everything into fast lookup structures on first load,
    then all queries are O(log N) via binary search on sorted address arrays.

    Persistence: we pickle the parsed tables to a cache file so subsequent
    starts are near-instant (just unpickle, no DWARF re-parse).

    Usage:
        bridge = DwarfBridge("/boot/vmlinux")
        bridge.load()   # parse DWARF (or load from cache)

        sym = bridge.addr_to_symbol(0xffffffff811abc00)
        layout = bridge.struct_layout("task_struct")
        line = bridge.addr_to_line(0xffffffff811abc04)
        syms = bridge.symbol_to_addrs("schedule")
    """

    def __init__(self, vmlinux_path: str | Path, cache_dir: str | Path | None = None):
        self.vmlinux_path = Path(vmlinux_path)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".kernel-talk" / "dwarf-cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Primary lookup tables (built during load())
        self._functions: list[BinarySymbol] = []          # sorted by addr_start
        self._func_addrs: list[int] = []                   # parallel sorted addr list for bisect
        self._symbol_index: dict[str, list[BinarySymbol]] = {}   # name → symbols
        self._struct_layouts: dict[str, StructLayout] = {}        # struct_name → layout
        self._line_table: list[LineEntry] = []             # sorted by address
        self._line_addrs: list[int] = []                   # parallel sorted addr list

        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, verbose: bool = True, use_cache: bool = True) -> None:
        """
        Parse vmlinux DWARF or load from cache.
        This is the expensive operation — subsequent queries are fast.
        """
        if self._loaded:
            return

        cache_file = self._cache_path()

        if use_cache and cache_file.exists():
            self._load_cache(cache_file, verbose)
        else:
            self._parse_dwarf(verbose)
            if use_cache:
                self._save_cache(cache_file, verbose)

        self._loaded = True

    # ── Primary Queries ───────────────────────────────────────────────────────

    def addr_to_symbol(self, addr: int) -> BinarySymbol | None:
        """
        Given a virtual address, find the function that contains it.

        Uses binary search on sorted addr_start list: O(log N).
        This is the foundation of 'addr2line' functionality.
        """
        if not self._func_addrs:
            return None

        # Find the rightmost function whose start ≤ addr
        idx = bisect.bisect_right(self._func_addrs, addr) - 1
        if idx < 0:
            return None

        sym = self._functions[idx]
        if sym.contains(addr):
            return sym
        return None

    def addr_to_line(self, addr: int) -> LineEntry | None:
        """
        Given a virtual address, find the C source line it came from.

        This is the inverse of compilation: machine instruction → C source.
        Uses the DWARF line number program tables.
        """
        if not self._line_addrs:
            return None

        idx = bisect.bisect_right(self._line_addrs, addr) - 1
        if idx < 0:
            return None
        return self._line_table[idx]

    def addr_to_source_ref(self, addr: int) -> str:
        """
        Human-readable source reference for a virtual address.
        Returns e.g. "kernel/sched/core.c:5234 [schedule()]"
        """
        line = self.addr_to_line(addr)
        sym  = self.addr_to_symbol(addr)

        if line and sym:
            offset = addr - sym.addr_start
            return f"{line.file_path}:{line.line} [{sym.name}+0x{offset:x}]"
        elif line:
            return f"{line.file_path}:{line.line}"
        elif sym:
            offset = addr - sym.addr_start
            return f"{sym.source_file}:{sym.source_line} [{sym.name}+0x{offset:x}]"
        return f"0x{addr:016x} (unknown)"

    def symbol_to_addrs(self, name: str) -> list[BinarySymbol]:
        """
        Find all compiled instances of a symbol by name.
        May return multiple (static functions with the same name in different CUs).
        """
        return self._symbol_index.get(name, [])

    def source_to_addrs(self, source_file: str, line: int) -> list[int]:
        """
        Given a source file + line number, return all machine addresses
        that correspond to that line.

        Inverse of addr_to_line(). Used by the X-Ray to say:
        "net/core/net-sysfs.c:operstate_show() is at address 0xffffffff..."
        """
        results = []
        for entry in self._line_table:
            if (entry.file_path.endswith(source_file) and
                entry.line == line and
                entry.is_stmt):
                results.append(entry.address)
        return results

    def struct_layout(self, struct_name: str) -> StructLayout | None:
        """
        Get the decoded memory layout of a C struct.

        This tells you exactly how to interpret raw kernel memory bytes
        as a typed struct — field names, byte offsets, and sizes.
        Essential for decoding /proc/kcore dumps without drgn's type system.
        """
        return self._struct_layouts.get(struct_name)

    def inline_chain(self, addr: int) -> list[BinarySymbol]:
        """
        Return the full inline function call chain at an address.

        Modern kernels inline aggressively. At a single address you might
        have 5 levels of inlined functions. This reconstructs that stack:
        [innermost_inline, ..., outermost_inline, actual_function]
        """
        chain = []
        sym = self.addr_to_symbol(addr)
        if sym:
            chain.append(sym)
        return chain  # Full inline chain requires more complex DWARF traversal (future work)

    # ── Iteration ─────────────────────────────────────────────────────────────

    def iter_functions(self) -> Iterator[BinarySymbol]:
        """Iterate all parsed function symbols."""
        yield from self._functions

    def iter_structs(self) -> Iterator[StructLayout]:
        """Iterate all parsed struct layouts."""
        yield from self._struct_layouts.values()

    # ── Statistics ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "functions":    len(self._functions),
            "unique_names": len(self._symbol_index),
            "struct_types": len(self._struct_layouts),
            "line_entries": len(self._line_table),
            "vmlinux":      str(self.vmlinux_path),
            "loaded":       self._loaded,
        }

    # ── DWARF Parsing (expensive, done once) ──────────────────────────────────

    def _parse_dwarf(self, verbose: bool = True) -> None:
        """
        Walk all DWARF Compilation Units and extract:
          - DW_TAG_subprogram          → BinarySymbol (function)
          - DW_TAG_structure_type      → StructLayout
          - Line number programs       → LineEntry list

        Tries the Rust extension (kernel_talk_dwarf_rs) first — ~20× faster
        than pyelftools. Falls back to pyelftools if the extension is not built.
        """
        # ── Fast path: Rust extension ─────────────────────────────────────────
        try:
            import kernel_talk_dwarf_rs as _rs
            self._parse_dwarf_rust(_rs, verbose)
            return
        except ImportError:
            if verbose:
                print("[dwarf] Rust extension not found, falling back to pyelftools.")
                print("[dwarf] Build it with: cd rust_ext/dwarf_reader && maturin develop --release")

        # ── Slow path: pyelftools ─────────────────────────────────────────────
        try:
            from elftools.elf.elffile import ELFFile
            from elftools.dwarf.descriptions import describe_form_class
        except ImportError:
            raise ImportError(
                "pyelftools is not installed.\n"
                "Run: pip install pyelftools"
            )

        if not self.vmlinux_path.exists():
            raise FileNotFoundError(
                f"vmlinux not found: {self.vmlinux_path}\n"
                "Ensure your kernel has debug symbols. See README § Debug Symbols."
            )

        if verbose:
            size_mb = self.vmlinux_path.stat().st_size / (1024 * 1024)
            print(f"[dwarf] Parsing {self.vmlinux_path} ({size_mb:.0f} MB) ...")
            print("[dwarf] This takes 30–120 seconds on first run. Subsequent runs use cache.")

        functions: list[BinarySymbol] = []
        struct_layouts: dict[str, StructLayout] = {}
        line_entries: list[LineEntry] = []
        cu_count = 0

        with open(self.vmlinux_path, "rb") as f:
            elf = ELFFile(f)

            if not elf.has_dwarf_info():
                raise ValueError(
                    f"{self.vmlinux_path} has no DWARF info.\n"
                    "Rebuild kernel with CONFIG_DEBUG_INFO=y."
                )

            dwarf = elf.get_dwarf_info()

            for CU in dwarf.iter_CUs():
                cu_count += 1
                if verbose and cu_count % 500 == 0:
                    print(f"  [dwarf] {cu_count} CUs processed, "
                          f"{len(functions)} functions, "
                          f"{len(struct_layouts)} structs ...")

                # Get the file name table for this CU (for line info)
                top_die = CU.get_top_DIE()
                cu_filename = top_die.attributes.get("DW_AT_name")
                cu_name = cu_filename.value.decode("utf-8", errors="replace") if cu_filename else ""

                # Build file index for this CU
                file_index = self._build_file_index(CU, dwarf)

                # Parse DIEs for functions and types
                for DIE in CU.iter_DIEs():
                    if DIE.tag == "DW_TAG_subprogram":
                        sym = self._parse_subprogram(DIE, file_index, cu_name)
                        if sym:
                            functions.append(sym)

                    elif DIE.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
                        layout = self._parse_struct(DIE, file_index, dwarf, CU)
                        if layout:
                            struct_layouts[layout.struct_name] = layout

                # Parse line number program for this CU
                cu_line_entries = self._parse_line_program(CU, dwarf, file_index)
                line_entries.extend(cu_line_entries)

        if verbose:
            print(f"[dwarf] Parsed {cu_count} CUs: "
                  f"{len(functions)} functions, "
                  f"{len(struct_layouts)} structs, "
                  f"{len(line_entries)} line entries")

        # Sort by address for binary search
        functions.sort(key=lambda s: s.addr_start)
        line_entries.sort(key=lambda e: e.address)

        # Build secondary indices
        symbol_index: dict[str, list[BinarySymbol]] = {}
        for sym in functions:
            symbol_index.setdefault(sym.name, []).append(sym)

        self._functions = functions
        self._func_addrs = [s.addr_start for s in functions]
        self._symbol_index = symbol_index
        self._struct_layouts = struct_layouts
        self._line_table = line_entries
        self._line_addrs = [e.address for e in line_entries]

    def _parse_dwarf_rust(self, rs_module, verbose: bool) -> None:
        """
        Parse DWARF using the Rust extension (kernel_talk_dwarf_rs).
        Fills the same internal data structures as _parse_dwarf().
        """
        if not self.vmlinux_path.exists():
            raise FileNotFoundError(
                f"vmlinux not found: {self.vmlinux_path}\n"
                "Ensure your kernel has debug symbols."
            )

        if verbose:
            size_mb = self.vmlinux_path.stat().st_size / (1024 * 1024)
            print(f"[dwarf] Parsing {self.vmlinux_path} ({size_mb:.0f} MB) via Rust extension ...")

        result = rs_module.parse_dwarf(str(self.vmlinux_path), verbose=verbose)

        functions: list[BinarySymbol] = []
        for f in result["functions"]:
            functions.append(BinarySymbol(
                name=f["name"],
                addr_start=f["addr_start"],
                addr_end=f["addr_end"],
                size=f["addr_end"] - f["addr_start"],
                section=".text",
                source_file=f["file_path"],
                source_line=int(f["line"]),
                symbol_type="function",
            ))

        struct_layouts: dict[str, StructLayout] = {}
        for s in result["structs"]:
            layout = StructLayout(
                struct_name=s["name"],
                fields={},
                total_size=0,
                source_file="",
            )
            struct_layouts[s["name"]] = layout

        line_entries: list[LineEntry] = []
        for le in result["line_entries"]:
            line_entries.append(LineEntry(
                address=le["address"],
                file_path=le["file_path"],
                line=int(le["line"]),
                column=0,
                is_stmt=True,
                end_sequence=False,
            ))

        if verbose:
            print(f"[dwarf] Rust: {len(functions)} functions, "
                  f"{len(struct_layouts)} structs, {len(line_entries)} line entries")

        functions.sort(key=lambda s: s.addr_start)
        line_entries.sort(key=lambda e: e.address)

        symbol_index: dict[str, list[BinarySymbol]] = {}
        for sym in functions:
            symbol_index.setdefault(sym.name, []).append(sym)

        self._functions = functions
        self._func_addrs = [s.addr_start for s in functions]
        self._symbol_index = symbol_index
        self._struct_layouts = struct_layouts
        self._line_table = line_entries
        self._line_addrs = [e.address for e in line_entries]


        self, DIE, file_index: dict[int, str], cu_name: str
    ) -> BinarySymbol | None:
        """Extract a BinarySymbol from a DW_TAG_subprogram DIE."""
        attrs = DIE.attributes

        # Must have a name and an address range
        name_attr = attrs.get("DW_AT_name")
        lo_attr   = attrs.get("DW_AT_low_pc")
        hi_attr   = attrs.get("DW_AT_high_pc")

        if not (name_attr and lo_attr):
            return None

        name = name_attr.value
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        lo_pc = lo_attr.value
        if not lo_pc:
            return None

        # DW_AT_high_pc can be absolute or relative to low_pc
        # Form class "address" = absolute; "constant" = relative offset
        hi_pc = lo_pc
        if hi_attr:
            from elftools.dwarf.descriptions import describe_form_class
            form_class = describe_form_class(hi_attr.form)
            if form_class == "address":
                hi_pc = hi_attr.value
            elif form_class == "constant":
                hi_pc = lo_pc + hi_attr.value

        if hi_pc <= lo_pc:
            return None

        # Source location
        file_attr = attrs.get("DW_AT_decl_file")
        line_attr = attrs.get("DW_AT_decl_line")
        src_file = file_index.get(file_attr.value, cu_name) if file_attr else cu_name
        src_line = line_attr.value if line_attr else 0

        # Is this inlined?
        inline_attr = attrs.get("DW_AT_inline")
        inlined_at = None
        is_inline = inline_attr and inline_attr.value in (1, 3)  # DW_INL_inlined

        return BinarySymbol(
            name=name,
            addr_start=lo_pc,
            addr_end=hi_pc,
            size=hi_pc - lo_pc,
            section=".text",
            source_file=src_file,
            source_line=src_line,
            symbol_type="inline" if is_inline else "function",
            inlined_at=inlined_at,
            compilation_unit=cu_name,
        )

    def _parse_struct(self, DIE, file_index: dict[int, str], dwarf, CU) -> StructLayout | None:
        """
        Extract a StructLayout from a DW_TAG_structure_type DIE.

        The critical piece: DW_TAG_member children have DW_AT_data_member_location
        which gives us the byte offset of each field within the struct.
        """
        attrs = DIE.attributes

        name_attr = attrs.get("DW_AT_name")
        size_attr = attrs.get("DW_AT_byte_size")

        if not name_attr or not size_attr:
            return None

        name = name_attr.value
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")

        total_size = size_attr.value

        file_attr = attrs.get("DW_AT_decl_file")
        src_file  = file_index.get(file_attr.value, "") if file_attr else ""

        layout = StructLayout(
            struct_name=name,
            total_size=total_size,
            source_file=src_file,
        )

        # Walk member DIEs.
        # F-16: anonymous unions/structs (e.g. in task_struct) have no
        # DW_AT_name.  We detect them and recursively flatten their children
        # into the parent layout, adjusting byte offsets.
        for child in DIE.iter_children():
            if child.tag != "DW_TAG_member":
                continue

            member_name_attr = child.attributes.get("DW_AT_name")
            offset_attr      = child.attributes.get("DW_AT_data_member_location")
            type_attr        = child.attributes.get("DW_AT_type")

            if not offset_attr:
                continue

            try:
                byte_offset = int(offset_attr.value)
            except (TypeError, ValueError):
                continue

            if not member_name_attr:
                # Anonymous struct/union — flatten its children
                if type_attr:
                    try:
                        inner_die = CU.get_DIE_from_refaddr(
                            CU.cu_offset + type_attr.value
                        )
                        if inner_die.tag in ("DW_TAG_structure_type",
                                              "DW_TAG_union_type"):
                            self._flatten_anon_members(
                                inner_die, byte_offset, layout, CU
                            )
                    except Exception:
                        pass
                continue

            mname = member_name_attr.value
            if isinstance(mname, bytes):
                mname = mname.decode("utf-8", errors="replace")

            # Try to determine field size from the referenced type DIE
            byte_size = 0
            c_type_name = "unknown"
            type_ref_offset = type_attr.value if type_attr else 0

            if type_attr:
                try:
                    type_die = CU.get_DIE_from_refaddr(
                        CU.cu_offset + type_attr.value
                    )
                    size_a = type_die.attributes.get("DW_AT_byte_size")
                    if size_a:
                        byte_size = size_a.value
                    type_name_a = type_die.attributes.get("DW_AT_name")
                    if type_name_a:
                        c_type_name = type_name_a.value
                        if isinstance(c_type_name, bytes):
                            c_type_name = c_type_name.decode("utf-8", errors="replace")
                except Exception:
                    pass

            layout.fields[mname] = FieldInfo(
                name=mname,
                byte_offset=byte_offset,
                byte_size=byte_size,
                c_type=c_type_name,
                dwarf_type_offset=type_ref_offset if type_attr else 0,
            )

        return layout if layout.fields else None

    def _flatten_anon_members(
        self,
        type_die,
        base_offset: int,
        layout: "StructLayout",
        CU,
    ) -> None:
        """
        F-16: Recursively flatten anonymous struct/union DIE children into
        the parent StructLayout, adjusting all byte offsets by base_offset.

        Example: task_struct has anonymous unions whose fields (like 'thread_info',
        'stack_canary') would otherwise be invisible to struct_field_offset().
        After flattening, every field — regardless of anonymous nesting depth —
        appears directly in layout.fields with its correct absolute offset.

        Handles arbitrary nesting depth.  Cycles are impossible in DWARF type
        trees (they're DAGs), so no visited-set is needed.
        """
        for child in type_die.iter_children():
            if child.tag != "DW_TAG_member":
                continue

            member_name_attr = child.attributes.get("DW_AT_name")
            offset_attr      = child.attributes.get("DW_AT_data_member_location")
            type_attr        = child.attributes.get("DW_AT_type")

            if not offset_attr:
                continue

            try:
                child_offset = int(offset_attr.value)
            except (TypeError, ValueError):
                continue

            abs_offset = base_offset + child_offset

            if not member_name_attr:
                # Another anonymous level — recurse
                if type_attr:
                    try:
                        inner_die = CU.get_DIE_from_refaddr(
                            CU.cu_offset + type_attr.value
                        )
                        if inner_die.tag in ("DW_TAG_structure_type",
                                              "DW_TAG_union_type"):
                            self._flatten_anon_members(inner_die, abs_offset, layout, CU)
                    except Exception:
                        pass
                continue

            mname = member_name_attr.value
            if isinstance(mname, bytes):
                mname = mname.decode("utf-8", errors="replace")

            byte_size = 0
            c_type_name = "unknown"
            type_ref_offset = type_attr.value if type_attr else 0

            if type_attr:
                try:
                    type_die_inner = CU.get_DIE_from_refaddr(
                        CU.cu_offset + type_attr.value
                    )
                    size_a = type_die_inner.attributes.get("DW_AT_byte_size")
                    if size_a:
                        byte_size = size_a.value
                    type_name_a = type_die_inner.attributes.get("DW_AT_name")
                    if type_name_a:
                        c_type_name = type_name_a.value
                        if isinstance(c_type_name, bytes):
                            c_type_name = c_type_name.decode("utf-8", errors="replace")
                except Exception:
                    pass

            # Don't overwrite if a named member with this name was already
            # added from the outer struct (outer definition wins)
            if mname not in layout.fields:
                layout.fields[mname] = FieldInfo(
                    name=mname,
                    byte_offset=abs_offset,
                    byte_size=byte_size,
                    c_type=c_type_name,
                    dwarf_type_offset=type_ref_offset,
                )

    def _parse_line_program(self, CU, dwarf, file_index: dict[int, str]) -> list[LineEntry]:
        """
        Parse the DWARF line number program for a Compilation Unit.

        The line program is a state machine that maps instruction addresses
        back to (file, line, column) tuples. This is the 'addr2line' data.
        """
        entries = []
        try:
            lineprog = dwarf.line_program_for_CU(CU)
            if lineprog is None:
                return []

            for entry in lineprog.get_entries():
                state = entry.state
                if state is None or state.end_sequence:
                    continue
                if state.address == 0:
                    continue

                # Map file number to actual path using this CU's file table
                file_path = file_index.get(state.file, "")
                if not file_path:
                    continue

                entries.append(LineEntry(
                    address=state.address,
                    file_path=file_path,
                    line=state.line,
                    column=state.column,
                    is_stmt=state.is_stmt,
                    end_sequence=False,
                ))
        except Exception:
            pass  # Some CUs have malformed line programs — skip gracefully

        return entries

    def _build_file_index(self, CU, dwarf) -> dict[int, str]:
        """
        Build a {file_number: path} mapping for a Compilation Unit.
        DWARF line programs reference files by index into this table.
        """
        index: dict[int, str] = {}
        try:
            lineprog = dwarf.line_program_for_CU(CU)
            if lineprog is None:
                return index

            header = lineprog.header
            include_dirs = [b""]  # index 0 is the compilation directory
            for d in header.include_directory:
                if isinstance(d, bytes):
                    include_dirs.append(d)
                else:
                    include_dirs.append(str(d).encode())

            for i, file_entry in enumerate(header.file_entry, start=1):
                fname = file_entry.name
                if isinstance(fname, bytes):
                    fname = fname.decode("utf-8", errors="replace")

                dir_idx = file_entry.dir_index
                if dir_idx < len(include_dirs):
                    d = include_dirs[dir_idx]
                    if isinstance(d, bytes):
                        d = d.decode("utf-8", errors="replace")
                    # Strip absolute prefix to get kernel-relative path
                    full = f"{d}/{fname}" if d else fname
                    # Normalize to kernel-relative: strip everything before "linux/"
                    if "/linux/" in full:
                        full = full[full.index("/linux/") + 7:]
                    elif full.startswith("/"):
                        full = full.lstrip("/")
                    index[i] = full
                else:
                    index[i] = fname
        except Exception:
            pass

        return index

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        """
        F-14: Robust cache key = mtime + size + CRC32 of first 64 KB.

        mtime alone is fragile:
          - Copy from another machine preserves mtime but is a different build
          - Filesystem truncates mtime to 1-second precision on some configs
          - Replacing vmlinux atomically (same mtime bucket) defeats the check

        Adding file size catches same-mtime / different-content collisions.
        CRC32 of the first 64 KB (ELF header + initial sections) catches
        same-mtime + same-size collisions essentially perfectly at negligible
        I/O cost (~0.1 ms for a 64 KB read).
        """
        if not self.vmlinux_path.exists():
            return self.cache_dir / "dwarf_missing.pkl"

        import zlib
        st = self.vmlinux_path.stat()
        mtime = int(st.st_mtime)
        size  = st.st_size

        with open(self.vmlinux_path, "rb") as f:
            header = f.read(65536)   # 64 KB — covers ELF header + section table
        crc = zlib.crc32(header) & 0xFFFFFFFF

        return self.cache_dir / f"dwarf_{mtime}_{size}_{crc:08x}.pkl"

    def _save_cache(self, cache_file: Path, verbose: bool = True) -> None:
        import pickle
        if verbose:
            print(f"[dwarf] Saving cache to {cache_file} ...")
        with open(cache_file, "wb") as f:
            pickle.dump({
                "functions":      self._functions,
                "symbol_index":   self._symbol_index,
                "struct_layouts": self._struct_layouts,
                "line_table":     self._line_table,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"[dwarf] Cache saved ({size_mb:.0f} MB)")

    def _load_cache(self, cache_file: Path, verbose: bool = True) -> None:
        import pickle
        if verbose:
            print(f"[dwarf] Loading DWARF cache from {cache_file} ...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        self._functions      = data["functions"]
        self._symbol_index   = data["symbol_index"]
        self._struct_layouts = data["struct_layouts"]
        self._line_table     = data["line_table"]
        self._func_addrs     = [s.addr_start for s in self._functions]
        self._line_addrs     = [e.address    for e in self._line_table]
        if verbose:
            print(f"[dwarf] Loaded: {len(self._functions)} functions, "
                  f"{len(self._struct_layouts)} structs, "
                  f"{len(self._line_table)} line entries")
