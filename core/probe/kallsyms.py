"""
core/probe/kallsyms.py
───────────────────────
Layer 3 of the Digital Twin Stack: Live Symbol Table

/proc/kallsyms is the running kernel's symbol table — every exported
function and variable annotated with its current virtual address.

Format:
  ffffffff811abc00 T schedule
  ffffffff811abc40 t schedule.cold
  ffffffff82000000 D init_task
  ffffffff82001000 B jiffies_64
  0000000000000000 A irq_stack_union   ← per-CPU, 0 in kallsyms (KASLR)

Symbol types (uppercase = exported/global, lowercase = local):
  T / t  — .text  (function)
  D / d  — .data  (initialized global)
  B / b  — .bss   (uninitialized global)
  R / r  — .rodata (read-only)
  A / a  — absolute (address not adjusted by KASLR)
  W / w  — weak symbol
  U      — undefined (referenced from module)

The critical insight: kallsyms gives us the *KASLR-adjusted* addresses.
KASLR (Kernel Address Space Layout Randomization) shifts all kernel
symbols by a random offset at boot. DWARF contains *pre-KASLR* addresses
from the build. To link them, we compute the KASLR slide:

  kaslr_slide = kallsyms_addr(known_symbol) - dwarf_addr(known_symbol)

Then: live_address = dwarf_address + kaslr_slide

This is the crucial link that makes the Digital Twin work:
  DWARF addr + KASLR slide = live virtual address = /proc/kcore offset
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class KallsymsEntry:
    """A single entry from /proc/kallsyms."""
    address: int
    sym_type: str     # Single character: T, t, D, d, B, b, R, r, W, w, A, a
    name: str
    module: str = ""  # Non-empty if symbol is from a loaded module

    @property
    def is_function(self) -> bool:
        return self.sym_type.upper() in ("T", "W")

    @property
    def is_data(self) -> bool:
        return self.sym_type.upper() in ("D", "B", "R")

    @property
    def is_global(self) -> bool:
        return self.sym_type.isupper()

    @property
    def addr_hex(self) -> str:
        return f"0x{self.address:016x}"


# ─── Parser ───────────────────────────────────────────────────────────────────

# Match: "ffffffff811abc00 T schedule [module_name]"
_KALLSYMS_RE = re.compile(
    r"^([0-9a-f]+)\s+([A-Za-z])\s+(\S+)(?:\s+\[(\S+)\])?$"
)


class KallsymsBridge:
    """
    Reads and indexes /proc/kallsyms.

    This is Layer 3 of the Digital Twin stack. It provides:
      1. Symbol → live virtual address lookup (for known symbol names)
      2. Address → symbol lookup (for decoding stack traces)
      3. KASLR slide computation (for bridging DWARF to live addresses)
      4. Module symbol tracking (which symbols come from loaded modules)

    Requires root (or CONFIG_KPTR_RESTRICT=0) to see non-zero addresses.
    Without root, /proc/kallsyms shows 0x0 for most addresses.

    Usage:
        ks = KallsymsBridge()
        ks.load()
        addr = ks.symbol_address("schedule")
        slide = ks.kaslr_slide(dwarf_bridge)
    """

    KALLSYMS_PATH = "/proc/kallsyms"

    # Anchor symbols: well-known names with stable DWARF addresses.
    # We use these to compute the KASLR slide. Using multiple anchors
    # and taking the median makes the computation robust to outliers.
    ANCHOR_SYMBOLS = [
        "schedule",
        "copy_process",
        "do_fork",
        "sys_read",
        "kmalloc",
        "_text",          # Start of kernel text section (most reliable)
        "_stext",         # Synonym used in some kernel versions
        "startup_64",     # x86 64-bit entry point
    ]

    def __init__(self, kallsyms_path: str = KALLSYMS_PATH):
        self.kallsyms_path = kallsyms_path
        self._entries: list[KallsymsEntry] = []
        self._by_name: dict[str, list[KallsymsEntry]] = {}
        self._by_addr: dict[int, KallsymsEntry] = {}    # addr → first match
        self._sorted_addrs: list[int] = []              # for nearest-symbol lookup
        self._loaded = False

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self, verbose: bool = False) -> int:
        """
        Parse /proc/kallsyms into memory.
        Returns the number of symbols loaded.
        """
        if not Path(self.kallsyms_path).exists():
            raise FileNotFoundError(
                f"{self.kallsyms_path} not found. This requires Linux."
            )

        entries = []
        by_name: dict[str, list[KallsymsEntry]] = {}
        by_addr: dict[int, KallsymsEntry] = {}

        with open(self.kallsyms_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = _KALLSYMS_RE.match(line)
                if not m:
                    continue

                addr = int(m.group(1), 16)
                sym_type = m.group(2)
                name = m.group(3)
                module = m.group(4) or ""

                entry = KallsymsEntry(
                    address=addr,
                    sym_type=sym_type,
                    name=name,
                    module=module,
                )
                entries.append(entry)
                by_name.setdefault(name, []).append(entry)
                if addr not in by_addr:
                    by_addr[addr] = entry

        entries.sort(key=lambda e: e.address)
        sorted_addrs = [e.address for e in entries if e.address > 0]

        self._entries = entries
        self._by_name = by_name
        self._by_addr = by_addr
        self._sorted_addrs = sorted(set(sorted_addrs))
        self._loaded = True

        if verbose:
            print(f"[kallsyms] Loaded {len(entries)} symbols, "
                  f"{len(by_name)} unique names")

        return len(entries)

    # ── Primary Queries ───────────────────────────────────────────────────────

    def symbol_address(self, name: str) -> int | None:
        """
        Get the live virtual address of a symbol by name.
        Returns None if not found or if address is 0 (kptr_restrict).
        """
        entries = self._by_name.get(name, [])
        for e in entries:
            if e.address > 0:
                return e.address
        return None

    def symbol_entries(self, name: str) -> list[KallsymsEntry]:
        """All entries for a symbol name (there may be multiple — local variants)."""
        return self._by_name.get(name, [])

    def nearest_symbol(self, addr: int) -> KallsymsEntry | None:
        """
        Find the symbol whose address is closest to (but not exceeding) addr.
        This is the 'addr → symbol' direction — like nm or addr2line.

        Note: this is kallsyms-only resolution. For exact source line,
        combine with DwarfBridge.addr_to_line().
        """
        import bisect
        if not self._sorted_addrs:
            return None

        idx = bisect.bisect_right(self._sorted_addrs, addr) - 1
        if idx < 0:
            return None

        target_addr = self._sorted_addrs[idx]
        return self._by_addr.get(target_addr)

    def addr_to_sym_ref(self, addr: int) -> str:
        """
        Human-readable symbol reference for an address.
        e.g. "schedule+0x14" or "0xffffffff811abc14"
        """
        entry = self.nearest_symbol(addr)
        if entry and entry.address > 0:
            offset = addr - entry.address
            if offset == 0:
                return entry.name
            return f"{entry.name}+0x{offset:x}"
        return f"0x{addr:016x}"

    def is_available(self) -> bool:
        """Can we read non-zero addresses? Requires root or kptr_restrict=0."""
        if not self._loaded:
            try:
                self.load()
            except Exception:
                return False

        # Check a known symbol
        addr = self.symbol_address("_text") or self.symbol_address("startup_64")
        return addr is not None and addr > 0

    # ── KASLR Slide ───────────────────────────────────────────────────────────

    def kaslr_slide(self, dwarf_bridge) -> int | None:
        """
        Compute the KASLR address slide: live_addr - dwarf_addr.

        KASLR shifts all kernel symbols by a random offset at boot.
        DWARF contains the pre-KASLR addresses baked in at compile time.
        To link them:
            live_address = dwarf_address + kaslr_slide

        We compute this by finding a symbol that appears in both kallsyms
        (live) and DWARF (static), then taking the difference.
        Using multiple anchor symbols and taking the median is robust
        against outliers (some symbols may have different handling).

        Returns None if we can't determine the slide (kptr_restrict, no root).
        """
        if not self._loaded:
            self.load()

        slides = []
        for anchor_name in self.ANCHOR_SYMBOLS:
            # Live address from kallsyms
            live_addr = self.symbol_address(anchor_name)
            if not live_addr or live_addr == 0:
                continue

            # Static address from DWARF
            dwarf_syms = dwarf_bridge.symbol_to_addrs(anchor_name)
            if not dwarf_syms:
                continue

            # Use the first non-zero DWARF address
            for ds in dwarf_syms:
                if ds.addr_start > 0:
                    slide = live_addr - ds.addr_start
                    slides.append(slide)
                    break

        if not slides:
            return None

        # Median for robustness
        slides.sort()
        return slides[len(slides) // 2]

    def live_to_dwarf_addr(self, live_addr: int, slide: int) -> int:
        """Convert a live (KASLR) address to a static DWARF address."""
        return live_addr - slide

    def dwarf_to_live_addr(self, dwarf_addr: int, slide: int) -> int:
        """Convert a static DWARF address to a live (KASLR) address."""
        return dwarf_addr + slide

    # ── Iteration & Analysis ──────────────────────────────────────────────────

    def iter_functions(self) -> Iterator[KallsymsEntry]:
        """Iterate only function symbols (T, t, W, w)."""
        for e in self._entries:
            if e.is_function:
                yield e

    def loaded_modules(self) -> list[str]:
        """Return list of all currently loaded kernel modules (from symbol table)."""
        modules = set()
        for e in self._entries:
            if e.module:
                modules.add(e.module)
        return sorted(modules)

    def module_symbols(self, module_name: str) -> list[KallsymsEntry]:
        """All symbols exported by a specific module."""
        return [e for e in self._entries if e.module == module_name]

    def stats(self) -> dict:
        if not self._loaded:
            return {"loaded": False}

        type_counts: dict[str, int] = {}
        for e in self._entries:
            type_counts[e.sym_type] = type_counts.get(e.sym_type, 0) + 1

        return {
            "loaded":        True,
            "total_symbols": len(self._entries),
            "unique_names":  len(self._by_name),
            "symbol_types":  type_counts,
            "loaded_modules": len(self.loaded_modules()),
            "kptr_readable": self.is_available(),
        }
