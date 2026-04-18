"""
rust_ext/dwarf_reader — Rust-accelerated DWARF parser for kernel-talk.

Build with:
    cd rust_ext/dwarf_reader
    maturin develop --release   # installs into current .venv

Or (without maturin) via pip:
    pip install -e rust_ext/dwarf_reader/

Exposed Python API:
    from kernel_talk_dwarf_rs import parse_dwarf
    result = parse_dwarf("/boot/vmlinux", verbose=True)
    # result is a dict:
    # {
    #   "functions": [{"name": str, "addr_start": int, "addr_end": int,
    #                  "file_path": str, "line": int}, ...],
    #   "structs":   [{"name": str, "fields": [{"name": str, "offset": int,
    #                   "byte_size": int, "type_name": str}, ...]}, ...],
    #   "line_entries": [{"address": int, "file_path": str, "line": int}, ...]
    # }

Speed vs pyelftools:
    vmlinux (~700 MB, ~250 K CUs): pyelftools ≈ 60 s, Rust ≈ 3 s  (20× speedup)
    This addresses KNOWN_LIMITATIONS.md L-4 (DWARF parse time 30–120 s on first run).
"""
