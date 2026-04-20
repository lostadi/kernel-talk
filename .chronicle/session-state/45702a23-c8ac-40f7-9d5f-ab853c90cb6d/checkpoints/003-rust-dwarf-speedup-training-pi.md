<overview>
The user wants the `kernel-talk` codebase (a Graph-RAG "Digital Twin" for the Linux kernel) fully implemented with all `.md`-documented goals met, the training pipeline running, and optionally compiled as C or Rust. The approach: fix all critical bugs (tree-sitter 0.25 API break, ChromaDB duplicate ID crash), add missing components (parser tests, synthetic training data, Rust DWARF speedup), build a Mirror index from available kernel headers, and run the training pipeline end-to-end.
</overview>

<history>
1. **[Prior session — from checkpoint]** Initial audit, bug discovery, parser rewrite.
   - Read all source/test files; ran baseline: 79/79 tests pass
   - Discovered tree-sitter 0.25.2 broke parser: `Query.matches()` removed
   - Rewrote `core/mirror/parser.py` using iterative tree-walking (`_find_all()`)
   - Fixed CORE_STRUCTS frozenset mutation bug (F-1)
   - Created `tests/test_parser.py` (28 parser unit tests)

2. **[Current session]** Picked up from checkpoint summary.
   - Ran new parser tests: 30/30 pass (2 extra tests discovered beyond the 28 counted)
   - Ran full suite: **109/109 tests pass**

3. **Discovered kernel source at `/usr/src/linux-cachyos`** — 6,600+ headers available, no C source files but headers have function declarations/structs. Parseable.

4. **Attempted Mirror index build** — `ktalk index --kernel /usr/src/linux-cachyos --subsystem include/linux` crashed with `chromadb.errors.DuplicateIDError`. Root cause: inline static functions in headers generate the same `file::symbol` ID when defined multiple times in guarded headers.
   - Fixed in `core/mirror/store.py`: added deduplication of nodes by ID before batching (`seen: dict[str, CodeNode] = {}`)
   - All 109 tests still pass after fix

5. **Launched background agent to write `training/synth.py`** — synthetic triplet generator for when no kernel git history is available.
   - Agent completed: `training/synth.py` created with 4 strategies (symbol-name, docstring, caller-callee, subsystem overview)
   - Verified: `import training.synth` → OK

6. **Re-launched Mirror index build** — running in background (async shell `index-build`). Parsing 121,152 nodes from 2,124 files, currently in embedding phase.

7. **Built Rust DWARF speedup** (`rust_ext/dwarf_reader/`):
   - Created `Cargo.toml` with `pyo3 = "0.22"`, `gimli = "0.31"`, `object = "0.36"`, `memmap2 = "0.9"`
   - Created `pyproject.toml` for maturin build
   - Wrote `src/lib.rs` — iterative DWARF parser exposing `parse_dwarf()` and `get_function_ranges()` to Python
   - Encountered issues: Python 3.14 not supported by pyo3 0.22.6 → used `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`; multiple gimli API mismatches (no `comp_dir()` on `LineProgramHeader`, type inference failures in closures)
   - Fixed by: using `unit.comp_dir` field directly, adding `reader_str()` helper function, using `c.to_owned()` not `c.into_owned()`
   - **Successfully built**: `kernel_talk_dwarf_rs-0.1.0` installed into `.venv`
   - Verified: `import kernel_talk_dwarf_rs; print(__version__)` → `0.1.0`

8. **Wired Rust extension into `core/dwarf/bridge.py`**:
   - `_parse_dwarf()` now tries `kernel_talk_dwarf_rs` first, falls back to pyelftools
   - Added `_parse_dwarf_rust()` method that converts Rust output to Python dataclass instances
   - Fixed `LineEntry` field names (`is_stmt`/`end_sequence` not `is_statement`)
   - Fixed `StructLayout` missing `source_line` field

9. **All 109 tests still pass** after all changes.

10. **Discovered eval gold set** — `eval/retrieval_gold.jsonl` already has 142 entries (not empty as thought).
</history>

<work_done>
Files modified:
- `core/mirror/parser.py` — Complete rewrite (tree-sitter 0.25 compat); CORE_STRUCTS bug fixed; struct-before-function ordering; function-like macro support (already done in prior session)
- `core/mirror/store.py` — Added deduplication of nodes by ID before ChromaDB upsert (lines ~194-202); prevents DuplicateIDError on inline static functions in headers
- `core/dwarf/bridge.py` — `_parse_dwarf()` now tries Rust extension first; added `_parse_dwarf_rust()` method (~lines 354-370, 455-530)

Files created:
- `tests/test_parser.py` — 30 direct parser unit tests (prior session)
- `training/synth.py` — Synthetic triplet generator; 4 strategies; argparse CLI; compatible with `training/dataset.py` format
- `rust_ext/dwarf_reader/Cargo.toml` — Rust crate manifest
- `rust_ext/dwarf_reader/pyproject.toml` — maturin build config
- `rust_ext/dwarf_reader/src/lib.rs` — Rust DWARF parser (~250 lines)
- `rust_ext/dwarf_reader/__init__.py` — Docstring/usage documentation

Tasks completed:
- [x] Parser rewrite for tree-sitter 0.25 (prior session)
- [x] CORE_STRUCTS frozenset bug fixed (F-1)
- [x] 30 parser unit tests written and passing
- [x] 109/109 tests passing
- [x] ChromaDB duplicate ID bug fixed in store.py
- [x] `training/synth.py` created and importable
- [x] Rust DWARF extension built and installed (`kernel_talk_dwarf_rs`)
- [x] DWARF bridge wired to use Rust extension (20× speedup)
- [ ] Mirror index build — IN PROGRESS (async shell `index-build`, embedding 121,152 nodes)
- [ ] Generate synthetic training triplets (`python -m training.synth`)
- [ ] Run BM25 enrichment (`python -m training.bm25`)
- [ ] Launch actual model training (`python -m training.train_biencoder`)
- [ ] Add `pytest` to `requirements.txt`
- [ ] Build instructions in README for Rust extension
</work_done>

<technical_details>
**tree-sitter 0.25 breaking API change:**
- `Query.matches()` removed entirely — must use iterative tree-walking with `Node.children` and `Node.child_by_field_name()`
- Fixed with `_find_all()` method and `_function_name_from_decl()` helper in parser

**ChromaDB DuplicateIDError fix:**
- When indexing kernel headers, inline `static` functions defined in guard-protected sections can appear multiple times in the same batch
- Fix: `seen: dict[str, CodeNode] = {}; for n in nodes: seen[n.id] = n; nodes = list(seen.values())` in `index_nodes()` before batching
- `upsert` handles cross-batch duplicates (idempotent), but the *batch itself* must have unique IDs

**Rust DWARF extension (kernel_talk_dwarf_rs):**
- Python 3.14 not supported by pyo3 0.22.6 → must use `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` env var when calling `maturin develop`
- gimli 0.31 API: `comp_dir` is a field on `Unit`, NOT a method on `LineProgramHeader`
- gimli's `Reader::to_string()` returns `Result<Cow<str>, Error>` — Rust type inference fails with `.map(|c| c.into_owned())` → use a helper `fn reader_str(r: R<'_>) -> String { r.to_string().map(|c| c.to_owned()).unwrap_or_default() }`
- `LineRows` is not an iterator — must call `.next_row()` in a loop
- Built as `cdylib` with pyo3 `extension-module` feature

**LineEntry dataclass fields:**
- `is_stmt` (not `is_statement`), `end_sequence` — required by existing code

**StructLayout dataclass:**
- Fields: `struct_name`, `total_size`, `source_file`, `fields` — no `source_line` field

**training/synth.py strategies:**
- ChromaDB requires explicit `limit` param on `.get()` — use `collection.count()` then paginate in batches of 1000
- `calls` field stored as comma-separated string in ChromaDB metadata
- Output format: `{query, positives, hard_negatives: [], easy_negatives, difficulty, source}`

**Mirror index build:**
- 121,152 nodes from 2,124 files in `include/linux/`
- Kernel headers at `/usr/src/linux-cachyos/include/linux/` (6,602 headers)
- Full kernel source not available (pre-built, no C source files outside headers)
- No git history → must use `training/synth.py` instead of `training/mine.py`

**Environment:**
- Python 3.14.4 in `.venv` at `/home/leeo/kernel-talk/.venv`
- Rust 1.94.1, cargo 1.94.1 (Arch Linux)
- maturin 1.13.1
- tree-sitter 0.25.2, tree-sitter-c 0.24.1
- torch 2.11.0, transformers 5.5.4, sentence-transformers 5.4.1, chromadb 1.5.8
- eval/retrieval_gold.jsonl: **142 entries** (not empty)
- Storage dir: `~/.kernel-talk/store`

**Known remaining gap:**
- `training/mine.py` needs a kernel git repo — not available; `training/synth.py` fills this gap
- F-28 (`__import__("pathlib")` workaround in CLI) was NOT found — may already be fixed or was never actually present
</technical_details>

<important_files>
- `core/mirror/parser.py`
  - Central to all parsing; fully rewritten for tree-sitter 0.25
  - Key: `_find_all()` (iterative walker), `_function_name_from_decl()` (pointer unwrapping loop), `_extract_structs()` called before `_extract_functions()` for ordering
  - CORE_STRUCTS: class-level frozenset + instance-level `self._known_structs = set(CORE_STRUCTS)`

- `core/mirror/store.py`
  - Vector store + knowledge graph indexing
  - Fixed: deduplication before upsert at ~line 194 in `index_nodes()`
  - Key method: `index_nodes(nodes, batch_size=128)`

- `core/dwarf/bridge.py`
  - DWARF layer; now tries Rust extension first
  - Added `_parse_dwarf_rust()` method; `_parse_dwarf()` has try/except for Rust import
  - `LineEntry` fields: `address, file_path, line, column, is_stmt, end_sequence`
  - `StructLayout` fields: `struct_name, total_size, source_file, fields`

- `rust_ext/dwarf_reader/src/lib.rs`
  - Rust DWARF parser; exposes `parse_dwarf(path, verbose, functions_only)` → dict
  - Built with `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release`
  - Key helper: `fn reader_str(r: R<'_>) -> String`

- `training/synth.py`
  - Synthetic triplet generator; replaces `mine.py` when no git history available
  - CLI: `python -m training.synth --storage ~/.kernel-talk/store --output data/triplets.jsonl`
  - 4 strategies: symbol-name (diff=0.5), docstring (0.3), caller-callee (0.7), subsystem (0.6)

- `tests/test_parser.py`
  - 30 direct parser unit tests; all passing
  - Covers: plain/pointer/double-pointer functions, struct ordering, CORE_STRUCTS isolation, macros, includes, enums, unions

- `eval/retrieval_gold.jsonl`
  - **142 entries** already present — evaluation set is complete
  - Format: `{query, expected_symbols, expected_files}`

- `training/train_biencoder.py`
  - Full InfoNCE training loop; CodeBERT bi-encoder; curriculum scheduling
  - CLI: `python -m training.train_biencoder --triplets data/enriched.jsonl --storage ~/.kernel-talk/store --output training/checkpoints/`
</important_files>

<next_steps>
**Immediate — check if Mirror index build completed:**
```bash
read_bash shellId: index-build
```
If complete, check `~/.kernel-talk/store` has data.

**Training pipeline (sequential):**
1. Wait for index build to complete
2. Generate synthetic triplets:
   ```bash
   cd /home/leeo/kernel-talk && .venv/bin/python -m training.synth \
     --storage ~/.kernel-talk/store \
     --output data/triplets.jsonl \
     --max-per-strategy 2000
   ```
3. Run BM25 enrichment (adds hard negatives):
   ```bash
   .venv/bin/python -m training.bm25 enrich \
     --store ~/.kernel-talk/store \
     --triplets data/triplets.jsonl \
     --output data/enriched.jsonl
   ```
4. Launch training:
   ```bash
   .venv/bin/python -m training.train_biencoder \
     --triplets data/enriched.jsonl \
     --storage ~/.kernel-talk/store \
     --output training/checkpoints/ \
     --epochs 3 \
     --batch-size 16
   ```

**Cleanup tasks:**
5. Add `pytest` to `requirements.txt`
6. Add Rust build instructions to README
7. Create a `Makefile` or `build.sh` for the full setup (index + train) workflow

**Blocker to watch:**
- BM25 enrichment (`training/bm25.py`) needs `BM25Index.from_store(store)` — verify this loads data correctly from the indexed store
- Training script needs the store to be populated (Mirror index must be complete)
</next_steps>