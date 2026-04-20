<overview>
The user wants the `kernel-talk` codebase (a Graph-RAG "Digital Twin" for the Linux kernel) fully implemented, all `.md`-documented goals met, the training pipeline running, and optionally compiled as C or Rust. The approach is: audit the codebase thoroughly, fix all known bugs from the spec, add missing CLI subcommands, run training end-to-end, and build a Rust DWARF parser speedup as an optional performance enhancement.
</overview>

<history>
1. **[Prior session — from summary]** User asked to fully implement the codebase, meet all `.md` goals, get it training, and optionally make it C/Rust compilable.
   - Read all source files: `parser.py`, `graph.py`, `store.py`, `embedder.py`, `drgn_bridge.py`, `kallsyms.py`, `synthesizer.py`, `xray.py`, `ktalk.py`, all training files
   - Read all test files: `test_graph.py`, `test_store.py`, `test_persistence.py`, `test_training.py`
   - Ran full test suite: **79/79 tests passed** in 4.53s
   - Catalogued all known bugs from `KERNEL_TALK_SPEC.md`

2. **[Current session]** Picked up from the checkpoint. Began implementing fixes.
   - Discovered the tree-sitter version (0.25.2) broke the parser: `Query.matches()` no longer exists on the `Query` object — this is a **blocking bug** (the entire parser is broken in production, though tests passed because they use `MockEmbedder` and synthetic `CodeNode` objects directly, not the parser)
   - Investigated the new tree-sitter 0.25 API: `Node.child_by_field_name()`, `Node.children`, `Node.walk()` all work; `Query.matches()` is gone entirely
   - Tested edge cases for function declarator shapes (`int schedule(void)` → function_declarator → identifier; `int *kmalloc(sz)` → pointer_declarator → function_declarator → identifier; `int **foo()` → pointer_declarator → pointer_declarator → function_declarator → identifier)
   - Verified field names for all node types (struct_specifier, preproc_def, preproc_function_def, etc.)
   - Called rubber-duck agent for critique of the parser rewrite plan — caught: (a) pointer unwrapping needs to be a loop not a single step, (b) struct-ref order dependency (structs must be extracted first so function refs see them), (c) anonymous structs in typedefs are skipped (acceptable), (d) lack of direct parser tests is a significant gap
   - **Rewrote `core/mirror/parser.py`** completely: replaced all `Query.matches()` calls with iterative `_find_all()` tree-walker + `child_by_field_name()` field access; fixed CORE_STRUCTS frozenset bug (F-1); added `preproc_function_def` handling for function-like macros; fixed struct extraction ordering
   - **Created `tests/test_parser.py`** with 28 direct parser unit tests covering all entity types and edge cases
   - Ran full test suite: **79/79 still pass** (existing tests unaffected)
   - New parser tests being added bring total to 79 + new parser tests (not yet counted)
</history>

<work_done>
Files modified:
- `core/mirror/parser.py` — Complete rewrite of internals (tree-walking replaces Query.matches API); all 8 extraction methods rewritten; CORE_STRUCTS frozenset bug fixed; function-like macro support added; struct pre-pass ordering fixed

Files created:
- `tests/test_parser.py` — 28 direct unit tests for the parser covering: plain/pointer-return/double-pointer functions, call extraction, direct-vs-indirect call distinction, struct extraction, forward declaration filtering, struct-before-and-after-function ordering, CORE_STRUCTS isolation test, union/enum/macro/include extraction, ID uniqueness, line number correctness, docstring capture

Work completed:
- [x] Full codebase exploration and audit (prior session)
- [x] Test baseline established: 79/79 tests passing (prior session)
- [x] **F-1 fix**: `CORE_STRUCTS.add()` on frozenset → instance-level `self._known_structs` set
- [x] **Critical parser fix**: Rewrote tree-sitter query layer to use tree-walking (tree-sitter 0.25 removed Query.matches())
- [x] **Struct ordering fix**: Structs extracted before functions so cross-file struct refs work
- [x] **Macro fix**: Added `preproc_function_def` extraction alongside `preproc_def`
- [x] Added direct parser tests (test_parser.py)
- [ ] F-2/F-9: chunk_text() integration into indexing path (embedder already has chunk_text, but it's now wired in embed_nodes — re-verify)
- [ ] F-28: `__import__("pathlib")` workaround in CLI (not critical)
- [ ] Run training pipeline end-to-end
- [ ] Populate `eval/retrieval_gold.jsonl` with starter entries
- [ ] Investigate Rust DWARF parser speedup
- [ ] Verify new test_parser.py tests all pass

Current state:
- Parser fully fixed and working (confirmed manually)
- All 79 existing tests pass
- test_parser.py created but NOT yet run (was being written at time of compaction)
</work_done>

<technical_details>
**tree-sitter 0.25 breaking API change:**
- `Language.query()` is deprecated and returns a `Query` object that no longer has `.matches()` or `.captures()` methods — these were the primary API in 0.22
- The `Query` object in 0.25 only has metadata methods (`capture_count`, `pattern_count`, etc.)
- The new approach: use `Node.child_by_field_name(field)` for structured field access, and write recursive/iterative tree walkers with `Node.children`
- `Node` has: `.type`, `.text`, `.children`, `.child_by_field_name()`, `.start_point`, `.end_point`, `.start_byte`, `.end_byte`, `.walk()` (TreeCursor)

**Tree-sitter AST field names for C grammar:**
- `function_definition`: fields `type`, `declarator`, `body`
- `function_declarator`: fields `declarator` (the name), `parameters`
- `pointer_declarator`: field `declarator` (what's being pointed to)
- `struct_specifier`/`union_specifier`: fields `name` (type_identifier), `body` (field_declaration_list)
- `enum_specifier`: fields `name`, `body` (enumerator_list)
- `preproc_def`: fields `name` (identifier), `value` (preproc_arg)
- `preproc_function_def`: fields `name`, `parameters`, `value` — for function-like macros `#define FOO(x) ...`
- `preproc_include`: field `path`
- `call_expression`: fields `function`, `arguments`

**Function name extraction — pointer-return functions:**
- `int schedule(void)` → `declarator` = `function_declarator` → `declarator` = `identifier`
- `int *kmalloc(sz)` → `declarator` = `pointer_declarator` → `declarator` = `function_declarator` → `declarator` = `identifier`
- `int **foo(void)` → `declarator` = `pointer_declarator` → `declarator` = `pointer_declarator` → `declarator` = `function_declarator` → `declarator` = `identifier`
- Fix: loop `while node.type == 'pointer_declarator': node = node.child_by_field_name('declarator')`

**CORE_STRUCTS bug (F-1):**
- Old code: class-level `frozenset` with `self.CORE_STRUCTS.add(symbol)` → AttributeError at runtime
- Fix: keep class-level `frozenset` as immutable seed; `__init__` creates `self._known_structs = set(self.CORE_STRUCTS)`; use `self._known_structs.add()` and `in self._known_structs` everywhere

**Struct-before-function ordering:**
- Rubber duck caught: if `_extract_functions` runs before `_extract_structs`, structs defined later in the same file won't be in `_known_structs` when computing `uses_structs` for functions
- Fix: in `parse_file()`, call `_extract_structs()` first (both struct and union), then `_extract_functions()`

**Indirect calls limitation:**
- `_extract_calls` only captures direct `identifier` calls (e.g., `schedule()`)
- `ops->fn()`, `(*fp)()`, `ns.method()` are intentionally NOT captured (indirect)
- This is documented as a known limitation — only affects graph edge completeness, not correctness

**Why tests passed before the parser fix:**
- All existing tests (79) use `MockEmbedder` and synthetic `CodeNode` objects constructed directly
- No test called `KernelParser.parse_file()` — the parser was 100% broken but tests didn't catch it
- The new `test_parser.py` closes this gap

**Environment:**
- Python 3.14.4 in `.venv` at `/home/leeo/kernel-talk/.venv`
- tree-sitter 0.25.2, tree-sitter-c 0.24.1
- torch 2.11.0, transformers 5.5.4, sentence-transformers 5.4.1, chromadb 1.5.8, networkx 3.6.1
- `pytest.ini` sets `testpaths = tests/`, tmp to `/tmp/kernel-talk-pytest-cache`

**F-2/F-9 status:** `chunk_text()` is already wired into `embed_nodes()` in `embedder.py` (nodes with `embedding_text()` > 1800 chars are chunked and mean-pooled). This was already fixed in the codebase before this session.

**CLI already complete:** `ktalk eval retrieval` (F-29) and `ktalk bench index` (F-30) commands are already implemented in `cli/ktalk.py` (lines 719+ and 883+). The F-29/F-30 gap mentioned in the prior session summary was already resolved in the existing code.
</technical_details>

<important_files>
- `core/mirror/parser.py`
  - **Central to the project** — the entire parsing pipeline was broken due to tree-sitter 0.25 API change
  - **Fully rewritten** — all `_q_*` query attributes removed; `_find_all()` iterative walker added; `_function_name_from_decl()` recursive unwrapper added; all 6 extraction methods rewritten; CORE_STRUCTS bug fixed; struct ordering fixed; function-like macro support added
  - Key methods: `_find_all()` (line ~210), `_function_name_from_decl()` (~230), `_extract_functions()`, `_extract_structs()`, `_extract_calls()`, `_extract_struct_refs()`

- `tests/test_parser.py`
  - **New file** — 28 direct unit tests for the parser
  - Covers all node types, all edge cases (pointer return, forward declarations, ordering, CORE_STRUCTS isolation)
  - NOT yet run — needs to be executed to verify all tests pass

- `core/mirror/embedder.py`
  - Chunk-text pipeline is already implemented correctly (F-2/F-9 already fixed)
  - `embed_nodes()` handles long nodes via `chunk_text()` with mean-pooling
  - No changes needed

- `cli/ktalk.py`
  - All CLI commands including `eval retrieval` and `bench index` are already implemented
  - No changes needed; F-28 (`__import__("pathlib")` workaround) is low priority cosmetic issue
  - Key sections: eval (line 719), bench (line 883), twin (line 607)

- `core/probe/drgn_bridge.py`
  - F-20 (task.__state vs task.state) already fixed with try/except
  - F-22 (cpu_rq guard) already fixed with try/except around import
  - F-21 (read_struct depth cap) — still no explicit depth cap, but in practice field list is bounded by `_default_fields()`
  - No urgent changes needed

- `eval/retrieval_gold.jsonl`
  - Currently empty — needs to be populated with ~30 hand-written (query, expected_symbols, expected_files) pairs
  - Required for `ktalk eval retrieval` to function

- `training/train_biencoder.py`
  - Full InfoNCE training loop implemented; needs triplets JSONL input
  - Requires: Linux kernel git repo for `mine.py`, or synthetic data generation

- `requirements.txt`
  - Lists all deps; pytest is NOT listed (installed ad-hoc)
  - Should add `pytest` to requirements
</important_files>

<next_steps>
Remaining work (priority order):

**Immediate — run new parser tests:**
1. Run `pytest tests/test_parser.py -v` to verify all 28 new parser tests pass
2. Run full `pytest tests/ -q` to confirm all ~107 tests pass together

**Training pipeline:**
3. Check if a Linux kernel git repo is available: `ls /usr/src/linux /home/leeo/linux* 2>/dev/null`
4. If no kernel source: generate synthetic training triplets using the existing scheduler fixtures in `tests/fixtures/`
5. Build a minimal KernelStore index on any available kernel source
6. Run `python -m training.mine --kernel <path> --output data/triplets.jsonl`
7. Run BM25 hard negative enrichment: `python -m training.bm25`
8. Launch training: `python -m training.train_biencoder --triplets data/triplets.jsonl`

**Eval gold set:**
9. Populate `eval/retrieval_gold.jsonl` with 15-30 scheduler-focused (query, expected_symbols) pairs

**Rust/C compilation (DWARF speedup):**
10. Check if Rust is installed: `which cargo`
11. Write a minimal Rust crate using `gimli` crate for DWARF function/struct extraction
12. Expose via PyO3 as a Python extension; replace pyelftools bottleneck in `core/dwarf/bridge.py`
13. This addresses spec's L-4 (DWARF parse time 30-120s on first run)

**Low priority:**
14. Fix F-28: replace `__import__("pathlib").Path()` with proper `import pathlib` in `cli/ktalk.py`
15. Add `pytest` to `requirements.txt`
</next_steps>