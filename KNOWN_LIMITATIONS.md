# Known Limitations

These are confirmed limitations of the current system — things that are
deliberately out of scope for Phase 1 and tracked here so they don't get
lost.  Each entry has a severity rating and a pointer to the relevant phase
or finding where it will be addressed.

---

## Layer 1 — Source Parsing

**L-1. tree-sitter misses complex macro-generated code**
Severity: Medium | Phase: Future

Linux makes heavy use of macros that generate entire function definitions
(`SYSCALL_DEFINE*`, `MODULE_LICENSE`, device model registration macros).
tree-sitter parses the *textual* source, so these are captured as macros,
not as the functions they expand to.  The compiled DWARF bridge (Layer 2)
correctly sees the expanded functions, but the source graph (Layer 1) won't
have `CodeNode`s for them.  This means cross-layer linking via `link_dwarf()`
will create `BinarySymbol` nodes with no corresponding source `CodeNode`.

**L-2. The parser does not handle `#ifdef` conditional compilation**
Severity: Low | Phase: Future

Code inside `#ifdef CONFIG_*` blocks is parsed as-is without considering
the kernel's `.config` settings.  A function that is only compiled when
`CONFIG_DEBUG_PREEMPT=y` will always appear in the graph even on systems
that don't compile it.  This can cause false CALLS/USES_STRUCT edges
pointing to symbols that aren't present in the actual binary.

**L-3. CORE_STRUCTS whitelist limits USES_STRUCT edge coverage**
Severity: Low | Phase: 2

`_extract_struct_refs()` only creates `USES_STRUCT` edges for the ~35 struct
names in `KernelParser.CORE_STRUCTS`.  Many subsystem-specific structs
(e.g. `struct ext4_inode`, `struct skcipher_request`) are invisible to the
edge resolution pass.  Extending CORE_STRUCTS is safe (it's now a `frozenset`)
but expanding it too aggressively adds noise — every `unsigned int` type
reference would match `uint`.  A proper solution is to extract the full struct
definition set from the parsed tree and use that instead of a whitelist.

---

## Layer 2 — DWARF Bridge

**L-4. DWARF parsing takes 30–120 seconds on first run**
Severity: High (UX) | Fixed: Partially (cache exists, but first run is slow)

The pyelftools-based DWARF parser is single-threaded and allocates heavily.
A full kernel vmlinux parses at ~200 MB/s, meaning a 500 MB vmlinux takes
~60 seconds.  The cache (keyed on mtime+size+CRC32 per F-14) makes subsequent
runs instant.  True fix: switch to a compiled DWARF parser (libdwarf bindings
or implement in Rust/C with ctypes).  Tracked for Phase 4.

**L-5. Anonymous bitfield members are silently skipped**
Severity: Low | Phase: Future

The anonymous struct/union flattening added in F-16 handles struct/union
members.  However, bitfields (e.g., `unsigned int foo:1`) that are anonymous
are still skipped because their `DW_AT_data_member_location` is a bit offset
inside a location expression, not a simple byte constant.  Most `task_struct`
bitfields fall into this category.  Decoding bitfield offsets requires
evaluating the DWARF location expression stack machine.

**L-6. Inlined functions are not fully expanded in the inline chain**
Severity: Medium | Phase: Future

`inline_chain(addr)` currently returns at most one symbol (the outermost
containing function).  A proper implementation requires walking
`DW_TAG_inlined_subroutine` DIEs that overlap the query address, which
requires a separate address-range index over inlined DIEs.  Modern kernels
inline very aggressively, so at any given address there may be 4–6 levels
of inlining that the current code misses.

**L-7. Source file paths in DWARF are build-server absolute paths**
Severity: Low | Fixed: Partially

`_build_file_index()` strips absolute path prefixes by looking for `/linux/`
in the path.  This works for standard kernel builds but may fail for:
- Kernels built outside a `/linux/` directory
- Cross-compiled kernels where the host path doesn't match the target
- Kernels built with `O=` out-of-tree build directory

The current fallback is to strip the leading `/` and use the remainder.
This may not match the relative paths in the source graph.

---

## Layer 3 — kallsyms Bridge

**L-8. Requires root or `kptr_restrict=0` for non-zero addresses**
Severity: High (functionality) | By design

`/proc/kallsyms` returns `0x0000000000000000` for all addresses if the
process is not root and `kptr_restrict > 0` (the default on most distros).
Without real addresses, KASLR slide computation fails and BINARY_TO_LIVE
edges cannot be created.  The workaround is: `echo 0 > /proc/sys/kernel/kptr_restrict`
(reversible, safe for development) or run `ktalk twin` as root.
This is a fundamental security/functionality tradeoff, not a bug.

**L-9. Per-CPU symbols and absolute symbols report address 0**
Severity: Low | By design

Symbols with type `A` (absolute) and `a` (local absolute) retain address 0
in kallsyms regardless of `kptr_restrict` because they have no KASLR
adjustment.  Per-CPU variables (`irq_stack_union`, etc.) also appear at 0
in kallsyms.  The BINARY_TO_LIVE linking step skips symbols whose live
address is 0, so per-CPU variables will lack Layer 3 edges.

---

## Layer 4 — drgn Bridge

**L-10. drgn is not available on non-Linux platforms**
Severity: Medium (dev experience) | By design

drgn requires Linux and `/proc/kcore`.  On macOS (common for development),
`DrgnBridge.is_available()` returns False and all probe methods return mock
data or None.  The mock data is useful for testing the synthesis pipeline
but does not reflect any real kernel state.  This is expected and documented.

**L-11. drgn program initialization is slow (3–10 seconds)**
Severity: Medium (UX) | Phase: Future

`drgn.program_from_kernel()` reads `/proc/kcore` and loads debug symbols
on first call.  This happens every time `ktalk probe` is invoked.  A
persistent drgn process (daemon mode) would fix this but requires IPC.
Tracked for Phase 4.

---

## Storage and Persistence

**L-12. GraphML cannot serialize complex edge attributes**
Severity: Low | Phase: Future (F-6 from spec)

When `store.save_graph()` is called after `link_dwarf()` / `link_kallsyms()`,
the cross-layer edge attributes (`kaslr_slide`, `live_addr`, `byte_offset`)
are written to GraphML.  GraphML only supports string/int/float values, so
dict-typed attributes are silently dropped by NetworkX.  The Layer 1 source
graph round-trips correctly; the Digital Twin enrichment does not persist.
Workaround: use pickle instead of GraphML for the enriched graph.

**L-13. No incremental update support**
Severity: Medium | Phase: 2

`ktalk index` always re-indexes all nodes in the target subsystem.  It does
not detect which files have changed since the last run.  For the full kernel
tree (500K+ nodes), re-indexing takes hours.  A content-addressed incremental
update using file mtime + size would make daily re-indexing practical.

---

## Retrieval and Synthesis

**L-14. The synthesizer has no hallucination detection**
Severity: Medium | Phase: 3

The LLM may cite functions or file paths that don't exist in the kernel
or in the retrieved context.  There is currently no post-processing step
that validates citations against the `SynthesisResult.sources` list.
A simple check: verify that every `file:function` reference in the LLM
output appears in at least one retrieved `CodeNode`.

**L-15. Token budget estimation is approximate**
Severity: Low | Phase: Future

`_estimate_tokens()` in `synthesizer.py` uses `tiktoken` if available,
otherwise falls back to `len(text) // 4`.  The 4-char-per-token heuristic
is accurate for English prose but underestimates token count for dense C code
with many special characters (`->`, `*`, `__attribute__`, etc.).  For Ollama
models with smaller context windows (4096 tokens), this may cause occasional
prompt truncation.  Use `tiktoken` for production deployments.

**L-16. No streaming support for the Anthropic backend**
Severity: Low | Phase: Future

`_anthropic()` in `KernelSynthesizer` is non-streaming.  The `synthesize_stream()`
method falls back to `_call_llm()` for Anthropic, which blocks until the full
response is ready.  The Anthropic SDK supports streaming via `client.messages.stream()`.
This will be added in Phase 3 alongside the training pipeline.

---

## Evaluation

**L-17. Gold eval set is not version-annotated**
Severity: Medium | See OPEN_QUESTIONS.md Q11

The 150 queries in `eval/retrieval_gold.jsonl` assume a recent kernel
(≥ 5.15).  Symbol names like `kernel_clone` (5.7+) and `io_uring_enter`
(5.1+) may not exist in older kernels.  If you evaluate against an older
index, expected_symbols may never appear, artificially deflating Recall@k.
