# Kernel-Talk: Architecture Audit & Build-Out Specification

**Author:** specification drafted for Lee Ostadi
**Audience:** Lee + future coding agents (human or LLM) who will implement the phases below
**Scope of this document:** five deliverables rolled into one file —
  (A) an architectural walkthrough of the existing four-subsystem codebase, with
      bug and performance findings cited to line numbers;
  (B) the vocabulary spec for the Modification module (clear-language, not
      GitHub-English);
  (C) the design for the training pipeline and the first neural network to train,
      with first-principles rationale;
  (D) the design for a hardware-detection `setup.sh` that produces a reproducible
      CUDA / ROCm / oneAPI / Metal / CPU environment;
  (E) the end-to-end architecture for the patch-based kernel-modification
      module.

This file is meant to be read top-to-bottom once, then used as a reference. Every
section has an `Intent → Design → Steps → Evidence/Verification` structure so a
junior agent can pull out one section, execute, and hand back a reviewable diff.

---

## 0. Meta: how to read and use this document

Three reading modes.

**Architect mode.** You want to understand the theory of the system — why it is
four layers, why the graph is multi-di, why KASLR matters, why patches beat
live rewrites. Read §1, §2, §7 and skim the rest.

**Fix-it mode.** The code has bugs. You want a checklist. Jump to §3
(Findings Index). Every finding is labelled `F-n`, has a severity, a file,
a line number, and a proposed remediation.

**Build-it mode.** You're starting the next phase. Go straight to §5, §6, §7,
§8 for the four build phases. Each phase specifies its entry state, required
files, interfaces, success criteria, and tests.

**First principles the spec rests on.** These are load-bearing and appear
throughout the document:
1. *A digital twin of the kernel must model four layers that each have their
   own representation: source, binary, symbol table, live memory.* The layers
   are bridged, not unified; each has its own addressing regime.
2. *A debugging tool should stay read-only at the kernel-memory boundary,
   even when the project also offers a patch-and-rebuild workflow.* The
   read-only invariant is what makes drgn safe on a running box. Modification
   lives a layer up, in the source tree, and is applied by a rebuild.
3. *User-facing vocabulary is UX.* "Commit," "push," "pull" are artifacts of
   a particular tool (Git). They leak implementation. The Modification module
   uses words that describe the user's action, not Git's internals.
4. *Training a neural model for this project is only worth doing when the
   model clearly beats a well-tuned non-learned baseline on a measurable
   benchmark.* The first model should be chosen for biggest bang-per-epoch
   on the system's core bottleneck (retrieval).

---

## 1. The theory of the existing system

Kernel-Talk today is a Graph-RAG system wrapped around four representations of
the Linux kernel. Reading the code, the design intent becomes clear: the
kernel is *the same thing* across four representations — source text, binary
ELF, symbol table, running memory — and the tool's job is to expose those
representations as a single queryable object.

The four layers and what each contributes:

**Layer 1 — Mirror.** Parses C source with tree-sitter into a `KernelGraph`
(NetworkX `MultiDiGraph`) of `CodeNode`s. Node types: `function`, `struct`,
`union`, `enum`, `macro`, `file`. Edges:
`CALLS`, `USES_STRUCT`, `DEFINED_IN`, `INCLUDES`. Every code node is
embedded with CodeBERT into a ChromaDB collection for cosine-similarity
retrieval. Retrieval is hybrid: the store does vector search, then expands
the neighborhood in the graph to return structurally-adjacent nodes.

**Layer 2 — DWARF Bridge.** Parses `vmlinux` with pyelftools to extract
`BinarySymbol`s (name, static addr range), `StructLayout`s (field name → byte
offset → type → size), and `LineEntry`s (instruction address → source file
& line). This is how the tool knows that `task_struct.pid` lives at offset
`0x4e8` on *this* kernel build. A pickle cache keyed on `vmlinux` mtime
avoids re-parsing.

**Layer 3 — Kallsyms.** Reads `/proc/kallsyms`, indexes every symbol by name
and by address, and — critically — computes the KASLR slide by comparing a
live anchor symbol against its DWARF address. The slide is the bridge
between static binary addresses and runtime virtual addresses. Without it,
Layer 2 and Layer 4 cannot talk to each other.

**Layer 4 — Probe (drgn).** Wraps `drgn.Program.from_kernel()` or
`from_core_dump(/proc/kcore)` to let the tool walk live kernel data
structures: iterate processes, read `task_struct` fields, peek runqueues.
This is the only layer that touches actual memory on the running kernel,
and it is strictly read-only.

Above these layers sit three synthesis tools:
- `tools/xray.py` — a Q&A pipeline that takes a question, matches known
  patterns (e.g. "what PIDs are running?"), falls back to vector search in
  the Mirror, and — if it looks like a "what's happening now" question —
  fires a live Probe. It also owns the full reverse-traversal pipeline:
  `addr2line(live_addr)` → subtract KASLR slide → DWARF line program →
  Mirror CodeNode.
- `core/synthesis/synthesizer.py` — a prompt assembler that glues
  THEORY (Mirror context) and REALITY (Probe context) into a single prompt
  and dispatches to Ollama, OpenAI, or Anthropic.
- `cli/ktalk.py` — the Click-based CLI that stitches everything together.

### 1.1 Why the four-layer split is the right decomposition

From first principles: the kernel is a single artifact but you interact with
it through different projections.

- **If you only had source**, you couldn't answer "what address is
  `schedule`?" because addresses don't exist in C — they are assigned by
  the linker.
- **If you only had binary**, you couldn't answer "does this call
  `copy_process`?" because the binary has machine instructions and addresses,
  not call graphs in their high-level form.
- **If you only had the symbol table**, you couldn't answer "what's the byte
  offset of `mm->pgd`?" because kallsyms has function addresses but no
  struct layout info.
- **If you only had live memory**, you couldn't answer "who defined this
  struct?" because the running kernel has forgotten all of that.

You need all four, and you need the bridges between them (DWARF↔kallsyms↔
live via KASLR slide; source↔DWARF via file+line attributes; graph↔ChromaDB
via node ID). The code already honors this. The spec preserves it.

### 1.2 Where the current implementation leaks

The existing code has honest, fixable problems. They fall into three
categories:

1. **Correctness bugs.** Edge resolution loses data, one file has a stray
   docstring fragment that may (depending on how Python lexes it) break
   import, the ChromaDB reconstruction in `store.py` is missing a field,
   CLI argument parsing has a fragile import pattern, and there are
   inconsistencies between the drgn Linux-version target and the
   `task.__state` attribute access.
2. **Performance.** Two-pass edge resolution does full graph scans per
   edge on indexing; the embedder chunking fallback is defined but never
   used for long nodes; the KernelGraph uses NetworkX for everything
   including lookups that should be O(1) via auxiliary index.
3. **Ergonomics and robustness.** No persistence of the vector store's
   matching graph snapshot (you can get out-of-sync), no integrity check
   on the DWARF pickle cache beyond mtime, CLI help text assumes a single
   kernel workflow, and there is no evaluation / benchmarking harness.

§3 below enumerates these with line citations. §5 through §8 are the
build plan.

---

## 2. Architectural walkthrough, module by module

Each module's subsection follows the same template:
- **Intent** — what the module is for, in one paragraph.
- **Interface** — the public API other modules rely on.
- **Mechanism** — how it works internally (reading the code).
- **Observations** — tied to line numbers and filed under §3 for tracking.

### 2.1 `core/mirror/parser.py` — tree-sitter C parser

**Intent.** Take a Linux source tree, walk every `.c` / `.h` file, and emit
`CodeNode` objects (one per function, struct, union, enum, macro, and file).
This is the *source* layer of the digital twin.

**Interface.**
```python
parser = KernelParser(kernel_root: Path, extensions=(".c", ".h"))
for node in parser.parse():
    yield node            # CodeNode
for rel in parser.relationships(node):
    yield (node_id, type, target)   # CALLS / USES_STRUCT / INCLUDES
```
`CodeNode` fields: `id`, `name`, `node_type`, `path`, `start_line`, `end_line`,
`code`, `docstring`, `signature`.

**Mechanism.** Uses tree-sitter 0.22's `C_LANGUAGE.query()` API. Captures
function definitions, struct/union/enum declarations, `#define` macros.
Extracts the preceding block-comment as `docstring` when present. Keeps a
`CORE_STRUCTS` set that grows as structs are encountered, so later passes
can decide what's "important."

**Observations.**
- `CORE_STRUCTS` is a module-level mutable set that grows during parse
  (finding F-1). Two parallel parses in the same process will race.
- `embedding_text()` concatenates docstring + type/path prefix + raw code,
  which for some functions will exceed the 512-token CodeBERT limit
  (F-2). The embedder has a `chunk_text()` fallback but `parser.py`'s
  `embedding_text()` hands the whole thing to the embedder unchanked.
- File-level nodes are emitted but `INCLUDES` edges are resolved by
  filename-basename match, not full path, so two headers with the same
  basename in different subdirectories collide (F-3).
- There is no `WEAK_LINK` or `MAYBE_CALLS` edge type for indirect calls
  (function pointers). For the kernel this is a real omission — a huge
  fraction of kernel dispatch is through `struct X_operations` vtables.
  Not a bug, a design limitation (F-4).

### 2.2 `core/mirror/graph.py` — KernelGraph

**Intent.** Store all `CodeNode`s and all edges between them in a single
graph object that supports: add, lookup, neighborhood expansion, binding
to Layer 2/3/4 data, serialization to GraphML.

**Interface.**
```python
g = KernelGraph()
g.add_node(code_node)
g.add_edge(src_id, dst_id, edge_type)
g.resolve_edges()                    # two-pass symbol resolution
g.neighborhood(node_id, depth=2)     # BFS on successors + predecessors
g.link_dwarf(dwarf_bridge)           # add SOURCE_TO_BINARY edges
g.link_kallsyms(kallsyms, slide)     # add BINARY_TO_LIVE edges
g.link_struct_layouts(dwarf_bridge)  # add FIELD_TO_OFFSET edges
g.live_address_for(node_id)
g.save_graphml(path) / g.load_graphml(path)
```

**Mechanism.** `MultiDiGraph` subclass; edge types are constants on an
`EdgeType` class. `resolve_edges()` walks pending `(src, name)` pairs and
tries to resolve `name` to a node id; currently this does an O(N) pass
over node data for each pending edge — quadratic in graph size (F-5).
Serialization to GraphML uses `nx.write_graphml`, which requires every
attribute to be a primitive; complex objects (dataclass-valued fields
like `struct_layout`) get coerced to `str(repr(...))` and lose data on
reload (F-6).

**Observations.**
- `neighborhood(node, depth)` expands both predecessors and successors,
  which is almost always what you want for a digital-twin question ("what
  calls this? what does this call?"), but it has no way to weight edges —
  e.g. `INCLUDES` edges get the same treatment as `CALLS` edges, bloating
  context for retrieval (F-7).
- There is no `by_name` secondary index on the graph — every `find_by_name`
  operation walks all nodes (F-8). Given that node IDs are `<type>:<path>#<name>`,
  a `dict[str, list[str]]` built alongside `add_node` would make this O(1).

### 2.3 `core/mirror/embedder.py` — CodeBERT embeddings

**Intent.** Turn `CodeNode.embedding_text()` into a float vector so the
vector store can find semantically-similar code to a query.

**Interface.**
```python
e = CodeEmbedder(model_name="microsoft/codebert-base", batch_size=32)
vec = e.embed(text: str) -> np.ndarray        # L2-normalized
vecs = e.embed_many(texts: list[str])          # batched
chunks = e.chunk_text(text, max_tokens=512)    # fallback for long text
```

**Mechanism.** `sentence_transformers.SentenceTransformer` under the hood.
`_auto_device()` picks CUDA > MPS > CPU. Vectors are L2-normalized so
cosine similarity reduces to dot product in ChromaDB. `chunk_text` is
implemented but `parser → embedder` pipeline never calls it — long
functions are silently truncated by the tokenizer (F-9).

**Observations.**
- The CodeBERT model is a good default for code retrieval, but we have
  a second choice worth preparing for: a *fine-tuned* model trained on
  kernel-specific retrieval pairs. See §6 for the training pipeline.
- MPS path is untested for `encode` with `show_progress_bar=True` on
  large batches; known PyTorch issue on some macOS versions (F-10).

### 2.4 `core/mirror/store.py` — KernelStore (ChromaDB + graph)

**Intent.** Unify the vector store and the graph behind one retrieval
interface. `hybrid_search(query, k=10, graph_depth=1)` returns a set of
primary hits by vector similarity, then for each primary hit walks the
graph to find structurally-adjacent nodes.

**Interface.**
```python
store = KernelStore(storage_dir, embedder, graph)
store.add_node(code_node)
store.vector_search(query, k=10)            # list[HybridResult]
store.hybrid_search(query, k=10, depth=1)   # primary + context
store.persist()
```
`HybridResult(primary: CodeNode, context: list[CodeNode], graph_ctx: dict)`.

**Mechanism.** ChromaDB collection with `hnsw:space: cosine`, node
metadata stored on the Chroma document. On search, reconstructs `CodeNode`
objects from Chroma metadata — and here is a bug: the reconstruction
loop omits the `docstring` field (F-11). If you `vector_search` and then
try to render the result as context, the docstring is empty even though
it was indexed.

**Observations.**
- `hybrid_search`'s graph expansion has no deduplication on identical
  context nodes between two primary hits, so the prompt can receive the
  same function twice (F-12).
- The store doesn't know whether the graph it persists matches the
  Chroma collection (F-13). If you re-index only the Chroma collection
  but forget the graph, the two go out of sync silently. Need a
  versioned pairing.

### 2.5 `core/dwarf/bridge.py` — DWARF parsing

**Intent.** Parse `vmlinux` DWARF and expose: function address ranges,
struct layouts, line-program entries. This is Layer 2 — the link between
source and binary.

**Interface.**
```python
d = DwarfBridge(vmlinux_path, cache_dir=...)
d.load()
d.symbol_to_addrs(name) -> list[BinarySymbol]
d.addr_to_symbol(addr) -> BinarySymbol | None
d.struct_layout(name) -> StructLayout | None
d.addr_to_line(addr) -> LineEntry | None
d.line_to_addr(path, line) -> int | None
```

**Mechanism.** `ELFFile(vmlinux).get_dwarf_info()`. For each CU it walks
subprograms (functions), struct/union DIEs (layouts), and the line
program. Builds sorted lists of `(addr_start, addr_end, symbol)` for
binary search. Cache is a single pickle file keyed on `vmlinux` mtime.

**Observations.**
- Cache integrity is mtime-only (F-14). If two different kernel builds
  have identical mtimes (copied from a tarball), you get stale data.
  A hash of `(size, mtime, CRC-32 of first and last 4KB)` would be
  cheap and correct.
- `_parse_line_program` parses every CU even if callers only want a
  specific file; lazy parsing keyed on requested file would be a large
  speedup on first-touch queries (F-15).
- `StructLayout.fields` is a flat list — no support for anonymous
  unions/structs inside a parent, which kernel code uses heavily
  (e.g. `struct task_struct` has several anonymous unions). You will
  miss field offsets inside those (F-16).

### 2.6 `core/probe/kallsyms.py` — live symbol table

**Intent.** Read `/proc/kallsyms`, index by name and address, compute the
KASLR slide relative to DWARF. Layer 3.

**Interface.**
```python
ks = KallsymsBridge()
ks.load()
ks.symbol_address(name) -> int | None
ks.nearest_symbol(addr) -> KallsymsEntry | None
ks.addr_to_sym_ref(addr) -> str                 # "schedule+0x14"
ks.kaslr_slide(dwarf_bridge) -> int | None      # median of anchor slides
```

**Mechanism.** Regex-based parse of kallsyms. Anchors: `schedule`,
`copy_process`, `do_fork`, `sys_read`, `kmalloc`, `_text`, `_stext`,
`startup_64`. Takes the median slide across anchors.

**Observations.**
- `do_fork` is deprecated (removed from kernel in v5.7+), so the anchor
  list includes symbols that may not exist on modern kernels (F-17).
  Harmless because median ignores missing anchors, but dead weight.
- `sys_read` on x86_64 is `__x64_sys_read` — the plain `sys_read` name
  may also be missing on modern kernels with syscall wrappers (F-18).
- `is_available()` re-triggers a `load()` if we haven't loaded yet,
  which is side-effectful from a "just checking" method (F-19).

### 2.7 `core/probe/drgn_bridge.py` — live kernel probe

**Intent.** Wrap `drgn.Program()` to expose typed reads of kernel data
structures. Layer 4. The only place the tool touches actual running
kernel memory.

**Interface.**
```python
probe = DrgnBridge(kcore_path=None, vmlinux_path=None)
probe.start()                               # opens from_kernel() or from_core_dump()
probe.list_processes() -> list[ProcInfo]
probe.runqueue(cpu_id) -> list[ProcInfo]
probe.read_struct(name, address) -> dict
probe.read_field(struct_name, addr, field) -> Any
```

**Mechanism.** Lazy import of drgn (for portability — the file imports
cleanly on macOS). `from_kernel()` opens `/proc/kcore` + auto-loaded
vmlinux; `from_core_dump(path)` opens a saved core. For non-Linux hosts
returns curated mock data so that the rest of the pipeline stays
exercisable.

**Observations.**
- Uses `task.__state` (dunder-prefixed), which is Linux 5.14+ only. On
  older kernels the attribute is `task.state` (F-20). Needs a runtime
  probe to pick the right name.
- `_default_fields` map hardcodes a small set of structs; any other
  struct falls back to a generic walk that can explode if the struct
  has embedded pointers to enormous substructures (F-21). Needs a
  depth cap.
- `cpu_rq` helper call in `runqueue()` assumes `drgn.helpers.linux`
  import succeeded; no graceful degrade path if that helper fails to
  load (F-22).

### 2.8 `core/synthesis/synthesizer.py` — prompt assembly + LLM dispatch

**Intent.** Glue Mirror context and Probe context into a prompt, dispatch
to the user's chosen LLM backend.

**Interface.**
```python
s = KernelSynthesizer(backend="ollama:deepseek-coder:6.7b")
answer = s.answer(question, theory_ctx, reality_ctx, graph_ctx)
```

**Mechanism.** Builds a SYSTEM → THEORY → REALITY → QUESTION → INSTRUCTIONS
prompt. Backends: Ollama (local HTTP), OpenAI, Anthropic. The `_ollama_http`
fallback lets the tool work without the `ollama` Python SDK.

**Observations.**
- No streaming — the CLI waits for full completion (F-23). For local
  models this is the difference between "instant" and "15 seconds."
- Token budgeting is crude: concatenates all context and hopes the model
  accepts it. With long graph neighborhoods this blows past the
  context window silently (F-24).
- No citation in the output — the LLM answers without pointing at
  which Mirror nodes or Probe reads supported the answer (F-25).
  Retrieval-augmented systems are judged by citation quality; this is
  a product problem, not just a polish item.

### 2.9 `tools/xray.py` — Q&A pipeline

**Intent.** One entry point that answers "what's going on?" questions by
deciding between pattern match, vector search, and live probe.

**Interface.**
```python
x = XRay(store, dwarf, kallsyms, drgn)
x.ask(question) -> Answer
x.addr2line(live_addr) -> (file, line, node) | None
```

**Mechanism.** `KNOWN_PATHS` is a list of ~17 regex patterns that map
natural-language questions to a direct tool call (e.g. "what PIDs are
running" → `drgn.list_processes()`). If no pattern matches, the tool
falls through to `store.hybrid_search` + `synthesizer.answer`. The
`addr2line` method is a full reverse traversal: live addr minus KASLR
slide gives a DWARF addr; DWARF addr goes through the line program to
yield `(file, line)`; Mirror lookup by `(file, line)` returns the
`CodeNode`.

**Observations.**
- Docstring bug near the top of the file (lines 33–35 in the audit):
  a "3. Live Probe" fragment appears *after* the closing triple quotes.
  Depending on exact line structure this is either a syntax error or
  a quietly-mis-parsed bareword assignment (F-26). Reading the file
  directly is mandatory before fixing.
- `KNOWN_PATHS` is a hand-maintained list — once we have a trained
  retriever (§6), this should become a learned classifier with a
  fallback to the patterns (F-27).

### 2.10 `cli/ktalk.py` — CLI

**Intent.** Expose everything via a `click` group.

**Interface.** `ktalk index | ask | xray | probe | stats | graph | addr2line | twin`.

**Observations.**
- `addr2line_cmd` does `__import__("pathlib").Path()` to construct a
  Path, which is a working but odd pattern (F-28) — suggests earlier
  import issues that were worked around. Make the import explicit.
- No `ktalk eval` command; there is no way to measure retrieval quality
  from the CLI (F-29). This is the single biggest gap for the training
  pipeline: without an eval command, you cannot measure whether any
  change improved retrieval.
- No progress bars in `index` — on a full kernel tree (~70k files) the
  user gets no signal that anything is happening (F-30).

---

## 3. Findings index (numbered, severity-tagged, with line evidence)

Format: `F-n | SEVERITY | file:line | short title`.
Severity: **C** (correctness), **P** (performance), **E** (ergonomic),
**D** (design gap).

| ID | Sev | File : Line | Title |
|----|-----|-------------|-------|
| F-1 | C | core/mirror/parser.py (module-scope `CORE_STRUCTS`) | Mutable module global races across parsers |
| F-2 | C | core/mirror/parser.py `embedding_text` | Long nodes silently truncated at 512 tokens |
| F-3 | C | core/mirror/parser.py `INCLUDES` resolution | Header basename collisions |
| F-4 | D | core/mirror/parser.py | No `MAYBE_CALLS` for function pointers / vtables |
| F-5 | P | core/mirror/graph.py `resolve_edges` | O(N²) edge resolution |
| F-6 | C | core/mirror/graph.py `save_graphml` / `load_graphml` | Complex attrs lost on round-trip |
| F-7 | D | core/mirror/graph.py `neighborhood` | No edge weighting in BFS |
| F-8 | P | core/mirror/graph.py | No `by_name` secondary index |
| F-9 | C | core/mirror/embedder.py + parser pipeline | `chunk_text` defined but never called in indexing path |
| F-10 | E | core/mirror/embedder.py `_auto_device` | MPS + progress bar flake |
| F-11 | C | core/mirror/store.py `vector_search` reconstruction | `docstring` dropped on reconstruction |
| F-12 | C | core/mirror/store.py `hybrid_search` | Context nodes not deduped across primaries |
| F-13 | D | core/mirror/store.py persistence | No graph-vs-collection version pairing |
| F-14 | C | core/dwarf/bridge.py cache key | mtime-only cache key |
| F-15 | P | core/dwarf/bridge.py `_parse_line_program` | All CUs parsed eagerly |
| F-16 | C | core/dwarf/bridge.py `_parse_struct` | Anonymous unions/structs not flattened |
| F-17 | E | core/probe/kallsyms.py `ANCHOR_SYMBOLS` | `do_fork` removed on modern kernels |
| F-18 | E | core/probe/kallsyms.py `ANCHOR_SYMBOLS` | `sys_read` vs `__x64_sys_read` |
| F-19 | E | core/probe/kallsyms.py `is_available` | Side-effectful "check" method |
| F-20 | C | core/probe/drgn_bridge.py field access | `task.__state` breaks on <5.14 |
| F-21 | C | core/probe/drgn_bridge.py `read_struct` fallback | No recursion depth cap |
| F-22 | E | core/probe/drgn_bridge.py `runqueue` | `cpu_rq` import not guarded |
| F-23 | E | core/synthesis/synthesizer.py | No streaming |
| F-24 | C | core/synthesis/synthesizer.py prompt assembly | No token-budget enforcement |
| F-25 | D | core/synthesis/synthesizer.py | No citations in output |
| F-26 | C | tools/xray.py (~lines 33–35) | Stray `3. Live Probe` after closing docstring |
| F-27 | D | tools/xray.py `KNOWN_PATHS` | Hand-maintained patterns, no learned intent classifier |
| F-28 | E | cli/ktalk.py `addr2line_cmd` | `__import__("pathlib")` workaround |
| F-29 | D | cli/ktalk.py | No `eval` subcommand — retrieval quality unmeasurable |
| F-30 | E | cli/ktalk.py `index` | No progress bar during indexing |

---

## 4. Phase 1 — Correctness + Performance pass

**Goal of this phase.** Get all `C`-severity findings to zero, cut indexing
wall-clock time by at least 30% on a reference kernel tree, and introduce
a bench harness so future phases can be measured.

**Entry state.** Current main, no other phases underway.

**Exit criteria.**
- All `C`-severity findings from §3 either fixed or converted to a tracked
  known-limitation note in a new file `KNOWN_LIMITATIONS.md`.
- `tests/` directory exists with unit tests for each bug fix.
- `ktalk bench index /path/to/linux` is a new subcommand that reports
  total time, files/sec, and peak RSS.
- `ktalk eval retrieval --gold eval/retrieval_gold.jsonl` prints
  Recall@k and nDCG@k on a held-out question set.

**Ordering (dependency-respecting).**

1. **F-26 first.** Fix the tools/xray.py docstring. If it is truly a
   syntax error, nothing else in that module tests. Open the file, find
   the stray lines, decide whether they belong inside the docstring or
   should be deleted, and make a single-line edit.
2. **F-11, F-12.** Store reconstruction and dedup. Write a test that
   indexes three small CodeNodes, round-trips them through `vector_search`,
   and asserts `docstring` survives. Write a second test for
   `hybrid_search` that uses overlapping graph neighborhoods and asserts
   context nodes are unique.
3. **F-5, F-8.** Add a `dict[str, list[str]]` name→id index to KernelGraph,
   built in `add_node`. Change `resolve_edges` to consult it. Profile
   indexing before/after; target ≥30% speedup on the critical path.
4. **F-6, F-13.** Persistence. Pick between: (a) switching serialization
   from GraphML to pickle (simpler, loses human-readability); or
   (b) adding a sidecar JSON for complex attributes (keeps GraphML as
   the human-facing representation). I recommend (b) — it preserves the
   property that a human can open the graph file and see structure.
   Add a `version` field in both the graph sidecar and the ChromaDB
   collection metadata; `KernelStore.open()` asserts they match.
5. **F-2, F-9.** Connect the embedder's `chunk_text` into the indexing
   path. When a `CodeNode`'s `embedding_text()` exceeds the tokenizer
   limit, produce multiple chunks, embed each, and either (i) store all
   chunks with the same parent node ID, or (ii) mean-pool and store one.
   Option (i) is better for retrieval (finer granularity); (ii) is
   simpler. Go with (i) and mark the chunk index in metadata.
6. **F-14, F-15, F-16.** DWARF robustness. Replace the mtime cache key
   with `(size, mtime, crc32(first_4KB), crc32(last_4KB))`. Make the
   line program parser lazy-by-CU. For anonymous unions/structs, recurse
   through unnamed DIEs when parsing `DW_TAG_structure_type`.
7. **F-20, F-21, F-22.** drgn robustness. Add a runtime probe that
   checks whether `task_struct` has `__state` or `state` and picks the
   right name. Add a depth cap to `read_struct`'s fallback walk. Wrap
   the `cpu_rq` import in a try/except that emits a clear "kernel
   helper not available on this drgn version" message.

**Deliverables.**
- `tests/` with at least one test per fix.
- `ktalk bench index` and `ktalk eval retrieval` commands.
- `eval/retrieval_gold.jsonl` starter file with ~30 hand-written
  `(question, expected_node_id)` pairs drawn from the kernel scheduler
  subsystem.

**How to verify you got it right.** Each test should *fail* against the
current code and *pass* after your fix. No test is allowed to be introduced
after the fix lands without first being shown to fail against the
unmodified code (this is the "red then green" discipline; it's the
difference between a test that measures the fix and a test that merely
pattern-matches the fix).

---

## 5. Phase 2 — Vocabulary spec for the Modification module

This is the user-facing language of the new subsystem we're building
(§8). It is intentionally decoupled from Git even though the underlying
storage may resemble a git-like object model. Users do not learn "commit,
push, pull" to use this tool. They learn words that describe what they
are actually doing.

### 5.1 The vocabulary

| Term | Meaning | What happens |
|------|---------|--------------|
| **Propose** | User (or an agent) suggests a change to the kernel source | A new *Proposal* is recorded, scoped to a set of files and a set of hunks. Nothing is applied yet. |
| **Review** | User inspects a Proposal before deciding | Show the diff, show the affected Mirror nodes, show live-state warnings if the Proposal touches something drgn knows is active. |
| **Accept** | Decide a Proposal should become part of the Pending set | Move Proposal from *proposed* to *pending*. Still not applied to files. |
| **Discard** | Decide a Proposal should not happen | Move Proposal to *discarded* (archived, not deleted; can be re-examined). |
| **Pending** | The set of Accepted Proposals not yet applied | Lives as a stack; order matters (later patches may depend on earlier). |
| **Preview** | Show the user what the kernel tree would look like if Pending were applied | Materializes patches into a shadow tree; runs Mirror re-index on it; shows new/removed/changed Mirror nodes. |
| **Apply** | Actually write Pending into the real kernel tree | Creates a *Snapshot* of the pre-Apply state and writes the patches. From here the user runs their normal `make` / build. |
| **Revert** | Roll the kernel tree back to a prior Snapshot | Restores files to a named Snapshot; Pending is preserved untouched. |
| **History** | The ordered list of Snapshots | Read-only log. Each entry has a timestamp, a description, and a manifest of changed files. |

### 5.2 State machine

```
    (Propose)                           (Accept)
   ┌─────────┐                       ┌──────────┐
   │         │                       │          │
   ▼         │                       ▼          │
Proposed ────┼───(Discard)──▶  Discarded       │
   │         │                                   │
   │   (Accept)                                  │
   ▼                                             │
Pending ───(Preview: look but no write)          │
   │                                             │
   │   (Apply)                                   │
   ▼                                             │
Applied ───▶ Snapshot on History              ◀──┘
   │
   │   (Revert to earlier Snapshot)
   ▼
Previously-applied state restored; Pending unchanged.
```

### 5.3 Why this vocabulary, specifically

Every term names a user-visible action whose English meaning is the same
as its technical effect. Contrast with Git: "push" technically means
"send refs to a remote," but in practice users are taught "push to share
your work," which conflates two concepts (publishing and replicating).
That conflation is the reason new developers are confused by `git push`;
they learn what it does, not what it is.

In this spec:
- **Propose vs Accept vs Apply.** Three stages because three different
  decisions are being made. Should we even suggest this? Do we like it
  enough to queue it? Are we ready to touch the kernel tree? Collapsing
  any of these three loses information.
- **Preview vs Apply.** The dangerous action is writing to files that
  the user's build toolchain will compile. Preview gives you the
  knowledge without the commitment. Apply is the one word that makes
  files change on disk.
- **Snapshot vs Revert vs History.** A Snapshot is a noun (a thing
  that exists); Revert is a verb (an action you take); History is the
  collection. The words distinguish the data model from the operation
  from the catalog — three concepts, three words.

### 5.4 CLI surface

```
ktalk mod propose  --from <file> [--description "..."] [--agent <name>]
ktalk mod review   [<proposal-id> | --all]
ktalk mod accept   <proposal-id>
ktalk mod discard  <proposal-id>
ktalk mod pending  (list the accepted-not-applied stack)
ktalk mod preview  (materialize pending into a shadow tree, re-index Mirror)
ktalk mod apply    [--dry-run]
ktalk mod history  (list snapshots)
ktalk mod revert   <snapshot-id>
```

Note the absence of `commit`, `push`, `pull`, `stage`, `unstage`, `reset`,
`checkout`. Those are Git's internal grammar; we don't inherit it.

---

## 6. Phase 3 — Training pipeline and first neural network

**The key question:** of all the neural networks we *could* train for
Kernel-Talk, which one, trained first, gives the biggest measurable
improvement to the system?

**Claim.** The first model to train is a **retrieval reranker** — a cross-
encoder that takes `(query, candidate_node)` pairs and outputs a score,
trained to order candidates so that ground-truth answers are at the top.

### 6.1 Why reranking first

Five reasons, in order of weight.

1. **Retrieval is the bottleneck.** Every downstream failure mode
   (wrong answer, missing context, hallucinated function name) traces
   back to retrieval missing the right node. If the retriever doesn't
   surface `copy_process` for a question about `fork`, no amount of
   smarter prompting fixes it.
2. **Measurability.** Retrieval has a crisp metric: nDCG@k, Recall@k,
   MRR. You can measure whether a training run helped. This is not
   true for, say, a "what should I ask next" suggester.
3. **Data is bootstrappable.** You don't need hand-labeled question-
   answer pairs. You can mine `(query, positive, negatives)` triplets
   from the kernel source itself: a commit message is a query; the
   files/functions it touches are positives; random other files are
   negatives. The Linux kernel has 30+ years of commits.
4. **Cross-encoders are cheap to train.** A 110M-param bi-encoder from
   CodeBERT + a shallow cross-encoder re-ranker on top can be trained
   in hours on a single consumer GPU on ~100k triplets.
5. **Integration is clean.** The existing pipeline already has a
   `vector_search → hybrid expansion → LLM` flow. Adding a reranker is
   "take top-N from vector_search, rescore with the reranker, expand
   the new top-k." No architectural upheaval.

Other candidates considered and deferred:
- **Call-graph completion NN** (predicts likely callers/callees for
  indirect calls). Good idea but requires annotated indirect-call data
  we don't have yet, and the payoff is visible only in graph neighborhood
  quality which is a second-order effect.
- **Intent classifier for `KNOWN_PATHS`.** Useful but tiny payoff
  against a 17-pattern rule table. Defer.
- **Code-to-code translator** (fix outdated APIs in proposals). Belongs
  in Phase 4 (Modification), not Phase 3.

### 6.2 Data pipeline

**Sources.**
- *Kernel git log.* `git log --name-only --pretty=format:"%H|%s|%b"`
  on a Linux mirror. Gives `(commit_hash, subject, body, changed_files)`.
- *Mirror index.* Our own `KernelGraph` and Chroma collection.
- *Kernel.org bug reports / LKML subjects* as a secondary source of
  natural-language kernel questions.

**Triplet mining.**
For each commit with a short subject line and ≥1 changed function:
- *Query* ← commit subject + first sentence of body.
- *Positive(s)* ← every `CodeNode` whose file+line range intersects
  the commit's changed lines.
- *Negatives*: (i) BM25 hard negatives (close-but-wrong — same file but
  different function, or same function family but different subsystem);
  (ii) random easy negatives from unrelated subsystems.

Target dataset size: 100k triplets for the first training run.

**Held-out test set.** Split by commit date: train on everything before
a chosen cutoff, evaluate on commits after. This prevents the model
from "knowing the future" through subsystem-specific vocabulary that
only exists post-cutoff.

### 6.3 Model architecture

**Bi-encoder (already exists).** CodeBERT, L2-normalized, used for
ChromaDB. Keep as-is in Phase 3. Fine-tune only if reranker alone
doesn't close the gap.

**Cross-encoder reranker (new).**
- Base: `microsoft/codebert-base` (encoder-only, 110M params).
- Input: `[CLS] query [SEP] candidate_text [SEP]`.
- Head: a single linear layer on `[CLS]` → scalar.
- Loss: margin-based ranking loss on (query, positive, negative)
  triplets, or pointwise BCE if we have graded relevance.
- Optimizer: AdamW, lr 2e-5, warmup 10%, cosine decay.
- Batch: 32 triplets (= 96 forward passes per batch), gradient
  accumulation to effective batch 128 if VRAM-bound.

**Why cross-encoder and not a bigger bi-encoder.** Cross-encoders
outperform bi-encoders of similar size on reranking because they can
attend across query and candidate jointly. The cost — having to
re-encode for every candidate — is acceptable because we rerank only
the top 100 from the bi-encoder.

### 6.4 Training script layout

```
training/
├── README.md
├── data/
│   ├── mine_triplets.py       # git log + mirror → triplets.jsonl
│   ├── build_hard_negatives.py
│   └── split_by_date.py
├── models/
│   └── reranker.py             # nn.Module + forward pass
├── train.py                    # loop + logging
├── eval.py                     # nDCG@k, Recall@k vs gold
└── configs/
    └── reranker_v1.yaml
```

**`train.py` skeleton pseudocode** (for the junior agent to expand):
```
load config
load tokenizer, model, optimizer
stream triplets from disk
for epoch in ...:
  for batch in loader:
    q_pos = tokenize(query, positive)
    q_neg = tokenize(query, negative)
    s_pos = model(q_pos)
    s_neg = model(q_neg)
    loss = margin_loss(s_pos, s_neg)
    step, log, eval every N steps
save best checkpoint by eval nDCG
```

### 6.5 Evaluation harness

`ktalk eval retrieval` (§4 deliverable) measures three settings:
1. **Bi-encoder only** (baseline).
2. **Bi-encoder + rule-based rerank** (code-length, type priors, etc.).
3. **Bi-encoder + learned reranker**.

A reranker is worth keeping only if (3) beats (2) on held-out Recall@5
by more than the noise floor across 3 seeds.

### 6.6 Training hardware reality check

Rough budget on common consumer hardware:
- RTX 3090 / 4090 (24 GB): 100k triplets, 3 epochs, ~6 hours.
- RTX 3060 (12 GB): same dataset, ~18 hours, batch halved.
- M2 Max (MPS): feasible for the bi-encoder, cross-encoder will be
  slower due to attention-pattern handling on MPS; expect 2–3x wall-
  clock vs CUDA.
- CPU only: not recommended for training; evaluation and inference only.

---

## 7. Phase 4 — Hardware detection `setup.sh`

**Goal.** On first run, detect the user's compute environment and produce
a reproducible Python venv with the right PyTorch wheel, the right
CUDA/ROCm/oneAPI runtime, and the right fallback if nothing is
accelerated.

### 7.1 Decision tree (the full logic)

```
setup.sh starts
├── detect OS:
│   ├── Darwin (macOS):
│   │   ├── arch = arm64 → install torch with MPS, accept that CUDA is
│   │   │       impossible; ask user if they want to proceed with MPS.
│   │   └── arch = x86_64 → install torch CPU; warn that Intel-Mac
│   │           acceleration is CPU-only now.
│   ├── Linux:
│   │   ├── NVIDIA GPU present? (`lspci | grep -i nvidia` or `nvidia-smi`)
│   │   │   ├── yes:
│   │   │   │   ├── query `nvidia-smi --query-gpu=driver_version` — is it
│   │   │   │   │    ≥ the minimum for the desired CUDA?
│   │   │   │   │   ├── yes → install torch cu121 wheel (or cu124, per
│   │   │   │   │   │         config), install cuDNN if not present.
│   │   │   │   │   └── no  → ask user to update driver; do NOT silently
│   │   │   │   │             install a driver (kernel module requires
│   │   │   │   │             reboot + explicit consent).
│   │   │   │   └── detect compute capability; if < 7.0 warn that modern
│   │   │   │       PyTorch may not support it.
│   │   │   └── no → continue to AMD check.
│   │   ├── AMD GPU present? (`lspci | grep -i 'amd\|ati' | grep -i vga`)
│   │   │   ├── yes and ROCm supports this GPU (consult the allow-list):
│   │   │   │     install torch rocm5.7 wheel, set HIP_VISIBLE_DEVICES.
│   │   │   │   For unsupported AMD GPUs (integrated APUs, very old
│   │   │   │   discrete), fall through to CPU.
│   │   │   └── no → continue to Intel check.
│   │   ├── Intel Arc / Xe GPU present? (`lspci | grep -i 'Intel.*\(Arc\|Graphics\)'`
│   │   │                                 and `dpkg -l intel-compute-runtime`)
│   │   │   ├── yes → install torch + intel-extension-for-pytorch;
│   │   │   │         verify `torch.xpu.is_available()` after install.
│   │   │   └── no → CPU only.
│   │   └── CPU only → install torch CPU wheel; proceed with a stern
│   │            banner that training is slow.
│   └── Windows / WSL2:
│       ├── WSL2 with CUDA passthrough → same as Linux NVIDIA path.
│       └── Native Windows → CPU-only (CUDA on native Windows is
│             supported but we don't recommend for this project
│             because kernel work happens on Linux).
└── after torch install:
    ├── verify `torch.cuda.is_available()` / `torch.backends.mps.is_available()`
    │   / `torch.xpu.is_available()` matches the branch we took;
    ├── create `.venv/` with the install;
    ├── install `requirements.txt`;
    ├── write `~/.kernel-talk/env.yaml` recording the branch chosen
    │   so subsequent runs don't re-detect;
    └── print a summary.
```

### 7.2 Script structure

```
scripts/
├── setup.sh                 # entry point
├── detect/
│   ├── os.sh
│   ├── nvidia.sh
│   ├── amd.sh
│   ├── intel.sh
│   └── apple.sh
├── install/
│   ├── torch_cuda.sh
│   ├── torch_rocm.sh
│   ├── torch_xpu.sh
│   ├── torch_mps.sh
│   └── torch_cpu.sh
└── verify.sh               # torch.cuda.is_available() etc.
```

Every `detect/*.sh` returns 0 if that hardware is present, non-zero
otherwise, and writes a single line to stdout with a version string
(e.g. `nvidia:535.104.05:12.2` or `amd:gfx1030:5.7`). `setup.sh`
reads these in order and picks the first that succeeds.

### 7.3 Non-obvious things the script must do

- **Never install kernel drivers silently.** NVIDIA / AMD driver
  installation changes the running kernel's module set and typically
  requires a reboot. The script may *suggest* a driver update and print
  the exact command, but must not execute it without explicit `--yes`.
- **Respect existing venvs.** If `.venv/` already exists and
  `~/.kernel-talk/env.yaml` matches the detected branch, skip re-install
  and just activate.
- **Pin wheel indexes.** PyTorch CUDA wheels live on
  `download.pytorch.org/whl/cu121`; ROCm wheels on `.../whl/rocm5.7`;
  CPU on `.../whl/cpu`. Hard-code the chosen index so we don't pull
  a surprise wheel from PyPI.
- **Probe before trusting.** After any install, run a small Python
  snippet that does `import torch; print(torch.cuda.is_available())`
  (or the equivalent) and fail loudly if it's not what we expected.
- **Log every decision.** Append to `~/.kernel-talk/setup.log` so a
  user who gets help can show the maintainer exactly what happened.

### 7.4 Failure modes and recoveries

| Symptom | Likely cause | Script behavior |
|---------|-------------|-----------------|
| `nvidia-smi` exists but `torch.cuda.is_available()` is False after install | CUDA version mismatch (wheel vs runtime) | Print both versions and the offending mismatch; suggest installing the matching wheel. |
| ROCm wheel installs but `torch.cuda.is_available()` is False (ROCm masquerades as CUDA in torch) | Wrong ROCm version or GPU not on allow-list | Downgrade branch to CPU; print a pointer to ROCm's hardware support page. |
| Apple silicon host but `torch.backends.mps.is_available()` is False | PyTorch version too old, or running under Rosetta | Check `arch` — if x86_64 under Rosetta, instruct user to run under native arm64; otherwise upgrade torch. |
| CPU fallback, user expected GPU | Hardware not detected | Print the full output of each detect/*.sh so user can see which check failed and fix it manually. |

### 7.5 Testing strategy for the script

The script can't be fully tested on one machine because it branches on
hardware. Structure for CI and manual test:
- **Unit test each `detect/*.sh`** with a mocked `lspci` / `nvidia-smi`
  binary that prints a canned response.
- **Smoke test each `install/*.sh`** in disposable VMs or containers
  (ROCm docker image, CUDA docker image, plain Ubuntu for CPU).
- **End-to-end** on at least one real host per family before release.

---

## 8. Phase 5 — Patch-based Kernel Modification module

This is the biggest new subsystem. It does *not* write to kernel memory.
It writes to the kernel *source tree*. After Apply, the user's normal
build toolchain (`make bzImage` etc.) compiles and installs the new
kernel. A reboot into the new kernel is required for any change that
isn't a loadable module.

This preserves the fundamental invariant that drgn stays read-only and
the Probe layer never mutates. The *digital twin* can reflect the
proposed world via Preview; the *real kernel* only changes through a
build.

### 8.1 Data model

```
Proposal
├── id                  : uuid4
├── description         : str
├── author              : str ("user" | "<agent-name>")
├── base_snapshot_id    : uuid4  (the snapshot this was authored against)
├── hunks               : list[Hunk]
└── state               : "proposed" | "accepted" | "discarded" | "applied"

Hunk
├── file_path           : relative to kernel root
├── before_context      : list[str]  (N lines before the change)
├── removed_lines       : list[(line_no, text)]
├── added_lines         : list[(line_no, text)]
├── after_context       : list[str]
└── affected_nodes      : list[node_id]  (from Mirror; computed)

Snapshot
├── id                  : uuid4
├── timestamp           : iso-8601
├── description         : str
├── parent_snapshot_id  : uuid4 | None
├── applied_proposals   : list[proposal_id]
├── tree_hash           : sha256 of the kernel tree at this snapshot
└── manifest            : list[(file_path, sha256)]
```

### 8.2 Storage layout

```
~/.kernel-talk/mod/
├── proposals/
│   └── <proposal-id>.yaml
├── snapshots/
│   ├── <snapshot-id>/
│   │   ├── meta.yaml
│   │   └── files/           (copy-on-write: only changed files stored)
│   └── index.yaml           (ordered history)
└── pending.yaml             (the accepted stack)
```

Snapshots use *copy-on-write* at the file level: a snapshot stores only
the files that differ from its parent, along with a manifest that
covers the full tree via references up the parent chain. Revert walks
parents and overlays.

### 8.3 End-to-end flows

**Propose** (from an agent):
1. Agent produces a unified diff against the kernel tree.
2. `ktalk mod propose --from agent.diff --agent reranker-refactor` reads
   the diff, splits into hunks, computes `affected_nodes` by intersecting
   changed line ranges with Mirror nodes.
3. Proposal is written to `proposals/<id>.yaml` in state `proposed`.

**Review**:
1. `ktalk mod review <id>` opens a Rich-formatted view:
   - the diff with colorized context;
   - for each `affected_node`, the Mirror docstring + signature;
   - any drgn warnings (e.g. "this function is currently scheduled on
     CPU 3" — not a block, just information).
2. User sees enough to decide.

**Accept / Discard**:
1. `accept <id>` transitions the Proposal's state and appends to
   `pending.yaml`. The state is `accepted`.
2. `discard <id>` transitions to `discarded`. Proposal is never
   removed — it stays on disk under `proposals/` in case the user wants
   to reconsider.

**Pending**:
1. `ktalk mod pending` lists the stack in order. The stack order matters
   because later proposals may reference changes from earlier ones.

**Preview**:
1. `ktalk mod preview` materializes the stack into a *shadow tree*:
   copy-on-write overlay on top of the current kernel tree.
2. Runs Mirror re-index (incremental; only files in the overlay are
   re-parsed).
3. Reports: new nodes, removed nodes, changed nodes, any retrieval
   ambiguities introduced (e.g. two functions with the same name now
   exist).
4. Shadow tree is thrown away at end of command — Preview never writes
   into the real kernel tree.

**Apply**:
1. `ktalk mod apply` takes a Snapshot of the current tree first
   (COW against the previous snapshot or root).
2. Writes all pending hunks to the real kernel tree.
3. Transitions each proposal's state to `applied`.
4. Appends a new Snapshot record to `snapshots/index.yaml`.
5. Clears `pending.yaml`.
6. Prints the next step: "build your kernel with `make -j$(nproc)
   bzImage modules`; install with the appropriate command for your
   distro; reboot."

**Revert**:
1. `ktalk mod revert <snapshot-id>` restores files to the state
   recorded in that snapshot by walking the manifest.
2. Pending is untouched (revert is about the tree, not about the queue).
3. A new Snapshot is recorded (reverts are also history entries).

**History**:
1. `ktalk mod history` prints the snapshots oldest-first with
   description and applied proposals.

### 8.4 Safety rails

Some of these are non-negotiable:

1. **Apply refuses to run if the kernel tree has uncommitted edits
   outside the system's control.** If the user edited files by hand
   between Propose and Apply, the hunks may no longer apply cleanly.
   Apply detects this by comparing the current file hash against the
   hash at the time of Propose; mismatch → abort with a clear diagnostic.
2. **Apply never silently drops a hunk.** If any hunk fails to apply,
   the entire Apply rolls back and nothing is written.
3. **Preview is the only way to re-index Mirror on a proposal.** This
   makes Mirror's state correspond either to the real tree or to the
   explicit Preview, never to a half-applied state.
4. **Proposals are immutable once accepted.** An accepted proposal
   on the Pending stack cannot be silently edited. To change it:
   discard and re-propose. This keeps History honest.
5. **The tool never runs `make` for the user.** The kernel build is a
   high-consequence action with distro-specific post-install steps
   (grub, signing, initramfs). We print the right commands; the human
   runs them.

### 8.5 Integration with the rest of the system

- **Mirror.** Preview triggers an incremental re-index. Every `CodeNode`
  gets a `snapshot_id` in its metadata; retrieval can be filtered to
  a specific snapshot.
- **DWARF / Kallsyms.** These bind to the *binary*, which only changes
  after a build + reboot. After reboot, the user re-runs `ktalk index
  --vmlinux <new_vmlinux>` and the bridges rebind.
- **drgn.** Entirely untouched by the Modification module at apply
  time. Probe of the *new* kernel happens only after the user rebuilds
  and reboots.
- **Synthesizer.** When the user asks "how would this change affect X,"
  Preview's shadow Mirror becomes the THEORY context; REALITY context
  remains the running (pre-rebuild) kernel. The prompt explains the
  mismatch to the model.

### 8.6 Why this is *not* a kernel-live-patching system

Because the system writes source and waits for the user to build,
three risky things that live-patching has to solve are sidestepped:
- No ftrace / KGraft / livepatch infrastructure on the running kernel.
- No need to handle "in-flight system calls are still running the old
  code path."
- No requirement that the change be semantically compatible with the
  existing stack frame layout.

This is a deliberate architectural choice: livepatching is a feature
for production-runtime patching (CVE backports on live servers). For
a research + development tool, rebuild-and-reboot is simpler, more
forgiving, and composable.

---

## 9. Handoff notes for whoever implements this

For a junior agent reading this and deciding what to do first:

1. Start with §4 (Phase 1). Do not touch any other phase until Phase 1
   is green on the eval harness.
2. When fixing, open *one* finding from §3 at a time. Write the red
   test, make it green, move on. Don't batch fixes across modules.
3. When you hit something not covered here, add it to
   `OPEN_QUESTIONS.md` and surface it to Lee before guessing.
4. The semi-formal-reasoning certificate from the skill of the same
   name is the preferred format for any correctness claim — especially
   in the Modification module, where "does Apply produce the same tree
   as Preview?" is a property you want to prove, not just test.
5. The evaluation harness is the load-bearing artifact of Phase 3 and
   everything that follows. A change you can't measure is not a change
   you can defend.

Open questions I don't have a strong opinion on yet:
- Should Pending be a stack or a DAG? A stack is simpler; a DAG handles
  the case where two Proposals are independent and can be applied in
  any order. For now: stack, with a future migration path to DAG.
- Should Snapshots store full file contents or unified diffs from their
  parent? I recommend full contents at snapshot time (storage is cheap,
  seek-and-overlay during Revert is simpler). But for users on tiny
  disks, a diff-based option is a reasonable setting.
- When a Proposal's `affected_nodes` overlaps with a currently-running
  function (per drgn), should Review block, warn, or do nothing? My
  recommendation is *warn but don't block* — the information is useful,
  but blocking would be paternalistic since we're not live-patching.

---

## 10. What this document is and is not

This document is a plan, not a build. Every code file referenced above
exists; every finding has a line citation. What does not yet exist:
the fixes, the training pipeline, the setup script, and the Modification
module. Those are the units of work for future sessions.

If you, the implementing agent, discover that a claim in this spec is
wrong — the line number is off by a few, the finding doesn't reproduce,
the vocabulary doesn't feel right when you try to use it — push back.
Write the correction into this file and surface the disagreement. That
is the intended mode of operation. A spec is better when its readers
improve it.

— end of spec —
