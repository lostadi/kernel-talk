# Open Questions

These are genuine unresolved questions about the design of Kernel-Talk.
They are not bugs — they are places where multiple defensible choices exist
and the right answer depends on empirical measurement or deliberate design
decisions we haven't made yet.

---

## Retrieval Architecture

**Q1. Should chunk-level vectors replace node-level vectors?**

Right now `embed_nodes()` mean-pools chunk embeddings into a single per-node
vector.  An alternative is to store each chunk as a separate ChromaDB
document (with IDs like `{node.id}::chunk{n}`) and deduplicate at query time.
Chunk-level storage preserves fine-grained similarity signals but makes
`vector_search` reconstruction more complex.  Which gives better Recall@k
on the gold eval?  Run `ktalk eval retrieval` with both approaches to measure.

**Q2. Should the graph expansion direction be tuned per query type?**

`neighborhood()` expands in both directions (successors + predecessors).
For a query like "what calls schedule?" we want predecessors.
For "what does schedule() depend on?" we want successors.
Could a query classifier route graph expansion direction?
Or should we always expand both and let the LLM filter?

**Q3. What is the right retrieval weight ratio for primary vs context nodes?**

The synthesizer receives both primary (vector-scored) and context (graph-expanded)
nodes.  Currently they're concatenated.  Should context nodes be down-weighted
in the prompt?  Should we compute a graph-distance score and blend it with the
vector score to produce a unified ranking?

**Q4. How should the system handle cross-subsystem queries?**

A query like "how does the scheduler interact with cgroups" pulls from
`kernel/sched/` and `kernel/cgroup/`.  These subsystems may not be co-indexed
if the user ran `ktalk index --subsystem kernel/sched`.  Should `hybrid_search`
detect multi-subsystem queries and merge results from multiple sub-indexes?

**Q5. Is CodeBERT the right embedding model for kernel C?**

CodeBERT was trained on GitHub code+docstring pairs, predominantly Python/Java/JS.
It has never seen a `struct task_struct` during pretraining.  Alternatives:
- Fine-tune CodeBERT on Linux kernel commit messages + changed code
- Use `StarEncoder` or `UniXcoder` (also code-aware but broader training)
- Distill a smaller model from CodeBERT reranking outputs

The retrieval gold eval (`eval/retrieval_gold.jsonl`) is the measurement
instrument to answer this question empirically.

---

## Digital Twin Accuracy

**Q6. How do we know the DWARF bridge is accurate?**

The DWARF bridge maps symbol names → address ranges via `_parse_subprogram()`.
But the Linux kernel uses heavy macro expansion, inline assembly, and LTO.
Some functions may not appear as `DW_TAG_subprogram` DIEs at all.
We need a ground-truth check: compare `bridge.symbol_to_addrs("schedule")`
against `/proc/kallsyms` output (adjusted for KASLR slide) for a sample of
100 known symbols and measure match rate.

**Q7. Is the KASLR slide stable within a boot session?**

The current code computes the slide once via `kaslr_slide()` and caches it.
If the system has live patching enabled (kpatch, livepatch), individual
function addresses can move within a session.  Should the slide be
recomputed per-query instead of once?  Or should we track individual
function slides for patched symbols?

**Q8. What is the right set of DWARF anchor symbols for non-x86 architectures?**

The current `ANCHOR_SYMBOLS` list is x86-64 biased (`startup_64`, `__x64_sys_read`).
On arm64 the entry point is `__primary_switched`; on RISC-V it's `soc_early_init`.
How should the anchor set be parameterized?  Should `KallsymsBridge` detect
the running architecture and load architecture-appropriate anchors?

---

## Training and Fine-tuning

**Q9. What is the right negative sampling strategy for the cross-encoder reranker?**

The Phase 3 training pipeline will fine-tune a cross-encoder reranker
on kernel git commit triplets: (commit message, changed function, unrelated function).
For negatives, we have:
- **Hard negatives**: functions from the same subsystem that weren't changed
- **Random negatives**: functions from an unrelated subsystem
- **BM25 negatives**: highest-BM25-scoring functions that aren't the gold answer

Hard negatives produce a stronger reranker but require more data to converge.
The right ratio of hard:easy negatives is an open empirical question.

**Q10. Can we use kernel bug reports (bugzilla, LKML) as additional training signal?**

A bug report that says "kswapd is burning CPU" links semantically to
`mm/vmscan.c:kswapd()`.  This is a natural language → code pair that
we don't currently use.  Is the signal:noise ratio high enough?
LKML archives go back to 1991 and contain millions of messages.

---

## System Design

**Q11. Should the Digital Twin use a time dimension?**

The kernel evolves.  `do_fork()` was removed in 5.7.  `__state` was
renamed in 5.14.  The current system has no concept of kernel version —
it indexes whatever source tree you point it at.  Should we add a `kernel_version`
tag to every CodeNode and allow version-scoped queries?  This would make
`ktalk ask` answers version-aware.

**Q12. How should the Modification module (Phase 2) handle forward pointers?**

A proposed change to `schedule()` may break callers in `kernel/time/hrtimer.c`
that we haven't analyzed.  The Modification module needs a "blast radius"
analysis: given a proposed change to symbol X, which other nodes would need
to change?  Should this use static analysis (call graph + type graph) or
semantic similarity (ask the LLM to identify dependencies)?

**Q13. What is the right granularity for the `eval/retrieval_gold.jsonl` format?**

Currently each gold entry has `expected_symbols` (exact function/struct names)
and `expected_files` (file path prefixes).  A more rigorous format would
include `expected_node_ids` (exact CodeNode.id values) to eliminate ambiguity
from static functions with the same name in different files.  But that would
require running `ktalk index` first to know what IDs exist.
Can the gold format remain independent of a particular index build?
