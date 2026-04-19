# kernel-talk — Project Conventions

This file documents project-specific conventions, known bug-prone areas, and
architecture decisions. Read it before starting a session.

---

## ChromaDB Conventions (Critical — bugs recur here)

The KernelStore uses a ChromaDB persistent client.  Three things **must** be
correct or the whole training/eval pipeline silently returns empty results:

| Parameter | Correct value | Common wrong value |
|-----------|---------------|-------------------|
| Storage path | `store/chroma/` (nested) | `store/` (missing `chroma/` subdir) |
| Collection name | `kernel_code` | `kernel_functions` |
| Code text field | `documents` | `metadatas["code"]` |

Code that reads from ChromaDB must call:

```python
# Correct
client = chromadb.PersistentClient(path="store/chroma")
collection = client.get_collection("kernel_code")
result = collection.get(..., include=["metadatas", "documents"])
code = result["documents"][i]   # ← code lives here
```

The `synth.py`, `bm25.py`, `dataset.py`, and `train_biencoder.py` files all
had these bugs fixed in commits 4694d29 / 4819ad5.  Do not reintroduce them.

---

## Gold Evaluation Data Format

`eval/retrieval_gold.jsonl` contains 142 queries.  Each entry has:

```json
{
  "query": "how does fork work",
  "expected_symbols": ["kernel_clone", "copy_process"],
  "expected_files": ["kernel/fork.c"],
  "relevant_ids": ["kernel/fork.c::kernel_clone", "kernel/fork.c::copy_process"]
}
```

`relevant_ids` = cross-product of `expected_files × expected_symbols`.
`training/eval.py` and `eval/retrieval.py` both auto-derive `relevant_ids`
from the legacy fields if the key is absent.

---

## Training Pipeline — Quick Reference

```bash
# 1. Index kernel source
make index KERNEL=/usr/src/linux-cachyos

# 2. Generate synthetic triplets (no git needed)
make synth

# 3. Build BM25 index + enrich with hard negatives
make bm25

# 4. Train bi-encoder
make train

# 5. Train cross-encoder reranker
make reranker

# 6. Evaluate
make eval
```

Or use the all-in-one target: `make pipeline` (skips reranker and eval).

---

## Hardware Notes (CachyOS, CPU-only)

- No GPU / NPU available for training.
- Use `OMP_NUM_THREADS=14` to utilise all cores.
- Intel IPEX requires Python ≤ 3.12; system has 3.14 — do **not** attempt NPU.
- Token lengths: `QUERY_MAX_LEN=128`, `CODE_MAX_LEN=128` (4× speedup vs 256).

---

## Key File Locations

| Purpose | Path |
|---------|------|
| KernelStore (default) | `~/.kernel-talk/store/` |
| ChromaDB data | `~/.kernel-talk/store/chroma/` |
| Synthetic triplets | `data/triplets.jsonl` |
| Enriched triplets | `data/enriched.jsonl` |
| BM25 index | `data/bm25.pkl` |
| Bi-encoder checkpoints | `training/checkpoints/` |
| Reranker checkpoints | `training/checkpoints/reranker/` |
| Eval gold queries | `eval/retrieval_gold.jsonl` |
| Eval results | `eval/results.json` |
| Architecture docs | `docs/architecture.tex` / `.pptx` |

---

## Node ID Convention

CodeNode IDs throughout the codebase follow the pattern:

```
{relative_file_path}::{symbol_name}
```

Examples:
- `kernel/sched/fair.c::tg_throttle_up`
- `mm/slab.c::kmalloc`
- `include/linux/sched.h::task_struct`

---

## Running Tests

```bash
python -m pytest tests/ -v
```

All 163 unit tests run without a kernel source tree, a ChromaDB store, or ML
dependencies (torch/transformers).

---

## Front-load Constraints

When starting a new session, include in your **first message**:
1. Current deadline / time constraint
2. Hardware limits (CPU-only, no GPU)
3. What already works vs what's broken
4. Which specific file/feature you want to change

This prevents the agent from discovering constraints 12 turns in and wasting
time on approaches that won't work on the available hardware.
