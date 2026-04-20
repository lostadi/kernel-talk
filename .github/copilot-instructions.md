# Copilot Instructions — kernel-talk

## Build & Test

Always activate the virtual environment first:
```bash
source .venv/bin/activate
```

Run the full test suite:
```bash
python -m pytest tests/ -q
```

Run a single test:
```bash
python -m pytest tests/test_parser.py::TestMacroExtraction -q
```

No LaTeX is installed on this machine — `docs/architecture.tex` can be edited but not compiled to PDF locally.

## Architecture

Three-layer pipeline:

1. **Mirror** (`core/mirror/`) — parses the Linux kernel source tree via tree-sitter into `CodeNode` objects, embeds them with CodeBERT, and stores them in ChromaDB + a NetworkX graph (`KernelGraph`).
2. **Synthesis** (`core/synthesis/`) — takes retrieval results and constructs prompts for an LLM backend (Ollama / OpenAI / Anthropic).
3. **CLI** (`cli/ktalk.py`) — Click-based entry point wiring everything together.

Training pipeline (`training/`) is separate from inference: `synth.py` → `bm25.py` → `train_biencoder.py` → `eval.py`.

## ChromaDB Store Schema

The store lives at `~/.kernel-talk/store/chroma/` (not `store/`).
- **Collection name:** `kernel_code` (not `kernel_functions`)
- **Code text:** stored in `documents` (not `metadatas`)
- **Node ID format:** `{rel_file_path}::{symbol_name}` (e.g., `include/linux/sched.h::schedule`)

Access via `KernelStore._get_collection()` — never construct the ChromaDB client directly.

## Training Process Management

Before starting any training run, always check for an existing one:
```bash
pgrep -af train_biencoder
```

If one is running, check for saved checkpoints before killing it:
```bash
ls -lh training/checkpoints/
```

Launch training with unbuffered output so logs appear immediately:
```bash
PYTHONUNBUFFERED=1 nohup python -m training.train_biencoder \
  --triplets data/enriched.jsonl \
  --storage ~/.kernel-talk/store \
  --output training/checkpoints/ \
  --epochs 3 --batch-size 16 \
  > /tmp/ktalk_biencoder.log 2>&1 &
```

Monitor:
```bash
tail -f /tmp/ktalk_biencoder.log
```

Real training metrics (epoch 1→3): loss 3.168→3.393, Recall@10 0.447→0.764.

## Key Conventions

- **Gitignore blocks `docs/*.pptx` and `docs/*.pdf`** — use `git add -f docs/<file>` to commit deliverables.
- **Linux kernel size:** 40M+ lines of C (Linux 6.14, January 2025). Do not use 5M, 30M, or 50M.
- **Gold eval format:** `eval/retrieval_gold.jsonl` uses `expected_symbols` / `expected_files` keys, not `relevant_ids`. The `_resolve_relevant_ids()` function in `training/eval.py` handles the mapping.
- **Parser macro loop:** `_extract_macros` in `core/mirror/parser.py` uses deliberate 2-space inner indentation — do not reformat it; the body scope is intentional.
- **Git remote:** `origin` points to `https://github.com/lostadi/kernel-talk.git`.
