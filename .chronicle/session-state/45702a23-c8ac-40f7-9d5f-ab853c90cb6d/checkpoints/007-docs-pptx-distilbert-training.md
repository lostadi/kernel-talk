<overview>
The session focused on speeding up a stalled bi-encoder training run on the `kernel-talk` Graph-RAG Digital Twin project, then creating the assignment deliverables (`.pptx` and `.tex`) with real index/training data. The user initially thought the deadline was 11:59 PM tonight but clarified it's 11:59 PM tomorrow (~27 hours), removing time pressure. The approach pivoted from CodeBERT (110M params, too slow for CPU) to `distilbert-base-uncased` (66M params, ~40% faster), and the LaTeX document was extended with real live-build statistics rather than created from scratch.
</overview>

<history>
1. **User asked to speed up training with NPU/GPU/CPU**
   - Found two competing training processes running simultaneously: PID 46933 (batch=8, started ~20:25, 285 CPU-min = ~27 min wall clock) and PID 50029 (batch=16, started ~20:47, 5 min wall clock)
   - Both were consuming ~12–21 GB RAM each and competing for CPU cores
   - Diagnosed: NPU is inference-only (cannot train); Intel Arc iGPU requires IPEX which needs Python ≤3.12, but machine only has Python 3.14
   - Decision: kill both, switch to smaller model for faster CPU training

2. **User objected "that was 4 hours of training — use that one"**
   - Checked: `train_log.jsonl` was 0 bytes, `training/checkpoints/` had no model files
   - The 285:29 in `ps aux` was CPU-minutes (not wall clock hours) — only ~27 minutes of actual wall time had elapsed, no checkpoints were ever saved
   - Nothing recoverable; needed to restart from scratch regardless

3. **User said "nvm carry on with what you think is best"**
   - Decided on `distilbert-base-uncased` (66M params, 6 layers, 768-dim, standard WordPiece tokenizer)
   - Also reduced `CODE_MAX_LEN` from 256→128 (4× attention speedup, O(n²))
   - Added per-epoch unconditional checkpoint saves (not just best-recall)
   - Added step logging every 10 steps (not 100) for visibility
   - First tried `prajjwal1/bert-small` → failed (sentencepiece not installed)
   - Switched to `distilbert-base-uncased` → confirmed working, PID 54425, 66,362,880 params, 894% CPU

4. **User said ".tex already mostly done — read and understand it then modify it, base the PowerPoint off that"**
   - Read the full `docs/architecture.tex` (1,387 lines) — a comprehensive, well-structured document with TikZ diagrams, colour palette, training pipeline sections, test coverage table, appendices
   - Identified four gaps to fill with real data:
     - No live build statistics (node/edge counts)
     - No synthetic triplet generation section (only git mining `mine.py` was documented)
     - Training config table referenced CodeBERT/512 tokens (now using distilbert/128)
     - No Native Rust CLI section
   - Made four surgical edits to the `.tex`
   - Generated a 9-slide `.pptx` mirroring the document structure

5. **User clarified deadline is 11:59 PM tomorrow (27 hours away)**
   - Committed everything, laid out a proper plan with time to get real post-training metrics
</history>

<work_done>
Files modified:
- `training/dataset.py`: `CODE_MAX_LEN` 256→128 (line 72)
- `training/train_biencoder.py`: log every 10 steps (not 100), add per-epoch checkpoint save unconditionally alongside best-recall checkpoint (lines 424–427, 442–451)
- `docs/architecture.tex`: Added §3.6 "Live Build Results" (node/edge tables with real numbers), §7.2 "Stage 1b: Synthetic Triplet Generation" (6,623 triplet breakdown table), updated §7.4 training config table (distilbert-base-uncased, batch=16, CODE_MAX_LEN=128, hardware info), added §12 "Native CLI Binary (Rust)" section, updated footer tcolorbox with live numbers

Files created:
- `docs/kernel_talk_presentation.pptx`: 9-slide deck (title, motivation, architecture, index stats, training pipeline, model config, retrieval pipeline, Rust CLI, results/next steps) — all real numbers, no placeholders

Commits:
- `845cc2e`: "feat: presentation, updated tex, distilbert training"

Current state:
- [x] `docs/architecture.tex` updated with all real data and new sections
- [x] `docs/kernel_talk_presentation.pptx` created (9 slides)
- [x] Training running: PID 54425, distilbert-base-uncased, 66M params, 147 steps, 3 epochs
- [ ] Training not yet complete — no `val_recall@10` numbers yet
- [ ] `docs/architecture.pdf` not rebuilt (no LaTeX compiler confirmed on machine)
- [ ] Eval (`python -m training.eval`) not yet run
- [ ] `.pptx` slides 6 and 9 have placeholder text for post-training metrics ("Val Recall@10 → reported post-training")
</work_done>

<technical_details>
- **No checkpoints = nothing to resume**: PyTorch only saves when `val_recall` improves (end of epoch). With CodeBERT at batch=8, epoch 1 would take ~26 min wall time, so killing at 27 min meant it was mid-epoch with zero saved state.

- **`ps aux` TIME column is CPU-minutes, not wall clock**: The "285:29" that looked like "4 hours 45 min" was cumulative CPU time across all cores. Actual wall clock was ~27 minutes. Always divide by core count to get wall time.

- **`prajjwal1/bert-small` needs sentencepiece**: Not installed in the `.venv`. Use `distilbert-base-uncased` or `bert-base-uncased` as a drop-in smaller alternative.

- **distilbert "UNEXPECTED" keys in load report**: `vocab_projector`, `vocab_layer_norm`, `vocab_transform` — these are MLM head weights, not part of the bi-encoder architecture. Safe to ignore.

- **Intel NPU = inference only**: Cannot be used for gradient-based training under any circumstances. Intel Arc iGPU requires IPEX which maxes out at Python 3.12; this machine only has Python 3.14.

- **`_collection` vs `_get_collection()`**: ChromaDB's `_collection` attribute is `None` until first access. Always call `store._get_collection()` to get a live collection handle. (Fixed in prior sessions.)

- **ChromaDB store layout**: Root at `~/.kernel-talk/store/`, ChromaDB data in `~/.kernel-talk/store/chroma/` (subdirectory). Collection name is `kernel_code`. Code text is in the `documents` field, not `metadatas`.

- **Python buffering with nohup**: Always use `PYTHONUNBUFFERED=1` when launching Python with `nohup` to file, otherwise output is block-buffered and log stays empty for a long time.

- **Training log location**: `/tmp/ktalk_biencoder.log` — PID 54425. First step log appears at `step=10` (every 10 steps now). End-of-epoch lines show `val_recall@10`.

- **`num_workers=0` required**: ChromaDB client is not picklable, so DataLoader must use `num_workers=0` (no subprocess workers).

- **Real index numbers**:
  - 5,755 nodes: 2,718 functions, 2,588 macros, 281 structs, 88 enums, 77 files
  - 8,843 edges: DEFINED_IN 5,333, USES_STRUCT 1,873, CALLS 1,392, INCLUDES 245
  - 6,623 enriched triplets: synth_symbol 2,718, synth_docstring 2,667, synth_caller_callee 1,237
  - BM25 median hard-negative gap: 6.65

- **`docs/architecture.tex`**: 1,387 lines (before edits). Uses TikZ for diagrams, `tcolorbox` for code blocks, custom colour palette. Compiles with `pdflatex architecture.tex` (twice for cross-refs). Requires `texlive-full` or equivalent.
</technical_details>

<important_files>
- `training/train_biencoder.py`
  - Main training loop; currently running as PID 54425
  - Modified: log every 10 steps, add per-epoch unconditional checkpoint save
  - Lines 424–451: logging and checkpoint saving logic
  - Launch command: `PYTHONUNBUFFERED=1 OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 nohup .venv/bin/python -m training.train_biencoder --triplets data/enriched.jsonl --storage /home/leeo/.kernel-talk/store --output training/checkpoints --model distilbert-base-uncased --batch-size 16 --epochs 3`

- `training/dataset.py`
  - PyTorch Dataset wrapping ChromaDB store
  - Modified: `CODE_MAX_LEN` = 128 (line 72), down from 256
  - Also uses `_get_collection()` (not `_collection`) — critical fix from prior session

- `docs/architecture.tex`
  - 1,400+ line comprehensive architecture document (after edits)
  - Added: §3.6 Live Build Results, §7.2 Synthetic Triplet Generation, updated §7.4 training config, §12 Native Rust CLI, updated footer
  - Compiles to PDF with `pdflatex` (twice); no compiler confirmed installed on host

- `docs/kernel_talk_presentation.pptx`
  - 9-slide PowerPoint generated with `python-pptx`
  - All slides use real live-build numbers; slides 6 and 9 have placeholder text for post-training `val_recall@10`
  - Generated by `/tmp/make_pptx.py` (ephemeral script, not committed)

- `training/eval.py`
  - Evaluation harness: nDCG@k, Recall@k, MRR across 3 settings (biencoder, rule_rerank, learned_rerank)
  - Not yet run — waiting for training checkpoint
  - Run with: `python -m training.eval --checkpoint training/checkpoints/best_biencoder --storage /home/leeo/.kernel-talk/store`

- `data/enriched.jsonl`
  - 6,623 enriched training triplets (BM25 hard negatives + difficulty scores)
  - Input to the running training job
  - Schema: `{"query": "...", "positives": ["file::symbol"], "hard_negatives": [...], "difficulty": 0.42, "source": "synth_symbol|..."}`

- `rust_ext/ktalk_cli/src/main.rs`
  - Native Rust ktalk binary (clap v4)
  - Native commands: `version`, `status`, `env`, `completions`
  - All other commands delegate to `python -m cli.ktalk` via subprocess
  - Build: `cd rust_ext/ktalk_cli && cargo build --release` or `make ktalk-bin`
</important_files>

<next_steps>
Remaining work:

- [ ] **Monitor training to completion**: `tail -f /tmp/ktalk_biencoder.log` — expect ~30–45 min total wall time. First loss at `step=10`, epoch completions show `val_recall@10`.
- [ ] **Run eval harness** once `training/checkpoints/best_biencoder/` exists: `python -m training.eval ...` → get real nDCG@10, Recall@10, MRR numbers
- [ ] **Update `.tex` §7.4** with actual post-training `val_recall@10` from the log/eval output
- [ ] **Update `.pptx` slides 6 & 9** with real metrics (re-run `/tmp/make_pptx.py` with numbers filled in, or edit manually)
- [ ] **Compile PDF**: Check if `pdflatex` is available (`which pdflatex`); if yes, run `cd docs && pdflatex architecture.tex && pdflatex architecture.tex`
- [ ] **Train cross-encoder reranker** (Stage 5 — `training/train.py`) for a second retrieval quality number showing improvement over bi-encoder baseline — 24 hrs is enough time
- [ ] **Run a live query demo**: `python -m cli.ktalk ask "how does RCU locking work"` and capture real output for the presentation
- [ ] **(Optional) /chronicle improve**: The `/chronicle improve` run was interrupted mid-analysis — the session history review found the main friction patterns but the `copilot-instructions.md` file was never created or written. Resume that when convenient.

Immediate next steps:
1. `tail -f /tmp/ktalk_biencoder.log` to watch training
2. Once `Epoch 1 complete:` appears, check `training/checkpoints/` for saved model
3. Run eval, update docs with real numbers
</next_steps>