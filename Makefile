# Makefile — Kernel-Talk
# ─────────────────────────────────────────────────────────────────────────────
# Targets:
#   make setup           — Create venv and install Python deps
#   make rust            — Build the optional Rust DWARF extension (~20× speedup)
#   make index           — Build the Mirror from /usr/src/linux-cachyos
#   make synth           — Generate synthetic training triplets
#   make bm25            — Build BM25 index + enrich triplets with hard negatives
#   make train           — Fine-tune CodeBERT bi-encoder
#   make pipeline        — Full pipeline: index → synth → bm25 → train
#   make test            — Run all tests
#   make eval            — Run retrieval evaluation on gold set
#   make clean           — Remove generated data files
#   make clean-all       — Remove venv, store, and all generated files

SHELL    := /bin/bash
PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip

# ── Configurable paths ─────────────────────────────────────────────────────────
KERNEL   ?= /usr/src/linux-cachyos
SUBSYS   ?= include/linux
STORAGE  ?= $(HOME)/.kernel-talk/store
DATA     ?= data
CKPTS    ?= training/checkpoints

# ── Training hyperparameters ───────────────────────────────────────────────────
EPOCHS       ?= 3
BATCH_SIZE   ?= 16
MAX_SYNTH    ?= 5000
N_HARD_NEG   ?= 16

# ─────────────────────────────────────────────────────────────────────────────
.PHONY: all setup rust index synth bm25 train pipeline test eval clean clean-all help

all: help

help:
	@echo "Kernel-Talk — available targets:"
	@echo ""
	@echo "  make setup         Create venv + install Python deps"
	@echo "  make rust          Build Rust DWARF extension (optional, 20× speedup)"
	@echo "  make index         Index kernel source into vector store"
	@echo "  make synth         Generate synthetic training triplets"
	@echo "  make bm25          Build BM25 index + enrich with hard negatives"
	@echo "  make train         Fine-tune CodeBERT bi-encoder"
	@echo "  make pipeline      Full pipeline (index → synth → bm25 → train)"
	@echo "  make test          Run all tests"
	@echo "  make eval          Run retrieval evaluation"
	@echo "  make clean         Remove generated data (keep store + checkpoints)"
	@echo "  make clean-all     Remove everything generated (venv, store, data)"
	@echo ""
	@echo "Configurable vars (override with make VAR=value):"
	@echo "  KERNEL=$(KERNEL)"
	@echo "  SUBSYS=$(SUBSYS)"
	@echo "  STORAGE=$(STORAGE)"
	@echo "  DATA=$(DATA)"
	@echo "  EPOCHS=$(EPOCHS)"
	@echo "  BATCH_SIZE=$(BATCH_SIZE)"

# ─── Setup ────────────────────────────────────────────────────────────────────
setup: .venv/bin/python

.venv/bin/python:
	python3 -m venv .venv
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -r requirements.txt --quiet
	@echo "[setup] Python environment ready."

# ─── Rust DWARF extension ─────────────────────────────────────────────────────
rust: setup
	@command -v cargo >/dev/null 2>&1 || { \
	  echo "[rust] cargo not found. Install Rust from https://rustup.rs"; exit 1; }
	@command -v maturin >/dev/null 2>&1 || $(PIP) install maturin --quiet
	cd rust_ext/dwarf_reader && \
	  PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
	@echo "[rust] kernel_talk_dwarf_rs installed — DWARF parsing ~20× faster."

# ─── Mirror index ─────────────────────────────────────────────────────────────
index: setup
	@mkdir -p $(DATA)
	$(PYTHON) -m cli.ktalk index \
	  --kernel $(KERNEL) \
	  --subsystem $(SUBSYS) \
	  --storage $(STORAGE)

# ─── Synthetic training data ──────────────────────────────────────────────────
$(DATA)/triplets.jsonl: setup
	@mkdir -p $(DATA)
	$(PYTHON) -m training.synth \
	  --storage $(STORAGE) \
	  --output $(DATA)/triplets.jsonl \
	  --max-per-strategy $(MAX_SYNTH)

synth: $(DATA)/triplets.jsonl

# ─── BM25 hard-negative enrichment ────────────────────────────────────────────
$(DATA)/bm25.pkl: $(DATA)/triplets.jsonl
	$(PYTHON) -m training.bm25 build \
	  --storage $(STORAGE) \
	  --output $(DATA)/bm25.pkl

$(DATA)/enriched.jsonl: $(DATA)/bm25.pkl $(DATA)/triplets.jsonl
	$(PYTHON) -m training.bm25 enrich \
	  --triplets $(DATA)/triplets.jsonl \
	  --bm25     $(DATA)/bm25.pkl \
	  --output   $(DATA)/enriched.jsonl \
	  --n-hard   $(N_HARD_NEG)

bm25: $(DATA)/enriched.jsonl

# ─── Training ─────────────────────────────────────────────────────────────────
train: $(DATA)/enriched.jsonl
	@mkdir -p $(CKPTS)
	$(PYTHON) -m training.train_biencoder \
	  --triplets   $(DATA)/enriched.jsonl \
	  --storage    $(STORAGE) \
	  --output     $(CKPTS) \
	  --epochs     $(EPOCHS) \
	  --batch-size $(BATCH_SIZE)

# ─── Full pipeline ─────────────────────────────────────────────────────────────
pipeline: index synth bm25 train
	@echo "[pipeline] Complete! Checkpoints at $(CKPTS)"

# ─── Tests ────────────────────────────────────────────────────────────────────
test: setup
	$(PYTHON) -m pytest tests/ -v

# ─── Eval ─────────────────────────────────────────────────────────────────────
eval: setup
	$(PYTHON) -m eval.run \
	  --gold eval/retrieval_gold.jsonl \
	  --storage $(STORAGE) \
	  --output eval/results.json 2>/dev/null || \
	$(PYTHON) -c "\
import json, sys; \
sys.path.insert(0,''); \
from core.mirror.store import KernelStore; \
from eval.retrieval import run_eval; \
run_eval('eval/retrieval_gold.jsonl', '$(STORAGE)')"

# ─── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -rf $(DATA)/triplets.jsonl $(DATA)/bm25.pkl $(DATA)/enriched.jsonl
	rm -rf eval/results.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clean-all: clean
	rm -rf .venv
	rm -rf $(STORAGE)
	rm -rf $(CKPTS)
	rm -rf rust_ext/dwarf_reader/target
