"""
training/dataset.py
────────────────────
PyTorch Dataset for kernel code retrieval training.

Produces (query, positive_code, negative_code) batches suitable for
InfoNCE / margin ranking loss training.

Negative sampling strategy
──────────────────────────
Each training example has:
  - 1 query (commit message subject)
  - 1 positive (function that was changed)
  - N in-batch negatives (other examples' positives — free)
  - k hard negatives (BM25-selected, stored in the JSONL)
  - k easy negatives (random, stored in the JSONL)

The DataLoader mixes hard and easy at ratio `hard_ratio` (default 0.3).
This follows the recipe from DPR (Karpukhin et al. 2020) which found
that ~30% hard negatives maximizes retrieval performance without causing
the gradient variance that comes from 100% hard negatives.

Encoding
─────────
Input text for the bi-encoder:
  Query encoder  : "[CLS] {commit_message} [SEP]"
  Code encoder   : "[CLS] {function_name}\n{code} [SEP]"

We prepend the function name to the code body because:
  1. The function signature carries the most compressed semantic info
     (e.g., `static int tg_throttle_up(...)` tells you it's about
     throttle + up + task group — all key discriminating terms)
  2. BERT attention can use it as an anchor for the rest of the body
  3. It costs only a few tokens and empirically helps Recall@k

Max token length: 512 (BERT limit). Long functions are truncated from
the end — the function signature at the start is preserved. This is
correct because the signature is what changes most meaningfully.

Usage:
    from training.dataset import TripletDataset, collate_fn
    ds = TripletDataset("data/enriched.jsonl", tokenizer, store)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn, shuffle=True)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizerBase
    from core.mirror.store import KernelStore

# Lazy torch import: use torch.utils.data.Dataset as base when available,
# otherwise fall back to object so curriculum/difficulty logic is importable
# in environments without torch (e.g., unit test sandboxes).
try:
    from torch.utils.data import Dataset as _DatasetBase
except ImportError:
    class _DatasetBase:  # type: ignore[no-redef]
        """Minimal no-op stub so TripletDataset is importable without torch."""


# ─── Token limits ─────────────────────────────────────────────────────────────

QUERY_MAX_LEN = 128    # Commit messages are short; 128 tokens is always enough
CODE_MAX_LEN  = 128    # 128 gives 4× attention speedup vs 256 on CPU; still covers ~80% of kernel funcs


# ─── Data item ────────────────────────────────────────────────────────────────

@dataclass
class TripletItem:
    """
    A single training example before tokenization.
    Holds raw text strings — the Dataset handles tokenization.
    """
    query: str
    positive_text: str          # function_name + "\n" + code
    hard_negative_texts: list[str]   # same format
    easy_negative_texts: list[str]


# ─── Text preparation ─────────────────────────────────────────────────────────

def _code_text(meta: dict) -> str:
    """
    Prepare code text for the encoder: "symbol_name\ncode".
    Falls back gracefully if fields are missing.
    """
    name = meta.get("symbol_name", "")
    code = meta.get("code", "")
    if name and code:
        return f"{name}\n{code}"
    return code or name or "(empty)"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TripletDataset(_DatasetBase):
    """
    Loads enriched triplets JSONL and fetches code text from the KernelStore.

    The JSONL format is:
    {
        "query": str,
        "positives": [node_id, ...],
        "hard_negatives": [node_id, ...],
        "easy_negatives": [node_id, ...],
        ...
    }

    Args:
        jsonl_path:     Path to enriched triplets JSONL (output of bm25.py enrich)
        tokenizer:      HuggingFace tokenizer (e.g., AutoTokenizer for CodeBERT)
        store:          KernelStore with indexed functions (for code text lookup)
        hard_ratio:     Maximum fraction of negatives that are hard (BM25) vs easy.
                        This is the *target* ratio at full training progress.
                        Early epochs use a lower effective ratio for each example
                        based on that example's BM25 difficulty gap.
        n_negatives:    Total negatives per example (hard + easy)
        max_items:      If set, truncate dataset (for fast dev runs)

    Curriculum scheduling
    ─────────────────────
    If the enriched JSONL contains a 'difficulty' field (written by bm25.py
    enrich), the dataset supports adaptive curriculum training. Call:

        dataset.set_curriculum_epoch(epoch, total_epochs)

    at the start of each training epoch. This updates an internal progress
    scalar `_curriculum_progress` ∈ [0, 1].

    During __getitem__, the per-example effective hard ratio is:

        effective_ratio = hard_ratio × (progress + (1−progress) × (1−difficulty))

    Unpacking this formula:
      At progress=0 (epoch 0):
        - Easy example (difficulty≈0): effective_ratio = hard_ratio × 1  ← full ratio
        - Hard example (difficulty≈1): effective_ratio = hard_ratio × 0  ← no hard negs
      At progress=1 (final epoch):
        - Any difficulty: effective_ratio = hard_ratio × 1               ← always full

    This guarantees peak performance is uncapped: by the final epoch, every
    example — including the hardest ones — sees the full configured hard_ratio.
    The curriculum only determines *when* hard examples enter full training,
    not whether they ever do.

    If 'difficulty' is absent (old-format JSONL without enrichment), all examples
    use the configured hard_ratio unchanged — backward compatible.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: "PreTrainedTokenizerBase",
        store: "KernelStore",
        hard_ratio: float = 0.3,
        n_negatives: int = 7,
        max_items: int | None = None,
    ):
        self.tokenizer   = tokenizer
        self.store       = store
        self.hard_ratio  = hard_ratio
        self.n_negatives = n_negatives

        # Curriculum progress: 0.0 at epoch 0, 1.0 at final epoch.
        # Updated externally by set_curriculum_epoch().
        self._curriculum_progress: float = 1.0   # default: no curriculum

        # Load all triplets into memory
        # ~50K triplets × ~200 bytes each ≈ 10 MB — fine
        self._records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self._records.append(json.loads(line))
                    if max_items and len(self._records) >= max_items:
                        break

        # Check whether difficulty metadata is present
        self._has_difficulty = (
            bool(self._records) and "difficulty" in self._records[0]
        )

        # Pre-warm the code lookup cache for all node IDs in the dataset
        # (avoids repeated ChromaDB round-trips during training)
        self._code_cache: dict[str, str] = {}
        self._warm_cache()

    def _warm_cache(self) -> None:
        """Pre-fetch code text for all node IDs in the dataset."""
        all_ids: set[str] = set()
        for rec in self._records:
            all_ids.update(rec.get("positives", []))
            all_ids.update(rec.get("hard_negatives", []))
            all_ids.update(rec.get("easy_negatives", []))

        # Batch lookup from ChromaDB
        id_list = list(all_ids)
        batch_size = 500
        for start in range(0, len(id_list), batch_size):
            batch = id_list[start:start + batch_size]
            try:
                result = self.store._get_collection().get(
                    ids=batch, include=["metadatas", "documents"]
                )
                for node_id, meta, doc in zip(
                    result["ids"], result["metadatas"], result["documents"]
                ):
                    code = doc if doc else meta.get("code", "")
                    self._code_cache[node_id] = code or _code_text(meta)
            except Exception:
                pass  # Cache miss is OK — we fall back to empty string

    def _get_code(self, node_id: str) -> str:
        return self._code_cache.get(node_id, "")

    def set_curriculum_epoch(self, epoch: int, total_epochs: int) -> None:
        """
        Advance the curriculum to a given epoch.

        Call this at the start of each training epoch:
            dataset.set_curriculum_epoch(epoch, total_epochs)

        This updates _curriculum_progress ∈ [0, 1]:
          epoch=0, total=3  → progress = 0.0  (start: hard examples get little hard ratio)
          epoch=1, total=3  → progress = 0.5
          epoch=2, total=3  → progress = 1.0  (end: all examples get full hard_ratio)

        If total_epochs=1, progress is immediately 1.0 (no curriculum — same as
        setting _curriculum_progress=1.0 at init, which is the default).

        Has no effect if the dataset has no 'difficulty' field.
        """
        if total_epochs <= 1:
            self._curriculum_progress = 1.0
        else:
            self._curriculum_progress = epoch / (total_epochs - 1)

    def _effective_hard_ratio(self, difficulty: float) -> float:
        """
        Compute the per-example effective hard_ratio given its difficulty
        and the current curriculum progress.

        Formula:
            effective = hard_ratio × (progress + (1−progress) × (1−difficulty))

        Derivation:
          Let p = _curriculum_progress, d = difficulty ∈ [0,1].
          We want a function that:
            (a) At p=1 always returns hard_ratio regardless of d  → peak uncapped
            (b) At p=0 returns hard_ratio for d=0 and 0 for d=1  → curriculum active
            (c) Is linear in both p and d                         → no hyperparameters

          Solving: f(p, d) = hard_ratio × (p + (1-p)×(1-d))
            f(1, any) = hard_ratio × 1           ✓ (a)
            f(0, 0)   = hard_ratio × 1           ✓ (b) easiest → full ratio immediately
            f(0, 1)   = hard_ratio × 0           ✓ (b) hardest → no hard negs at start
            f(0, 0.5) = hard_ratio × 0.5         intermediate difficulty → half ratio

        This is the minimal-hyperparameter curriculum that satisfies all three
        constraints. No temperature, no schedule shape parameter.
        """
        p = self._curriculum_progress
        effective = self.hard_ratio * (p + (1.0 - p) * (1.0 - difficulty))
        return max(0.0, min(1.0, effective))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> TripletItem:
        rec = self._records[idx]

        # Pick one positive at random (a commit may touch multiple functions)
        pos_ids = rec.get("positives", [])
        positive_id = random.choice(pos_ids) if pos_ids else ""
        positive_text = self._get_code(positive_id)

        # Per-example effective hard_ratio: full hard_ratio for easy examples
        # early in training; grows toward hard_ratio for hard examples as
        # curriculum_progress increases. Falls back to self.hard_ratio if
        # no difficulty metadata present (backward compatible).
        if self._has_difficulty:
            difficulty = rec.get("difficulty", 0.0)
            ratio = self._effective_hard_ratio(difficulty)
        else:
            ratio = self.hard_ratio

        # Sample negatives: mix of hard and easy at the effective ratio
        n_hard = int(self.n_negatives * ratio)
        n_easy = self.n_negatives - n_hard

        # Guarantee at least one negative total
        if n_hard == 0 and n_easy == 0:
            n_easy = 1

        hard_pool = rec.get("hard_negatives", [])
        easy_pool = rec.get("easy_negatives", [])

        hard_sample = random.sample(hard_pool, min(n_hard, len(hard_pool)))
        easy_sample = random.sample(easy_pool, min(n_easy, len(easy_pool)))

        hard_texts = [self._get_code(nid) for nid in hard_sample]
        easy_texts = [self._get_code(nid) for nid in easy_sample]

        return TripletItem(
            query=rec["query"],
            positive_text=positive_text,
            hard_negative_texts=hard_texts,
            easy_negative_texts=easy_texts,
        )


# ─── Collate ──────────────────────────────────────────────────────────────────

@dataclass
class TrainingBatch:
    """
    The output of collate_fn — a batch ready for the model.

    query_input_ids:      (B, Q_len)  int64
    query_attention_mask: (B, Q_len)  int64
    code_input_ids:       (B*(1+N), C_len)  int64   — positive + negatives stacked
    code_attention_mask:  (B*(1+N), C_len)  int64
    labels:               (B,)  int64 — index of the positive in [0..N] for each query
                          Always 0 because we put the positive first.
    """
    query_input_ids:       "torch.Tensor"
    query_attention_mask:  "torch.Tensor"
    code_input_ids:        "torch.Tensor"
    code_attention_mask:   "torch.Tensor"
    labels:                "torch.Tensor"


def make_collate_fn(tokenizer: "PreTrainedTokenizerBase"):
    """
    Returns a collate_fn that tokenizes and stacks a list of TripletItems.

    The code tensor is (B*(1+N), C_len):
        Row 0: positive for query 0
        Row 1..N: negatives for query 0
        Row N+1: positive for query 1
        ...

    Labels = [0, 0, ..., 0] because the positive is always first.
    Cross-entropy loss with these labels = InfoNCE.

    In-batch negatives are handled by the training loop (not the collator)
    by treating other examples' positives as additional negatives — this
    is free and is what makes InfoNCE efficient at large batch sizes.
    """

    def collate_fn(items: list[TripletItem]) -> TrainingBatch:
        queries = [item.query for item in items]

        # Tokenize queries
        q_enc = tokenizer(
            queries,
            max_length=QUERY_MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # For each example, stack [positive] + negatives into one list
        all_code_texts = []
        n_per_item: list[int] = []
        for item in items:
            codes = [item.positive_text] + item.hard_negative_texts + item.easy_negative_texts
            all_code_texts.extend(codes)
            n_per_item.append(len(codes))

        # Tokenize all code texts at once (efficient batching)
        c_enc = tokenizer(
            all_code_texts,
            max_length=CODE_MAX_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        import torch  # lazy: only needed when actually collating tensors

        # Labels: positive is always at position 0 within each item's slice
        labels = torch.zeros(len(items), dtype=torch.long)

        return TrainingBatch(
            query_input_ids=q_enc["input_ids"],
            query_attention_mask=q_enc["attention_mask"],
            code_input_ids=c_enc["input_ids"],
            code_attention_mask=c_enc["attention_mask"],
            labels=labels,
        )

    return collate_fn
