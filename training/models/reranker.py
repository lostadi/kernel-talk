"""
training/models/reranker.py — Cross-encoder reranker model.

Spec §6.3:
  Base: microsoft/codebert-base (encoder-only, 110M params).
  Input: [CLS] query [SEP] candidate_text [SEP].
  Head: single linear layer on [CLS] → scalar score.
  Loss: margin-based ranking loss on (query, positive, negative) triplets.

The reranker is used *on top of* the bi-encoder:
  1. Bi-encoder retrieves top-N candidates (fast ANN search, O(N) comparisons)
  2. Reranker rescores top-N with full cross-attention (O(N) transformer passes)
  3. Return reranker's top-k

This two-stage pipeline lets us use a more expensive but accurate model (the
cross-encoder) while keeping retrieval latency acceptable.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class KernelReranker(nn.Module):
    """
    Cross-encoder reranker for kernel code retrieval.

    Takes (query, candidate) text pairs and outputs a scalar relevance score.
    Higher score = more relevant.
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for codebert-base
        self.dropout = nn.Dropout(dropout)
        self.score_head = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.score_head.weight, std=0.02)
        nn.init.zeros_(self.score_head.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:       (batch, seq_len)
            attention_mask:  (batch, seq_len)
            token_type_ids:  (batch, seq_len) — optional for RoBERTa-based models

        Returns:
            scores: (batch,) — unbounded scalar relevance scores
        """
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        cls = self.dropout(cls)
        scores = self.score_head(cls).squeeze(-1)  # (batch,)
        return scores

    def save(self, output_dir: str | Path) -> None:
        """Save model weights and encoder config."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "reranker.pt")
        self.encoder.config.save_pretrained(str(output_dir))
        (output_dir / "model_name.txt").write_text(
            self.encoder.config.name_or_path or "microsoft/codebert-base"
        )

    @classmethod
    def load(cls, checkpoint_dir: str | Path) -> "KernelReranker":
        """Load a saved reranker from a checkpoint directory."""
        checkpoint_dir = Path(checkpoint_dir)
        model_name_file = checkpoint_dir / "model_name.txt"
        model_name = (
            model_name_file.read_text().strip()
            if model_name_file.exists()
            else "microsoft/codebert-base"
        )
        model = cls(model_name=model_name)
        state = torch.load(checkpoint_dir / "reranker.pt", map_location="cpu")
        model.load_state_dict(state)
        return model


def margin_ranking_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Margin-based ranking loss (spec §6.3).

    L = mean(max(0, margin - score_pos + score_neg))

    For each triplet (query, positive, negative):
      - If score_pos > score_neg + margin: loss = 0 (already correct with margin)
      - Otherwise: push pos higher and neg lower

    Args:
        pos_scores: (batch,) scores for (query, positive) pairs
        neg_scores: (batch,) scores for (query, negative) pairs
        margin:     minimum score gap we require between pos and neg

    Returns:
        scalar loss
    """
    loss = torch.clamp(margin - pos_scores + neg_scores, min=0.0)
    return loss.mean()


def tokenize_pair(
    tokenizer: AutoTokenizer,
    queries: list[str],
    candidates: list[str],
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Tokenize (query, candidate) pairs into [CLS] q [SEP] c [SEP] format.

    Returns a dict with input_ids, attention_mask (and token_type_ids
    for BERT-style models that use them).
    """
    return tokenizer(
        queries,
        candidates,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
