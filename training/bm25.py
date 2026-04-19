"""
training/bm25.py
─────────────────
BM25 hard negative mining for the training triplets.

Background: why hard negatives matter
──────────────────────────────────────
If you train a retrieval model only with random negatives, it learns to
distinguish "scheduler code" from "filesystem code" — an easy problem.
What it fails to learn is: given "sched: fix CFS bandwidth throttle",
why is `tg_throttle_up()` the answer and not `throttle_cfs_rq()` or
`update_curr()`? Those are all in kernel/sched/fair.c and all involve
throttling. Only hard negatives force the model to learn fine-grained
discrimination.

BM25 is the right tool for finding hard negatives because:
  1. It's fast (inverted index, O(|query_terms|) at query time)
  2. It naturally surfaces lexically similar functions — exactly the
     hard cases a semantic model must distinguish
  3. It's interpretable: you can see WHY a negative is hard

Okapi BM25 score for a query q and document d:
    score(q, d) = Σ_t IDF(t) · (tf(t,d) · (k1+1)) / (tf(t,d) + k1 · (1-b+b·|d|/avgdl))

where:
  t   = query term
  IDF = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)   [Robertson IDF]
  tf  = term frequency in document
  |d| = document length in tokens
  avgdl = average document length
  k1 = 1.5  (term frequency saturation)
  b  = 0.75 (length normalization)

Implementation note: we implement BM25 from scratch rather than using
`rank_bm25` to make the tokenization kernel-C-aware. Specifically:
  - Split on C operator boundaries (not just whitespace)
  - Keep snake_case identifiers intact (don't split at _)
  - Downweight very common C tokens (int, void, return, struct)

Usage:
    # Build index from a KernelStore
    idx = BM25Index.from_store(store)
    idx.save("data/bm25.pkl")

    # Load and query
    idx = BM25Index.load("data/bm25.pkl")
    hard_negs = idx.hard_negatives(
        query="sched: fix CFS bandwidth throttle on RT task wakeup",
        positive_ids={"kernel/sched/fair.c::tg_throttle_up"},
        k=16,
    )
"""

from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.mirror.store import KernelStore


# ─── Tokenizer ────────────────────────────────────────────────────────────────

# C tokens that carry almost no semantic information at BM25 retrieval time.
# Keeping them would make every C function look similar.
_C_STOPWORDS = frozenset({
    "int", "void", "char", "long", "unsigned", "signed", "static",
    "inline", "return", "if", "else", "for", "while", "do", "switch",
    "case", "break", "continue", "goto", "typedef", "struct", "union",
    "enum", "const", "volatile", "extern", "register", "sizeof",
    "NULL", "true", "false", "0", "1", "2",
    "the", "a", "an", "is", "of", "to", "in", "for", "and", "or",
})

# Split on: whitespace, C operators, punctuation — but keep snake_case intact
_TOKENIZE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """
    Tokenize C source code or a natural language commit message.

    Strategy:
      - Extract word-like tokens (C identifiers, numbers) with a regex
      - Lowercase everything
      - Remove stopwords
      - Keep snake_case identifiers whole (schedule_timeout stays as-is)
        because the full name is the semantic unit in kernel C

    We do NOT split snake_case because `tg_throttle_up` and `throttle_cfs_rq`
    should NOT share tokens — they're different functions. Splitting would
    collapse them, defeating the purpose of hard negative mining.
    """
    tokens = _TOKENIZE_RE.findall(text)
    return [t.lower() for t in tokens if t.lower() not in _C_STOPWORDS and len(t) > 1]


# ─── BM25 Index ───────────────────────────────────────────────────────────────

@dataclass
class _DocRecord:
    """Internal record for a single indexed document."""
    node_id: str           # CodeNode.id
    term_freqs: dict[str, int]
    length: int            # token count


class BM25Index:
    """
    BM25 inverted index over kernel function code.

    Index documents = CodeNode.code + CodeNode.symbol_name
    (concatenating the function name with its body gives better recall
    for queries that mention the function name explicitly)

    Query documents = commit message subject line (or any natural language query)
    """

    K1: float = 1.5    # term frequency saturation
    B: float  = 0.75   # length normalization strength

    def __init__(self) -> None:
        self._docs: list[_DocRecord] = []
        self._id_to_idx: dict[str, int] = {}       # node_id → list index
        self._df: Counter[str] = Counter()          # document frequency per term
        self._avg_dl: float = 0.0
        self._N: int = 0                            # total documents

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_store(cls, store: "KernelStore", verbose: bool = True) -> "BM25Index":
        """
        Build a BM25 index from all CodeNodes in the KernelStore.

        This reads directly from ChromaDB, so the Mirror must be indexed first
        (ktalk index ...).  We index function name + code body together so that
        queries mentioning "schedule" retrieve schedule() even if the word only
        appears in the function signature.
        """
        idx = cls()

        # Pull all documents from ChromaDB (may be large; kernel has ~500K nodes)
        # We paginate to avoid OOM
        batch_size = 1000
        offset = 0
        docs_added = 0

        while True:
            result = store._get_collection().get(
                limit=batch_size,
                offset=offset,
                include=["metadatas", "documents"],
            )
            if not result["ids"]:
                break

            for node_id, meta, doc in zip(
                result["ids"], result["metadatas"], result["documents"]
            ):
                code = doc if doc else meta.get("code", "")
                text = (
                    meta.get("symbol_name", "") + " " + code
                )
                idx._add_document(node_id, text)
                docs_added += 1

            offset += batch_size
            if verbose and docs_added % 50_000 == 0:
                print(f"[bm25] Indexed {docs_added} documents ...")

            if len(result["ids"]) < batch_size:
                break

        idx._finalize()
        if verbose:
            print(f"[bm25] Index complete: {idx._N} documents, "
                  f"{len(idx._df)} unique terms, "
                  f"avg length={idx._avg_dl:.0f} tokens")
        return idx

    def _add_document(self, node_id: str, text: str) -> None:
        tokens = tokenize(text)
        tf = Counter(tokens)
        rec = _DocRecord(node_id=node_id, term_freqs=dict(tf), length=len(tokens))
        idx = len(self._docs)
        self._docs.append(rec)
        self._id_to_idx[node_id] = idx
        for term in tf:
            self._df[term] += 1

    def _finalize(self) -> None:
        self._N = len(self._docs)
        total_len = sum(d.length for d in self._docs)
        self._avg_dl = total_len / self._N if self._N > 0 else 1.0

    # ── Query ─────────────────────────────────────────────────────────────────

    def score(self, query: str, node_id: str) -> float:
        """BM25 score for a single (query, document) pair."""
        if node_id not in self._id_to_idx:
            return 0.0

        doc = self._docs[self._id_to_idx[node_id]]
        q_tokens = tokenize(query)
        if not q_tokens:
            return 0.0

        score = 0.0
        dl = doc.length
        avgdl = self._avg_dl

        for term in q_tokens:
            tf = doc.term_freqs.get(term, 0)
            if tf == 0:
                continue

            df = self._df.get(term, 0)
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

            tf_norm = (tf * (self.K1 + 1)) / (
                tf + self.K1 * (1 - self.B + self.B * dl / avgdl)
            )
            score += idf * tf_norm

        return score

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        """
        Return top-k (node_id, score) pairs for `query`.

        Optimized: only score documents that contain at least one query term.
        This is the core BM25 retrieval loop — standard inverted index lookup.
        """
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return []

        # Candidate set: documents containing any query term
        candidates: set[int] = set()
        for term in q_tokens:
            if term in self._df:
                for i, doc in enumerate(self._docs):
                    if term in doc.term_freqs:
                        candidates.add(i)

        if not candidates:
            return []

        # Score candidates (only need full query here)
        scored = []
        q_str = " ".join(q_tokens)  # re-join for score()
        for i in candidates:
            doc = self._docs[i]
            s = self._score_doc(doc, list(q_tokens))
            scored.append((doc.node_id, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def _score_doc(self, doc: _DocRecord, q_tokens: list[str]) -> float:
        """Score a single doc against a list of query tokens."""
        dl = doc.length
        avgdl = self._avg_dl
        score = 0.0

        for term in q_tokens:
            tf = doc.term_freqs.get(term, 0)
            if tf == 0:
                continue
            df = self._df.get(term, 0)
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            tf_norm = (tf * (self.K1 + 1)) / (
                tf + self.K1 * (1 - self.B + self.B * dl / avgdl)
            )
            score += idf * tf_norm

        return score

    def difficulty_gap(
        self,
        query: str,
        positive_ids: set[str],
        hard_neg_ids: list[str],
    ) -> float:
        """
        Compute the BM25 difficulty gap for a (query, positive, hard_neg) triplet.

        gap = max_score(positives) - score(hardest_negative)

        Interpretation:
          Large positive gap  → BM25 already separates them well; the hard
                                negative isn't actually that confusable at the
                                lexical level. Easier training example.
          Near-zero gap       → BM25 scores positive and hard negative almost
                                identically. The model must learn something truly
                                semantic to distinguish them. Harder training example.
          Negative gap        → The hard negative scores HIGHER than the positive
                                on BM25. The query's surface tokens match the
                                negative better than the positive (e.g., a commit
                                message mentions a function name that isn't the
                                one changed). These are the hardest cases and
                                should be introduced latest in curriculum.

        Returns 0.0 if no positive or no hard negative is indexed.
        """
        pos_scores = [self.score(query, pid) for pid in positive_ids
                      if pid in self._id_to_idx]
        if not pos_scores or not hard_neg_ids:
            return 0.0

        pos_max = max(pos_scores)
        # Score of the single hardest (highest-ranked) hard negative
        hard_neg_score = self.score(query, hard_neg_ids[0]) if hard_neg_ids else 0.0
        return pos_max - hard_neg_score

    def hard_negatives(
        self,
        query: str,
        positive_ids: set[str],
        k: int = 16,
        subsystem_filter: str | None = None,
    ) -> list[str]:
        """
        Return the k hardest negatives for a given query and positive set.

        "Hard" = highest BM25 score among non-positive documents.

        Args:
            query:            The natural language query / commit message
            positive_ids:     Set of CodeNode IDs that are gold positives (exclude these)
            k:                How many hard negatives to return
            subsystem_filter: If given, only consider documents whose node_id
                              starts with this path prefix. Use to stay within
                              the same subsystem — these are the hardest hard negatives.

        Design note: returning k=16 is intentional. In practice, during
        training we'll randomly sample 1–4 of these per batch item. Having
        16 candidates gives the DataLoader flexibility to:
          a) Use in-batch negatives when batch size is large
          b) Mix easy (random) and hard (BM25) negatives in the right ratio
          c) Re-sample different hard negatives across epochs (to prevent
             the model from overfitting to specific false positives)
        """
        results = self.search(query, k=k + len(positive_ids) + 50)

        hard_negs = []
        for node_id, score in results:
            if node_id in positive_ids:
                continue
            if subsystem_filter and not node_id.startswith(subsystem_filter):
                continue
            hard_negs.append(node_id)
            if len(hard_negs) >= k:
                break

        return hard_negs

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Pickle the index to disk. Typical size: ~200 MB for full kernel."""
        with open(path, "wb") as f:
            pickle.dump({
                "docs":    self._docs,
                "id_to_idx": self._id_to_idx,
                "df":      self._df,
                "avg_dl":  self._avg_dl,
                "N":       self._N,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        """Load a previously saved BM25 index."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._docs     = data["docs"]
        idx._id_to_idx = data["id_to_idx"]
        idx._df       = data["df"]
        idx._avg_dl   = data["avg_dl"]
        idx._N        = data["N"]
        return idx


# ─── Enrich triplets with hard negatives ──────────────────────────────────────

def enrich_triplets(
    triplets_path: str | Path,
    bm25_index: BM25Index,
    output_path: str | Path,
    n_hard: int = 16,
    n_easy: int = 8,
    verbose: bool = True,
) -> None:
    """
    Read a triplets JSONL file, add hard_negatives, easy_negatives, and
    adaptive difficulty metadata to each record, then write an enriched JSONL.

    Enriched format per record:
    {
      "query":             str,
      "positives":         [node_id, ...],
      "hard_negatives":    [16 BM25-selected node_ids],
      "easy_negatives":    [8 randomly-sampled node_ids],
      "hard_negative_gap": float,   ← raw BM25 gap: score(pos) - score(hardest_neg)
      "difficulty":        float,   ← [0, 1] normalized; 0=easiest, 1=hardest
      ...
    }

    Difficulty normalization
    ────────────────────────
    We run in two passes:
      Pass 1: compute all raw gaps, accumulate into a list
      Pass 2: percentile-rank each gap, invert to get difficulty ∈ [0, 1]

    Percentile ranking is robust to outliers (a few commits where the
    positive has a very high BM25 score won't distort the scale for everyone
    else). The inversion is: difficulty = 1 - (gap_rank / N) so that:
      - Largest gap (easiest) → difficulty ≈ 0
      - Smallest gap (hardest, including negative gaps) → difficulty ≈ 1

    The 'difficulty' field is what the DataLoader uses for curriculum
    scheduling in dataset.py. The raw 'hard_negative_gap' is stored for
    debugging and analysis.
    """
    import random

    all_node_ids = list(bm25_index._id_to_idx.keys())

    # ── Pass 1: compute negatives and raw gaps ────────────────────────────────
    records: list[dict] = []

    for i, line in enumerate(open(triplets_path)):
        line = line.strip()
        if not line:
            continue

        # Support both synth.py output (plain dict) and mine.py Triplet format
        data = json.loads(line)
        # Normalize to common schema
        query = data.get("query", "")
        positives = data.get("positives", [])
        if not positives:
            # Single-positive format from some strategies
            pos = data.get("positive")
            if pos:
                positives = [pos]

        if not query or not positives:
            continue

        positive_set = set(positives)

        # Determine subsystem prefix from first positive's path
        subsystem_prefix = None
        if positives:
            parts = positives[0].split("::")
            if parts:
                path_parts = parts[0].split("/")
                if len(path_parts) >= 2:
                    subsystem_prefix = "/".join(path_parts[:2])

        # Hard negatives: BM25, optionally within same subsystem
        hard_negs = bm25_index.hard_negatives(
            query=query,
            positive_ids=positive_set,
            k=n_hard,
            subsystem_filter=subsystem_prefix,
        )
        # Fallback: widen search if subsystem is too narrow
        if len(hard_negs) < n_hard // 2:
            hard_negs = bm25_index.hard_negatives(
                query=query,
                positive_ids=positive_set,
                k=n_hard,
            )

        # Easy negatives: random sample from outside this subsystem
        easy_pool = [
            nid for nid in all_node_ids
            if nid not in positive_set
            and (subsystem_prefix is None or not nid.startswith(subsystem_prefix))
        ]
        easy_negs = random.sample(easy_pool, min(n_easy, len(easy_pool)))

        # Raw difficulty gap: positive BM25 score minus hardest-negative BM25 score.
        gap = bm25_index.difficulty_gap(
            query=query,
            positive_ids=positive_set,
            hard_neg_ids=hard_negs,
        )

        records.append({
            **data,
            "positives":         positives,
            "hard_negatives":    hard_negs,
            "easy_negatives":    easy_negs,
            "hard_negative_gap": gap,
        })

        if verbose and i % 1000 == 0:
            print(f"[enrich] Pass 1: {i} triplets processed ...", file=sys.stderr)

    # ── Pass 2: percentile-normalize gaps → difficulty ∈ [0, 1] ─────────────
    # Sort gaps ascending (smallest/hardest first) to get rank.
    # difficulty = 1 - (gap_rank / N) so hardest → difficulty ≈ 1.
    N = len(records)
    if N == 0:
        if verbose:
            print("[enrich] No records to enrich.", file=sys.stderr)
        return

    gaps = [r["hard_negative_gap"] for r in records]
    sorted_gaps = sorted(gaps)                       # ascending: hardest first among equals

    # Build a lookup: gap value → its rank among all gaps (0-indexed, ascending)
    # Ties get the same rank (rank of first occurrence in sorted order).
    # We use a dict from gap → rank, built from the sorted list.
    gap_to_rank: dict[float, int] = {}
    for rank, g in enumerate(sorted_gaps):
        if g not in gap_to_rank:
            gap_to_rank[g] = rank    # first (lowest) rank for ties

    if verbose:
        import statistics
        print(f"[enrich] Gap stats: min={min(gaps):.3f}, "
              f"median={statistics.median(gaps):.3f}, "
              f"max={max(gaps):.3f}", file=sys.stderr)

    # ── Write enriched JSONL ──────────────────────────────────────────────────
    with open(output_path, "w") as fout:
        for record in records:
            rank = gap_to_rank[record["hard_negative_gap"]]
            # rank=0 → hardest → difficulty=1; rank=N-1 → easiest → difficulty≈0
            difficulty = 1.0 - (rank / max(N - 1, 1))
            record["difficulty"] = round(difficulty, 6)
            print(json.dumps(record), file=fout)

    if verbose:
        print(f"[enrich] Done: {N} records written to {output_path}", file=sys.stderr)


# ─── CLI ──────────────────────────────────────────────────────────────────────

import json
import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build BM25 index or enrich triplets.")
    subparsers = parser.add_subparsers(dest="cmd")

    # Build index from a KernelStore
    build_p = subparsers.add_parser("build", help="Build BM25 index from KernelStore")
    build_p.add_argument("--storage", required=True, help="KernelStore storage dir")
    build_p.add_argument("--output",  required=True, help="Output .pkl file")

    # Enrich triplets JSONL with hard negatives
    enrich_p = subparsers.add_parser("enrich", help="Add hard negatives to triplets JSONL")
    enrich_p.add_argument("--triplets", required=True, help="Input triplets JSONL")
    enrich_p.add_argument("--bm25",     required=True, help="BM25 index .pkl file")
    enrich_p.add_argument("--output",   required=True, help="Output enriched JSONL")
    enrich_p.add_argument("--n-hard",   type=int, default=16)
    enrich_p.add_argument("--n-easy",   type=int, default=8)

    args = parser.parse_args()

    if args.cmd == "build":
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from core.mirror.store import KernelStore
        store = KernelStore(storage_dir=args.storage)
        idx = BM25Index.from_store(store)
        idx.save(args.output)
        print(f"[bm25] Saved index to {args.output}")

    elif args.cmd == "enrich":
        idx = BM25Index.load(args.bm25)
        enrich_triplets(
            triplets_path=args.triplets,
            bm25_index=idx,
            output_path=args.output,
            n_hard=args.n_hard,
            n_easy=args.n_easy,
        )
        print(f"[bm25] Enriched triplets written to {args.output}")

    else:
        parser.print_help()
