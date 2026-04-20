"""
training/synth.py
──────────────────
Synthetic training triplet generator for kernel-talk.

When no Linux kernel git repository is available (mine.py requires git history),
this module generates training triplets directly from an indexed KernelStore.
This allows training to start immediately without any git dependency.

Four generation strategies
──────────────────────────
1. Symbol-name query
   "how does {symbol_name} work" → positive = that function's node ID.
   Difficulty: 0.5.  Rationale: the query mentions the exact function name,
   making this moderately hard — the model must learn to connect "how does X
   work" to X's code, not just any function containing the word X.

2. Docstring / comment query
   If the function begins with a docstring or block comment, we use its first
   sentence as the query.  This produces the most natural (query, answer) pairs
   because a developer literally wrote what the function does.
   Difficulty: 0.3 (easiest — the query text is drawn directly from the code).

3. Caller-callee query
   "find functions that call {callee_name}" → positives = all callers in the
   store that list callee_name in their `calls` metadata field.
   Difficulty: 0.7 (hardest — the query is structural, not lexical).

4. Subsystem overview query
   "explain memory management in {subsystem}" (subsystem = first directory
   component of file_path) → positives = top 5 representative functions from
   that subsystem, chosen by code length as a proxy for importance.
   Difficulty: 0.6.

Easy negatives
──────────────
For all strategies: sample functions at random from *other* subsystems.
Cross-subsystem negatives are reliably easy (scheduler code vs. filesystem
code) without being trivially identical, so they give the model a clear
learning signal even at the start of training.

Hard negatives
──────────────
Not computed here (that would require a full BM25 index).  The `hard_negatives`
field is emitted as an empty list so that the output is compatible with
training/dataset.py, which gracefully falls back to easy negatives when
the hard list is empty.

Output format (one JSON object per line, compatible with training/dataset.py):
{
  "query":          "how does schedule work",
  "positives":      ["kernel/sched/core.c::schedule"],
  "hard_negatives": [],
  "easy_negatives": ["mm/slab.c::kmalloc", "fs/ext4/inode.c::ext4_write_begin"],
  "difficulty":     0.5,
  "source":         "synth_symbol"
}

Usage:
    python -m training.synth \\
        --storage .mirror \\
        --output  data/synth_triplets.jsonl \\
        --max-per-strategy 2000 \\
        --min-code-len 30
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class SynthTriplet:
    """A synthetically generated training triplet."""
    query: str
    positives: list[str]
    hard_negatives: list[str]
    easy_negatives: list[str]
    difficulty: float
    source: str              # e.g. "synth_symbol", "synth_docstring", …

    def to_json(self) -> str:
        return json.dumps({
            "query":          self.query,
            "positives":      self.positives,
            "hard_negatives": self.hard_negatives,
            "easy_negatives": self.easy_negatives,
            "difficulty":     self.difficulty,
            "source":         self.source,
        })


# ─── Store loader ─────────────────────────────────────────────────────────────

def _load_functions(storage: str, min_code_len: int) -> list[dict]:
    """
    Load all function nodes from the KernelStore's ChromaDB collection.

    Returns a list of flat dicts, each with:
        id, symbol_name, file_path, code, calls, uses_structs, docstring

    Paginates in batches of 1000 to avoid OOM on large collections.
    """
    import chromadb

    # KernelStore nests chroma inside a 'chroma/' subdirectory
    chroma_path = storage
    import os
    candidate = os.path.join(storage, "chroma")
    if os.path.isdir(candidate):
        chroma_path = candidate
    client = chromadb.PersistentClient(path=chroma_path)

    # The collection name used by KernelStore (see core/mirror/store.py)
    try:
        collection = client.get_collection("kernel_code")
    except Exception:
        # Fallback: try any available collection
        collections = client.list_collections()
        if not collections:
            raise RuntimeError(
                f"No ChromaDB collections found in {storage}. "
                "Run 'ktalk index' first."
            )
        collection = client.get_collection(collections[0].name)

    total = collection.count()
    print(f"[synth] Collection has {total} total documents", file=sys.stderr)

    batch_size = 1000
    all_nodes: list[dict] = []

    for offset in range(0, total, batch_size):
        result = collection.get(
            limit=batch_size,
            offset=offset,
            where={"node_type": "function"},
            include=["metadatas", "documents"],
        )
        if not result["ids"]:
            break

        for node_id, meta, doc in zip(
            result["ids"], result["metadatas"], result["documents"]
        ):
            # Code is stored in the ChromaDB document field
            code = doc if doc else meta.get("code", "")
            if len(code) < min_code_len:
                continue
            all_nodes.append({
                "id":           node_id,
                "symbol_name":  meta.get("symbol_name", ""),
                "file_path":    meta.get("file_path", ""),
                "code":         code,
                "calls":        meta.get("calls", ""),
                "uses_structs": meta.get("uses_structs", ""),
                "docstring":    meta.get("docstring", ""),
            })

        if len(result["ids"]) < batch_size:
            break

    print(f"[synth] Loaded {len(all_nodes)} function nodes "
          f"(≥{min_code_len} chars)", file=sys.stderr)
    return all_nodes


# ─── Subsystem helpers ────────────────────────────────────────────────────────

def _subsystem(file_path: str) -> str:
    """Return the first directory component of a file path (the subsystem)."""
    parts = file_path.replace("\\", "/").split("/")
    return parts[0] if parts else "unknown"


def _build_subsystem_index(nodes: list[dict]) -> dict[str, list[str]]:
    """Map subsystem name → list of node IDs."""
    idx: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        idx[_subsystem(node["file_path"])].append(node["id"])
    return idx


# ─── Docstring extraction ─────────────────────────────────────────────────────

# Matches /* ... */ block comment at the very start of a function body or just
# before a function definition.  Captures the interior text.
_BLOCK_COMMENT_RE = re.compile(
    r"/\*+\s*(.*?)\s*\*+/",
    re.DOTALL,
)

# Single-line // comment
_LINE_COMMENT_RE = re.compile(r"//+\s*(.+)")


def _extract_docstring_sentence(node: dict) -> str | None:
    """
    Try to extract the first meaningful sentence from the node's docstring
    metadata or, as a fallback, from the leading comment in the code body.

    Returns None if no usable docstring is found.
    """
    # Prefer the explicit docstring field (set by KernelParser)
    doc = node.get("docstring", "").strip()

    if not doc:
        # Try to find a leading block comment in the code
        m = _BLOCK_COMMENT_RE.search(node["code"][:800])
        if m:
            doc = m.group(1).strip()
        else:
            m = _LINE_COMMENT_RE.search(node["code"][:400])
            if m:
                doc = m.group(1).strip()

    if not doc:
        return None

    # Flatten whitespace / asterisks used for comment box-drawing
    doc = re.sub(r"\s*\n\s*\*?\s*", " ", doc).strip()
    doc = re.sub(r"\s+", " ", doc)

    # Take the first sentence (up to the first ". " or end of string)
    sentence_end = re.search(r"\.\s", doc)
    if sentence_end:
        sentence = doc[: sentence_end.start() + 1].strip()
    else:
        sentence = doc.strip()

    # Quality gate: must be a plausible natural language sentence
    if len(sentence) < 15 or len(sentence) > 300:
        return None
    # Skip sentences that are mostly C code (too many braces/semicolons)
    if sentence.count("{") + sentence.count(";") > 3:
        return None

    return sentence


# ─── Easy negative sampler ────────────────────────────────────────────────────

def _easy_negatives(
    positive_subsystems: set[str],
    subsystem_index: dict[str, list[str]],
    n: int = 5,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Sample `n` random function IDs from subsystems other than the positives'.
    """
    if rng is None:
        rng = random.Random()

    other_ids: list[str] = []
    for sub, ids in subsystem_index.items():
        if sub not in positive_subsystems:
            other_ids.extend(ids)

    if not other_ids:
        return []
    return rng.sample(other_ids, min(n, len(other_ids)))


# ─── Generation strategies ────────────────────────────────────────────────────

def _strategy_symbol(
    nodes: list[dict],
    subsystem_index: dict[str, list[str]],
    max_n: int,
    n_easy: int,
    rng: random.Random,
) -> Iterator[SynthTriplet]:
    """
    Strategy 1 — symbol-name query.
    "how does {symbol_name} work"  →  positive = that function.
    """
    candidates = [n for n in nodes if n["symbol_name"]]
    rng.shuffle(candidates)

    for node in candidates[:max_n]:
        sym = node["symbol_name"]
        pos_sub = {_subsystem(node["file_path"])}
        yield SynthTriplet(
            query=f"how does {sym} work",
            positives=[node["id"]],
            hard_negatives=[],
            easy_negatives=_easy_negatives(pos_sub, subsystem_index, n_easy, rng),
            difficulty=0.5,
            source="synth_symbol",
        )


def _strategy_docstring(
    nodes: list[dict],
    subsystem_index: dict[str, list[str]],
    max_n: int,
    n_easy: int,
    rng: random.Random,
) -> Iterator[SynthTriplet]:
    """
    Strategy 2 — docstring/comment first-sentence query.
    First sentence of the leading comment  →  positive = that function.
    """
    candidates = [n for n in nodes if n["symbol_name"]]
    rng.shuffle(candidates)

    produced = 0
    for node in candidates:
        if produced >= max_n:
            break
        sentence = _extract_docstring_sentence(node)
        if sentence is None:
            continue
        pos_sub = {_subsystem(node["file_path"])}
        yield SynthTriplet(
            query=sentence,
            positives=[node["id"]],
            hard_negatives=[],
            easy_negatives=_easy_negatives(pos_sub, subsystem_index, n_easy, rng),
            difficulty=0.3,
            source="synth_docstring",
        )
        produced += 1


def _strategy_caller_callee(
    nodes: list[dict],
    subsystem_index: dict[str, list[str]],
    max_n: int,
    n_easy: int,
    rng: random.Random,
) -> Iterator[SynthTriplet]:
    """
    Strategy 3 — caller-callee structural query.
    "find functions that call {callee}"  →  positives = all callers.

    We build an inverted index: callee_name → [caller_node_ids].
    """
    # Build callee → callers index
    callee_to_callers: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        calls_str = node.get("calls", "")
        if not calls_str:
            continue
        for callee in calls_str.split(","):
            callee = callee.strip()
            if callee:
                callee_to_callers[callee].append(node["id"])

    # Shuffle callee names and emit triplets
    callee_names = list(callee_to_callers.keys())
    rng.shuffle(callee_names)

    produced = 0
    for callee in callee_names:
        if produced >= max_n:
            break
        callers = callee_to_callers[callee]
        if not callers:
            continue

        # Determine subsystems of all callers for easy-negative exclusion
        caller_subs = {
            _subsystem(n["file_path"])
            for n in nodes
            if n["id"] in set(callers)
        }

        yield SynthTriplet(
            query=f"find functions that call {callee}",
            positives=callers,
            hard_negatives=[],
            easy_negatives=_easy_negatives(caller_subs, subsystem_index, n_easy, rng),
            difficulty=0.7,
            source="synth_caller_callee",
        )
        produced += 1


def _strategy_subsystem(
    nodes: list[dict],
    subsystem_index: dict[str, list[str]],
    max_n: int,
    n_easy: int,
    rng: random.Random,
) -> Iterator[SynthTriplet]:
    """
    Strategy 4 — subsystem overview query.
    "explain memory management in {subsystem}"  →  top-5 longest functions.

    Longer functions are used as a proxy for importance (they tend to be
    the core logic, not trivial helpers).  This is a rough heuristic but
    better than random selection within the subsystem.
    """
    # Map subsystem → nodes sorted descending by code length
    sub_to_nodes: dict[str, list[dict]] = defaultdict(list)
    for node in nodes:
        sub_to_nodes[_subsystem(node["file_path"])].append(node)

    subsystems = list(sub_to_nodes.keys())
    rng.shuffle(subsystems)

    # Generic subsystem purpose descriptions for a plausible natural query
    _SUBSYSTEM_TOPIC: dict[str, str] = {
        "mm":        "memory management",
        "kernel":    "core kernel",
        "fs":        "filesystem",
        "net":       "networking",
        "drivers":   "device drivers",
        "arch":      "architecture",
        "block":     "block I/O",
        "crypto":    "cryptography",
        "ipc":       "inter-process communication",
        "security":  "security",
        "sound":     "audio",
        "lib":       "library utilities",
    }

    produced = 0
    for sub in subsystems:
        if produced >= max_n:
            break

        sub_nodes = sub_to_nodes[sub]
        if len(sub_nodes) < 3:
            continue  # Too few functions — not a real subsystem node

        # Top 5 by code length
        top5 = sorted(sub_nodes, key=lambda n: len(n["code"]), reverse=True)[:5]
        positives = [n["id"] for n in top5]

        topic = _SUBSYSTEM_TOPIC.get(sub, sub.replace("_", " "))
        query = f"explain {topic} in {sub}"

        yield SynthTriplet(
            query=query,
            positives=positives,
            hard_negatives=[],
            easy_negatives=_easy_negatives({sub}, subsystem_index, n_easy, rng),
            difficulty=0.6,
            source="synth_subsystem",
        )
        produced += 1


# ─── Main generator ───────────────────────────────────────────────────────────

def generate_triplets(
    storage: str,
    output: str,
    max_per_strategy: int = 2000,
    min_code_len: int = 30,
    n_easy: int = 5,
    seed: int = 42,
) -> None:
    """
    Generate synthetic training triplets from a KernelStore and write to JSONL.

    Args:
        storage:            Path to the KernelStore directory (ChromaDB).
        output:             Path to write the output JSONL file.
        max_per_strategy:   Maximum triplets per generation strategy.
        min_code_len:       Skip functions whose code body is shorter than this.
        n_easy:             Number of easy negatives per triplet.
        seed:               Random seed for reproducibility.
    """
    rng = random.Random(seed)

    # 1. Load all function nodes
    nodes = _load_functions(storage, min_code_len)
    if not nodes:
        print("[synth] ERROR: no function nodes found.  Is the store indexed?",
              file=sys.stderr)
        sys.exit(1)

    subsystem_index = _build_subsystem_index(nodes)
    print(f"[synth] {len(subsystem_index)} subsystems found", file=sys.stderr)

    # 2. Run all strategies
    strategies = [
        ("symbol",        _strategy_symbol),
        ("docstring",     _strategy_docstring),
        ("caller_callee", _strategy_caller_callee),
        ("subsystem",     _strategy_subsystem),
    ]

    all_triplets: list[SynthTriplet] = []
    counts: dict[str, int] = {}

    for name, fn in strategies:
        before = len(all_triplets)
        for triplet in fn(nodes, subsystem_index, max_per_strategy, n_easy, rng):
            all_triplets.append(triplet)
        counts[name] = len(all_triplets) - before
        print(f"[synth] Strategy '{name}': {counts[name]} triplets",
              file=sys.stderr)

    # 3. Shuffle and write
    rng.shuffle(all_triplets)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for triplet in all_triplets:
            f.write(triplet.to_json() + "\n")

    # 4. Print stats
    total = len(all_triplets)
    print(f"\n[synth] ── Stats ─────────────────────────────────", file=sys.stderr)
    print(f"[synth]   Total triplets : {total}", file=sys.stderr)
    for name, count in counts.items():
        pct = 100.0 * count / total if total else 0.0
        print(f"[synth]   {name:<20}: {count:>6}  ({pct:.1f}%)", file=sys.stderr)
    print(f"[synth]   Output         : {output_path}", file=sys.stderr)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic (query, positive) training triplets from a "
            "KernelStore, without requiring a Linux kernel git repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="Path to the KernelStore directory (ChromaDB persistent storage)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for synthetic triplets",
    )
    parser.add_argument(
        "--max-per-strategy",
        type=int,
        default=2000,
        metavar="N",
        help="Maximum triplets to generate per strategy (default: 2000)",
    )
    parser.add_argument(
        "--min-code-len",
        type=int,
        default=30,
        metavar="CHARS",
        help="Skip functions with code shorter than this many characters (default: 30)",
    )
    parser.add_argument(
        "--n-easy",
        type=int,
        default=5,
        metavar="N",
        help="Number of easy negatives per triplet (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    generate_triplets(
        storage=args.storage,
        output=args.output,
        max_per_strategy=args.max_per_strategy,
        min_code_len=args.min_code_len,
        n_easy=args.n_easy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
