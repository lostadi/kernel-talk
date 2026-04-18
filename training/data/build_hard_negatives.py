"""training/data/build_hard_negatives.py — Wrapper for training.bm25."""
import subprocess, sys
from pathlib import Path

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "training.bm25"] + sys.argv[1:], check=True)
