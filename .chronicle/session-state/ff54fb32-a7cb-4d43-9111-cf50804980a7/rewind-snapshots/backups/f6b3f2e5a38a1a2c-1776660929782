"""training/data/mine_triplets.py — Wrapper script to mine triplets from git log."""
import subprocess, sys
from pathlib import Path

if __name__ == "__main__":
    # Delegate to training.mine module
    script = str(Path(__file__).parent.parent / "mine.py")
    subprocess.run([sys.executable, "-m", "training.mine"] + sys.argv[1:], check=True)
