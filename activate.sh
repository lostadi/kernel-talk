#!/usr/bin/env bash
# activate.sh — Source this file to enter the kernel-talk environment.
#
#   source activate.sh
#
# After sourcing, `python cli/ktalk.py` works from any sub-directory
# and the KTALK_* env vars are pre-configured for Ollama + local storage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Default configuration (override by re-exporting before sourcing, or after)
export KTALK_STORAGE="${KTALK_STORAGE:-$HOME/.kernel-talk/store}"
export KTALK_KERNEL="${KTALK_KERNEL:-/usr/src/linux}"
export KTALK_MODEL="${KTALK_MODEL:-ollama:deepseek-coder:6.7b}"

# Convenience alias so `ktalk` works without typing `python cli/ktalk.py`
alias ktalk="python $SCRIPT_DIR/cli/ktalk.py"

echo "kernel-talk env activated."
echo "  KTALK_STORAGE = $KTALK_STORAGE"
echo "  KTALK_KERNEL  = $KTALK_KERNEL"
echo "  KTALK_MODEL   = $KTALK_MODEL"
echo ""
echo "Usage:"
echo "  ktalk index --kernel /path/to/linux --subsystem kernel/sched"
echo "  ktalk ask \"why does schedule() yield the CPU?\""
echo "  ktalk xray /proc/meminfo"
echo "  ktalk stats"
