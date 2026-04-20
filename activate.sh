#!/usr/bin/env bash
# activate.sh — Kernel-Talk environment setup
#
# Usage:
#   source activate.sh
#   source activate.sh --kernel /path/to/linux

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

# ── Parse optional --kernel argument ──────────────────────────────────────────
KERNEL_PATH="/usr/src/linux"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel) KERNEL_PATH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; shift ;;
    esac
done

# ── Create venv if needed ──────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[ktalk] Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
    echo "[ktalk] Installing dependencies ..."
    "$VENV_DIR/bin/pip" install --upgrade pip --quiet
    "$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet
    echo "[ktalk] Dependencies installed."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"

# ── Environment defaults ───────────────────────────────────────────────────────
export KTALK_STORAGE="${KTALK_STORAGE:-$HOME/.kernel-talk/store}"
export KTALK_KERNEL="${KTALK_KERNEL:-$KERNEL_PATH}"
export KTALK_MODEL="${KTALK_MODEL:-ollama:deepseek-coder:6.7b}"

# ── Shell alias ────────────────────────────────────────────────────────────────
alias ktalk="python3 $SCRIPT_DIR/cli/ktalk.py"

echo "[ktalk] Environment ready."
echo "  Storage : $KTALK_STORAGE"
echo "  Kernel  : $KTALK_KERNEL"
echo "  Model   : $KTALK_MODEL"
echo ""
echo "  Commands: ktalk index | ktalk ask | ktalk xray | ktalk probe | ktalk stats | ktalk graph"
echo "  Run 'ktalk --help' for details."
