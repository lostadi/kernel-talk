#!/usr/bin/env bash
# install/torch_cuda.sh — Install PyTorch with CUDA support.
# Called by setup.sh with VENV_PIP already set.
# Usage: VENV_PIP=/path/to/pip bash install/torch_cuda.sh [cu121|cu124]
set -e
CUDA_VER="${1:-cu124}"
INDEX="https://download.pytorch.org/whl/${CUDA_VER}"
echo "[install/torch_cuda] Installing torch from $INDEX ..."
"${VENV_PIP:-pip}" install torch torchvision torchaudio --index-url "$INDEX" --quiet
echo "[install/torch_cuda] Done."
