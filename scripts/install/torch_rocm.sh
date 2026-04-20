#!/usr/bin/env bash
# install/torch_rocm.sh — Install PyTorch with ROCm support.
set -e
ROCM_VER="${1:-rocm6.2}"
INDEX="https://download.pytorch.org/whl/${ROCM_VER}"
echo "[install/torch_rocm] Installing torch from $INDEX ..."
"${VENV_PIP:-pip}" install torch torchvision torchaudio --index-url "$INDEX" --quiet
echo "[install/torch_rocm] Done."
