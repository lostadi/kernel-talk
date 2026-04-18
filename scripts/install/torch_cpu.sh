#!/usr/bin/env bash
# install/torch_cpu.sh — Install CPU-only PyTorch.
set -e
INDEX="https://download.pytorch.org/whl/cpu"
echo "[install/torch_cpu] Installing CPU-only torch from $INDEX ..."
"${VENV_PIP:-pip}" install torch torchvision torchaudio --index-url "$INDEX" --quiet
echo "[install/torch_cpu] Done."
