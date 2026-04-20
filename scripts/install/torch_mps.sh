#!/usr/bin/env bash
# install/torch_mps.sh — Install PyTorch for Apple Silicon (MPS built-in).
set -e
echo "[install/torch_mps] Installing PyTorch (MPS backend is built-in) ..."
"${VENV_PIP:-pip}" install torch torchvision torchaudio --quiet
echo "[install/torch_mps] Done."
