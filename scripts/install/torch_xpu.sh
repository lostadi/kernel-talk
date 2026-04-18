#!/usr/bin/env bash
# install/torch_xpu.sh — Install PyTorch with Intel XPU support.
set -e
INDEX="https://download.pytorch.org/whl/xpu"
echo "[install/torch_xpu] Installing torch + intel-extension-for-pytorch from $INDEX ..."
"${VENV_PIP:-pip}" install torch intel-extension-for-pytorch --index-url "$INDEX" --quiet
echo "[install/torch_xpu] Done."
