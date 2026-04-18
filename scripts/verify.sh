#!/usr/bin/env bash
# verify.sh — Post-install verification: confirm torch can see the expected backend.
#
# Usage: verify.sh <backend>
#   backend: cuda | rocm | mps | xpu | cpu
# Returns 0 if the backend is available, non-zero otherwise.

set -e

BACKEND="${1:-cpu}"
PYTHON="${PYTHON:-python3}"

case "$BACKEND" in
    cuda|rocm)
        $PYTHON -c "
import torch, sys
avail = torch.cuda.is_available()
if avail:
    dev = torch.cuda.get_device_name(0)
    ver = torch.version.cuda
    print(f'[verify] CUDA available: {dev}, CUDA {ver}')
else:
    print('[verify] ERROR: torch.cuda.is_available() returned False', file=sys.stderr)
    sys.exit(1)
"
        ;;
    mps)
        $PYTHON -c "
import torch, sys
avail = torch.backends.mps.is_available()
if avail:
    print('[verify] MPS available on Apple Silicon')
else:
    print('[verify] ERROR: torch.backends.mps.is_available() returned False', file=sys.stderr)
    sys.exit(1)
"
        ;;
    xpu)
        $PYTHON -c "
import torch, sys
try:
    avail = torch.xpu.is_available()
except AttributeError:
    print('[verify] ERROR: torch.xpu not present in this build', file=sys.stderr)
    sys.exit(1)
if avail:
    print('[verify] Intel XPU available')
else:
    print('[verify] ERROR: torch.xpu.is_available() returned False', file=sys.stderr)
    sys.exit(1)
"
        ;;
    cpu)
        $PYTHON -c "
import torch
print(f'[verify] CPU-only PyTorch {torch.__version__} ready')
"
        ;;
    *)
        echo "[verify] Unknown backend: $BACKEND" >&2
        exit 1
        ;;
esac
