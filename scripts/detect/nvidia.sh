#!/usr/bin/env bash
# detect/nvidia.sh — Detect NVIDIA GPU and driver/CUDA versions.
#
# Returns 0 if NVIDIA GPU is present and usable, non-zero otherwise.
# Writes to stdout: nvidia:<driver_version>:<cuda_version>
# Writes warnings to stderr.

set -e

# Check for NVIDIA hardware via lspci (fallback if nvidia-smi absent)
if command -v lspci &>/dev/null; then
    if ! lspci 2>/dev/null | grep -qi 'nvidia'; then
        exit 1
    fi
fi

# Prefer nvidia-smi for authoritative info
if ! command -v nvidia-smi &>/dev/null; then
    echo "nvidia:unknown:unknown"
    echo "[detect/nvidia] nvidia-smi not found; driver may not be installed" >&2
    exit 1
fi

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
if [[ -z "$DRIVER" ]]; then
    echo "[detect/nvidia] nvidia-smi returned empty driver version" >&2
    exit 1
fi

# Extract CUDA version from nvidia-smi header (e.g. "CUDA Version: 12.2")
CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
CUDA="${CUDA:-unknown}"

# Minimum driver version for CUDA 12.x
MIN_DRIVER_MAJOR=525
DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)
if [[ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ]]; then
    echo "[detect/nvidia] Driver $DRIVER is below minimum ($MIN_DRIVER_MAJOR.x) for CUDA 12" >&2
    echo "  → Please run: sudo apt install nvidia-driver-535  (or newer)" >&2
    exit 2
fi

echo "nvidia:${DRIVER}:${CUDA}"
exit 0
