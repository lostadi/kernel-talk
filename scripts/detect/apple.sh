#!/usr/bin/env bash
# detect/apple.sh — Detect Apple Silicon + MPS availability.
#
# Returns 0 if running on Apple Silicon (arm64 macOS).
# Writes to stdout: apple:<arch>:<macos_version>

set -e

OS="$(uname -s)"
ARCH="$(uname -m)"

if [[ "$OS" != "Darwin" ]]; then
    exit 1
fi

MACOS_VER=$(sw_vers -productVersion 2>/dev/null || echo "unknown")

if [[ "$ARCH" == "arm64" ]]; then
    echo "apple:arm64:${MACOS_VER}"
    exit 0
fi

# x86_64 macOS — MPS not available
echo "[detect/apple] Intel Mac detected; MPS requires Apple Silicon (arm64)" >&2
echo "[detect/apple] Falling back to CPU" >&2
exit 1
