#!/usr/bin/env bash
# detect/amd.sh — Detect AMD GPU + ROCm availability.
#
# Returns 0 if AMD GPU is present and ROCm-supported, non-zero otherwise.
# Writes to stdout: amd:<gfx_arch>:<rocm_version>
#
# ROCm officially supports: gfx900, gfx902, gfx906, gfx908, gfx90a, gfx1030,
# gfx1031, gfx1032, gfx1034, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102.
# Integrated APUs and very old dGPUs fall through to CPU.

set -e

AMD_ALLOW_LIST=(
    gfx900 gfx902 gfx906 gfx908 gfx90a
    gfx1010 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1036
    gfx1100 gfx1101 gfx1102 gfx1103
)

# Check for AMD discrete GPU via lspci
if ! command -v lspci &>/dev/null; then
    exit 1
fi

AMD_LINE=$(lspci 2>/dev/null | grep -iE '(amd|ati).*vga|vga.*amd|radeon|navi|vega|polaris' | head -1)
if [[ -z "$AMD_LINE" ]]; then
    exit 1
fi

# Try to get GFX arch from rocminfo
GFX_ARCH="unknown"
ROCM_VER="unknown"

if command -v rocminfo &>/dev/null; then
    GFX_ARCH=$(rocminfo 2>/dev/null | grep -oP 'gfx[0-9a-f]+' | head -1)
    GFX_ARCH="${GFX_ARCH:-unknown}"
fi

if command -v rocm-smi &>/dev/null; then
    ROCM_VER=$(rocm-smi --showdriverversion 2>/dev/null | grep -oP '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    ROCM_VER="${ROCM_VER:-unknown}"
fi

# Check if GFX arch is on the allow-list
if [[ "$GFX_ARCH" != "unknown" ]]; then
    SUPPORTED=false
    for allowed in "${AMD_ALLOW_LIST[@]}"; do
        if [[ "$GFX_ARCH" == "$allowed" ]]; then
            SUPPORTED=true
            break
        fi
    done
    if [[ "$SUPPORTED" == "false" ]]; then
        echo "[detect/amd] GPU arch $GFX_ARCH not in ROCm allow-list; falling back to CPU" >&2
        exit 1
    fi
fi

echo "amd:${GFX_ARCH}:${ROCM_VER}"
exit 0
