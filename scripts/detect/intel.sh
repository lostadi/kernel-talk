#!/usr/bin/env bash
# detect/intel.sh — Detect Intel Arc/Xe GPU + oneAPI/XPU availability.
#
# Returns 0 if Intel discrete GPU + compute runtime is present.
# Writes to stdout: intel:<device>:<runtime_version>

set -e

if ! command -v lspci &>/dev/null; then
    exit 1
fi

INTEL_LINE=$(lspci 2>/dev/null | grep -iE 'intel.*(arc|xe|a[0-9]{3}|uhd 7[0-9]{2}|iris xe)' | head -1)
if [[ -z "$INTEL_LINE" ]]; then
    exit 1
fi

DEVICE=$(echo "$INTEL_LINE" | grep -oP 'Intel.*' | head -1 | cut -c1-60)

# Check for Intel compute runtime (needed for XPU/SYCL)
RUNTIME_VER="unknown"
if command -v dpkg &>/dev/null; then
    RUNTIME_VER=$(dpkg -l intel-compute-runtime 2>/dev/null | awk '/^ii/{print $3}' | head -1)
fi
if [[ -z "$RUNTIME_VER" ]]; then
    echo "[detect/intel] Intel GPU found but intel-compute-runtime not installed" >&2
    echo "  → Install: https://github.com/intel/compute-runtime/releases" >&2
    exit 1
fi

echo "intel:${DEVICE}:${RUNTIME_VER}"
exit 0
