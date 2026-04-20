#!/usr/bin/env bash
# detect/os.sh — Detect OS and architecture.
# Writes one line to stdout: linux|darwin|windows followed by arch.
# Returns 0 always (OS detection always succeeds).

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        # Detect WSL2
        if grep -qi microsoft /proc/version 2>/dev/null; then
            echo "wsl2:${ARCH}"
        else
            echo "linux:${ARCH}"
        fi
        ;;
    Darwin)
        echo "darwin:${ARCH}"
        ;;
    MINGW*|CYGWIN*|MSYS*)
        echo "windows:${ARCH}"
        ;;
    *)
        echo "unknown:${ARCH}"
        ;;
esac
exit 0
