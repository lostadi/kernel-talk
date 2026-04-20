#!/usr/bin/env bash
# setup.sh — Kernel-Talk hardware detection and environment setup.
#
# Detects the best available compute backend (CUDA / ROCm / XPU / MPS / CPU),
# installs the matching PyTorch wheel, installs requirements.txt, and writes
# ~/.kernel-talk/env.yaml so subsequent runs skip re-detection.
#
# Usage:
#   bash scripts/setup.sh [--yes] [--kernel /path/to/linux] [--force]
#
# Flags:
#   --yes    Skip interactive prompts (accept all defaults)
#   --kernel Path to kernel source tree (stored in env.yaml)
#   --force  Re-detect even if env.yaml already exists

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJ_DIR/.venv"
KTALK_DIR="$HOME/.kernel-talk"
ENV_YAML="$KTALK_DIR/env.yaml"
SETUP_LOG="$KTALK_DIR/setup.log"
DETECT_DIR="$SCRIPT_DIR/detect"

# PyTorch wheel indexes (pinned for reproducibility)
TORCH_CUDA_INDEX="https://download.pytorch.org/whl/cu124"
TORCH_ROCM_INDEX="https://download.pytorch.org/whl/rocm6.2"
TORCH_XPU_INDEX="https://download.pytorch.org/whl/xpu"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"

# ── Argument parsing ───────────────────────────────────────────────────────────
YES=false
FORCE=false
KERNEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --yes)    YES=true; shift ;;
        --force)  FORCE=true; shift ;;
        --kernel) KERNEL_PATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash scripts/setup.sh [--yes] [--kernel PATH] [--force]"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

# ── Logging ────────────────────────────────────────────────────────────────────
mkdir -p "$KTALK_DIR"
log() {
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$ts] $*" | tee -a "$SETUP_LOG"
}
log "=== kernel-talk setup.sh started ==="

# ── Confirm function (respects --yes) ─────────────────────────────────────────
confirm() {
    local msg="$1"
    if [[ "$YES" == "true" ]]; then
        log "  [auto-yes] $msg"
        return 0
    fi
    read -rp "$msg [y/N] " ans
    [[ "${ans,,}" == "y" ]]
}

# ── Check if setup already done ───────────────────────────────────────────────
if [[ -f "$ENV_YAML" && "$FORCE" == "false" && -d "$VENV_DIR" ]]; then
    CACHED_BACKEND=$(grep '^backend:' "$ENV_YAML" 2>/dev/null | cut -d: -f2 | tr -d ' ')
    log "Existing env.yaml found (backend=$CACHED_BACKEND). Skipping re-detection."
    log "  → Run with --force to re-detect."
    log "  → Activating: source $VENV_DIR/bin/activate"
    exit 0
fi

# ── Step 1: OS detection ───────────────────────────────────────────────────────
log "Step 1: Detecting OS..."
OS_INFO="$(bash "$DETECT_DIR/os.sh")"
OS_TYPE="${OS_INFO%%:*}"
OS_ARCH="${OS_INFO##*:}"
log "  OS: $OS_TYPE, Arch: $OS_ARCH"

# ── Step 2: Hardware detection ─────────────────────────────────────────────────
log "Step 2: Detecting compute hardware..."

BACKEND="cpu"
HW_INFO=""

case "$OS_TYPE" in
    darwin)
        if bash "$DETECT_DIR/apple.sh" 2>>"$SETUP_LOG"; then
            HW_INFO="$(bash "$DETECT_DIR/apple.sh" 2>/dev/null)"
            BACKEND="mps"
            log "  ✓ Apple Silicon detected: $HW_INFO"
        else
            log "  Intel Mac: CPU-only"
            BACKEND="cpu"
        fi
        ;;
    linux|wsl2)
        # Try NVIDIA first
        NVIDIA_OUT=""
        if NVIDIA_OUT="$(bash "$DETECT_DIR/nvidia.sh" 2>>"$SETUP_LOG")"; then
            HW_INFO="$NVIDIA_OUT"
            BACKEND="cuda"
            log "  ✓ NVIDIA GPU detected: $HW_INFO"
        else
            NVIDIA_EXIT=$?
            if [[ $NVIDIA_EXIT -eq 2 ]]; then
                log "  ⚠ NVIDIA GPU found but driver too old — see $SETUP_LOG for instructions"
                log "    Install a newer driver manually, then re-run setup.sh."
                if ! confirm "  Continue with CPU fallback?"; then
                    log "Setup aborted by user."
                    exit 1
                fi
            fi

            # Try AMD
            AMD_OUT=""
            if AMD_OUT="$(bash "$DETECT_DIR/amd.sh" 2>>"$SETUP_LOG")"; then
                HW_INFO="$AMD_OUT"
                BACKEND="rocm"
                log "  ✓ AMD GPU (ROCm) detected: $HW_INFO"
            else
                # Try Intel Arc
                INTEL_OUT=""
                if INTEL_OUT="$(bash "$DETECT_DIR/intel.sh" 2>>"$SETUP_LOG")"; then
                    HW_INFO="$INTEL_OUT"
                    BACKEND="xpu"
                    log "  ✓ Intel Arc/XPU detected: $HW_INFO"
                else
                    log "  No GPU acceleration detected. Using CPU."
                    BACKEND="cpu"
                    echo ""
                    echo "  ┌─────────────────────────────────────────────────────────┐"
                    echo "  │  WARNING: CPU-only mode                                  │"
                    echo "  │  Training on CPU is very slow (hours, not minutes).      │"
                    echo "  │  Evaluation and inference are fine.                      │"
                    echo "  └─────────────────────────────────────────────────────────┘"
                    echo ""
                fi
            fi
        fi
        ;;
    windows)
        log "  Native Windows: CPU-only (use WSL2 for GPU acceleration)"
        BACKEND="cpu"
        ;;
    *)
        log "  Unknown OS: falling back to CPU"
        BACKEND="cpu"
        ;;
esac

log "  Selected backend: $BACKEND"

# ── Step 3: Create venv ────────────────────────────────────────────────────────
log "Step 3: Setting up Python virtual environment..."

PYTHON="$(command -v python3.12 || command -v python3.11 || command -v python3 || echo 'python3')"
log "  Using Python: $($PYTHON --version 2>&1)"

if [[ ! -d "$VENV_DIR" ]]; then
    log "  Creating venv at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

"$VENV_PIP" install --upgrade pip --quiet

# ── Step 4: Install PyTorch wheel ─────────────────────────────────────────────
log "Step 4: Installing PyTorch (backend=$BACKEND)..."

case "$BACKEND" in
    cuda)
        log "  Installing torch from $TORCH_CUDA_INDEX ..."
        "$VENV_PIP" install torch --index-url "$TORCH_CUDA_INDEX" --quiet
        ;;
    rocm)
        log "  Installing torch from $TORCH_ROCM_INDEX ..."
        "$VENV_PIP" install torch --index-url "$TORCH_ROCM_INDEX" --quiet
        ;;
    xpu)
        log "  Installing torch from $TORCH_XPU_INDEX ..."
        "$VENV_PIP" install torch intel-extension-for-pytorch --index-url "$TORCH_XPU_INDEX" --quiet
        ;;
    mps)
        log "  Installing standard torch (MPS is built-in since torch 1.12) ..."
        "$VENV_PIP" install torch --quiet
        ;;
    cpu)
        log "  Installing CPU-only torch from $TORCH_CPU_INDEX ..."
        "$VENV_PIP" install torch --index-url "$TORCH_CPU_INDEX" --quiet
        ;;
esac

# ── Step 5: Install requirements ──────────────────────────────────────────────
log "Step 5: Installing requirements.txt ..."
"$VENV_PIP" install -r "$PROJ_DIR/requirements.txt" --quiet

# ── Step 6: Verify backend ────────────────────────────────────────────────────
log "Step 6: Verifying $BACKEND backend ..."
PYTHON="$VENV_PY" bash "$SCRIPT_DIR/verify.sh" "$BACKEND" 2>&1 | tee -a "$SETUP_LOG"

# ── Step 7: Write env.yaml ────────────────────────────────────────────────────
log "Step 7: Writing $ENV_YAML ..."
mkdir -p "$KTALK_DIR"
cat > "$ENV_YAML" << YAML
# kernel-talk environment config — auto-generated by setup.sh
# Edit manually or re-run setup.sh --force to regenerate.

backend: ${BACKEND}
hw_info: "${HW_INFO}"
venv: "${VENV_DIR}"
kernel: "${KERNEL_PATH:-/usr/src/linux}"
storage: "${KTALK_DIR}/store"
model: "ollama:deepseek-coder:6.7b"
setup_date: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
YAML

log "=== Setup complete ==="
log ""
log "To activate the environment:"
log "  source ${VENV_DIR}/bin/activate"
log ""
log "  Or use the convenience script:"
log "  source activate.sh"
log ""
log "Backend: ${BACKEND} | Kernel: ${KERNEL_PATH:-/usr/src/linux}"
log "Log: ${SETUP_LOG}"
