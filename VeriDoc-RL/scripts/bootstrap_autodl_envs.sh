#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_FLAVOR="${1:-auto}"
INSTALL_TARGET="${2:-all}"

TRAIN_ENV_DIR="${TRAIN_ENV_DIR:-$ROOT_DIR/.venv-train}"
RL_ENV_DIR="${RL_ENV_DIR:-$ROOT_DIR/.venv-rl}"

CACHE_ROOT="${AUTODL_CACHE_ROOT:-/root/autodl-tmp/.cache}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"

TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
VERL_VERSION="${VERL_VERSION:-0.4.1}"
SGLANG_VERSION="${SGLANG_VERSION:-0.4.6.post5}"
SGLANG_INSTALL_SPEC="${SGLANG_INSTALL_SPEC:-sglang[srt]==${SGLANG_VERSION}}"
FLASHINFER_TORCH_SERIES="${FLASHINFER_TORCH_SERIES:-torch2.6}"

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_autodl_envs.sh [auto|cu126|cu124] [all|train|rl]

Default behavior:
  - rebuilds .venv-train for SFT / DPO / offline inference
  - rebuilds .venv-rl for SGLang serving + verl rollout
  - keeps training and RL-serving dependencies isolated

Examples:
  bash scripts/bootstrap_autodl_envs.sh
  bash scripts/bootstrap_autodl_envs.sh cu126 all
  bash scripts/bootstrap_autodl_envs.sh auto train
  PYTHON_BIN=python3.12 bash scripts/bootstrap_autodl_envs.sh auto rl
EOF
}

info() {
  echo "[bootstrap_autodl_envs] $*"
}

warn() {
  echo "[bootstrap_autodl_envs] Warning: $*" >&2
}

die() {
  echo "[bootstrap_autodl_envs] Error: $*" >&2
  exit 1
}

require_command_or_path() {
  local command_name="$1"
  if [[ "$command_name" == */* ]]; then
    [[ -x "$command_name" ]] || die "Missing required executable: $command_name"
    return
  fi
  command -v "$command_name" >/dev/null 2>&1 || die "Missing required command: $command_name"
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  local candidate
  for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done
  die "Cannot find a usable Python interpreter. Set PYTHON_BIN explicitly."
}

assert_python_version() {
  local python_bin="$1"
  if ! "$python_bin" - <<'PY'; then
import sys

if sys.version_info < (3, 12):
    raise SystemExit(1)
PY
    die "Python 3.12+ is required. Current interpreter: $python_bin"
  fi
}

safe_rm_rf() {
  local target="$1"
  if [[ -z "$target" || "$target" != "$ROOT_DIR/"* ]]; then
    die "Refusing to remove unsafe path: $target"
  fi
  rm -rf "$target"
}

venv_python() {
  local env_dir="$1"
  echo "$env_dir/bin/python"
}

venv_pip() {
  local env_dir="$1"
  shift
  "$(venv_python "$env_dir")" -m pip "$@"
}

install_base_tooling() {
  local env_dir="$1"
  venv_pip "$env_dir" install --upgrade pip setuptools wheel
}

install_torch_stack() {
  local env_dir="$1"
  venv_pip "$env_dir" install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"
}

install_train_stack() {
  local env_dir="$1"
  install_base_tooling "$env_dir"
  install_torch_stack "$env_dir"
  venv_pip "$env_dir" install -e "$ROOT_DIR[dev,train]"
  venv_pip "$env_dir" check
}

install_rl_stack() {
  local env_dir="$1"
  install_base_tooling "$env_dir"
  install_torch_stack "$env_dir"
  venv_pip "$env_dir" install -e "$ROOT_DIR[dev]"
  venv_pip "$env_dir" install \
    --find-links "$FLASHINFER_FIND_LINKS" \
    "$SGLANG_INSTALL_SPEC"
  venv_pip "$env_dir" install pyarrow "verl==${VERL_VERSION}"
  venv_pip "$env_dir" check
}

detect_cuda_version() {
  if [[ -f /usr/local/cuda/version.json ]]; then
    "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("/usr/local/cuda/version.json").read_text(encoding="utf-8"))
print(payload.get("cuda", {}).get("version", ""))
PY
    return
  fi

  nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n 1
}

resolve_cuda_flavor() {
  local requested="$1"
  if [[ "$requested" == "cu126" || "$requested" == "cu124" ]]; then
    echo "$requested"
    return
  fi
  if [[ "$requested" != "auto" ]]; then
    die "Unsupported CUDA flavor: $requested"
  fi

  local detected_version
  detected_version="$(detect_cuda_version)"
  if [[ -z "$detected_version" ]]; then
    die "Cannot detect CUDA version. Pass cu126 or cu124 explicitly."
  fi

  case "$detected_version" in
    12.6*|12.7*|12.8*|12.9*|13.*)
      echo "cu126"
      ;;
    12.4*|12.5*)
      echo "cu124"
      ;;
    *)
      die "Detected CUDA version $detected_version. Use an AutoDL image with CUDA 12.4+."
      ;;
  esac
}

validate_install_target() {
  case "$1" in
    all|train|rl)
      return
      ;;
    *)
      die "Unsupported install target: $1"
      ;;
  esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

PYTHON_BIN="$(resolve_python_bin)"
assert_python_version "$PYTHON_BIN"
validate_install_target "$INSTALL_TARGET"

require_command_or_path "$PYTHON_BIN"
require_command_or_path git
require_command_or_path nvidia-smi

CUDA_FLAVOR="$(resolve_cuda_flavor "$CUDA_FLAVOR")"

case "$CUDA_FLAVOR" in
  cu126)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    FLASHINFER_FIND_LINKS="https://flashinfer.ai/whl/cu126/${FLASHINFER_TORCH_SERIES}/flashinfer-python/"
    ;;
  cu124)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    FLASHINFER_FIND_LINKS="https://flashinfer.ai/whl/cu124/${FLASHINFER_TORCH_SERIES}/flashinfer-python/"
    ;;
esac

if ! nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi failed. AutoDL GPU instance is not ready."
fi

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi

mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
export PIP_CACHE_DIR
export UV_CACHE_DIR

info "Using Python $PYTHON_BIN"
info "Using CUDA flavor $CUDA_FLAVOR"

if [[ "$INSTALL_TARGET" == "all" || "$INSTALL_TARGET" == "train" ]]; then
  info "Building train env at $TRAIN_ENV_DIR"
  safe_rm_rf "$TRAIN_ENV_DIR"
  "$PYTHON_BIN" -m venv "$TRAIN_ENV_DIR"
  install_train_stack "$TRAIN_ENV_DIR"
  "$(venv_python "$TRAIN_ENV_DIR")" -c "import torch, transformers, trl, huggingface_hub; print('train-env', torch.__version__, torch.version.cuda, transformers.__version__, trl.__version__, huggingface_hub.__version__)"
fi

if [[ "$INSTALL_TARGET" == "all" || "$INSTALL_TARGET" == "rl" ]]; then
  info "Building RL/SGLang env at $RL_ENV_DIR"
  safe_rm_rf "$RL_ENV_DIR"
  "$PYTHON_BIN" -m venv "$RL_ENV_DIR"
  install_rl_stack "$RL_ENV_DIR"
  "$(venv_python "$RL_ENV_DIR")" -c "import torch, pyarrow, verl, sglang, fastapi, uvicorn, uvloop; print('rl-env', torch.__version__, torch.version.cuda, verl.__version__, sglang.__version__, fastapi.__version__, uvicorn.__version__)"
fi

info "Bootstrap complete."
if [[ "$INSTALL_TARGET" == "all" || "$INSTALL_TARGET" == "train" ]]; then
  info "Train env:"
  info "  source .venv-train/bin/activate"
fi
if [[ "$INSTALL_TARGET" == "all" || "$INSTALL_TARGET" == "rl" ]]; then
  info "RL env:"
  info "  source .venv-rl/bin/activate"
  info "Start SGLang:"
  info "  bash scripts/start_sglang_server.sh"
fi
