#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT_DIR/.pip-cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

CUDA_FLAVOR="cu126"
WITH_VLLM=0

VLLM_VERSION="${VLLM_VERSION:-0.17.1}"
VLLM_GIT_REF="${VLLM_GIT_REF:-v${VLLM_VERSION}}"
VLLM_REPO_URL="${VLLM_REPO_URL:-https://github.com/vllm-project/vllm.git}"
VLLM_SRC_PARENT="${VLLM_SRC_PARENT:-$ROOT_DIR/.build}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-$VLLM_SRC_PARENT/vllm-$VLLM_VERSION}"
VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-1}"
VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
VLLM_CMAKE_BUILD_TYPE="${VLLM_CMAKE_BUILD_TYPE:-Release}"
VLLM_TORCH_VERSION="${VLLM_TORCH_VERSION:-2.10.0}"
VLLM_TORCHVISION_VERSION="${VLLM_TORCHVISION_VERSION:-0.25.0}"
VLLM_TORCHAUDIO_VERSION="${VLLM_TORCHAUDIO_VERSION:-2.10.0}"

RL_TORCH_VERSION="${RL_TORCH_VERSION:-2.9.1}"
RL_TORCHVISION_VERSION="${RL_TORCHVISION_VERSION:-0.24.1}"
RL_TORCHAUDIO_VERSION="${RL_TORCHAUDIO_VERSION:-2.9.1}"
VERL_VERSION="${VERL_VERSION:-0.7.1}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.9}"

usage() {
  cat <<'EOF'
Usage: bash scripts/rebuild_wsl_envs.sh [cu126|cu124] [--with-vllm]

Default behavior:
  - rebuilds only .venv-rl
  - installs torch + repo train deps + pyarrow + verl + sglang

Optional:
  --with-vllm   also rebuild .venv-vllm and build vLLM from source
EOF
}

info() {
  echo "[rebuild_wsl_envs] $*"
}

warn() {
  echo "[rebuild_wsl_envs] Warning: $*" >&2
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
}

safe_rm_rf() {
  local target="$1"
  if [[ -z "$target" || "$target" != "$ROOT_DIR/"* ]]; then
    echo "Refusing to remove unsafe path: $target" >&2
    exit 1
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
  local torch_version="$2"
  local torchvision_version="$3"
  local torchaudio_version="$4"
  venv_pip "$env_dir" install \
    --index-url "$TORCH_INDEX_URL" \
    "torch==${torch_version}" \
    "torchvision==${torchvision_version}" \
    "torchaudio==${torchaudio_version}"
}

prepare_vllm_source() {
  info "Refreshing vLLM source at $VLLM_SRC_DIR"
  mkdir -p "$VLLM_SRC_PARENT"
  safe_rm_rf "$VLLM_SRC_DIR"
  git clone --branch "$VLLM_GIT_REF" --depth 1 "$VLLM_REPO_URL" "$VLLM_SRC_DIR"
}

install_vllm_from_source() {
  local env_dir="$1"
  local build_requirements=""

  prepare_vllm_source

  if [[ -f "$VLLM_SRC_DIR/requirements/build.txt" ]]; then
    build_requirements="$VLLM_SRC_DIR/requirements/build.txt"
  elif [[ -f "$VLLM_SRC_DIR/requirements-build.txt" ]]; then
    build_requirements="$VLLM_SRC_DIR/requirements-build.txt"
  else
    echo "Cannot find vLLM build requirements file under $VLLM_SRC_DIR" >&2
    exit 1
  fi

  (
    cd "$VLLM_SRC_DIR"
    export MAX_JOBS="$VLLM_MAX_JOBS"
    export NVCC_THREADS="$VLLM_NVCC_THREADS"
    export CMAKE_BUILD_TYPE="$VLLM_CMAKE_BUILD_TYPE"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$MAX_JOBS}"
    export VLLM_USE_PRECOMPILED=0
    export PATH="$env_dir/bin:$PATH"
    export UV_CACHE_DIR
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    "$(venv_python "$env_dir")" use_existing_torch.py
    uv pip install \
      --python "$(venv_python "$env_dir")" \
      --torch-backend "$CUDA_FLAVOR" \
      -r "$build_requirements"
    uv pip install \
      --python "$(venv_python "$env_dir")" \
      --torch-backend "$CUDA_FLAVOR" \
      --no-build-isolation \
      -e .
  )
}

install_rl_stack() {
  local env_dir="$1"
  install_base_tooling "$env_dir"
  install_torch_stack \
    "$env_dir" \
    "$RL_TORCH_VERSION" \
    "$RL_TORCHVISION_VERSION" \
    "$RL_TORCHAUDIO_VERSION"
  venv_pip "$env_dir" install -e "$ROOT_DIR[dev,train]"
  venv_pip "$env_dir" install pyarrow "verl==${VERL_VERSION}" "sglang==${SGLANG_VERSION}"
  venv_pip "$env_dir" check
}

while (($# > 0)); do
  case "$1" in
    cu126|cu124)
      CUDA_FLAVOR="$1"
      ;;
    --with-vllm)
      WITH_VLLM=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

case "$CUDA_FLAVOR" in
  cu126)
    CUDA_SERIES="12.6"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
    ;;
  cu124)
    CUDA_SERIES="12.4"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
esac

require_command "$PYTHON_BIN"
require_command git

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

require_command nvcc

CUDA_VERSION=""
if [[ -f "$CUDA_HOME/version.json" ]]; then
  CUDA_VERSION="$("$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

payload = json.loads(
    Path(os.environ["CUDA_HOME"]).joinpath("version.json").read_text(encoding="utf-8")
)
print(payload.get("cuda", {}).get("version", ""))
PY
)"
fi

if [[ -z "$CUDA_VERSION" ]]; then
  echo "Cannot detect $CUDA_HOME/version.json. Install CUDA $CUDA_SERIES first." >&2
  exit 1
fi

if [[ "$CUDA_VERSION" != "$CUDA_SERIES".* ]]; then
  echo "$CUDA_HOME points to CUDA $CUDA_VERSION, expected $CUDA_SERIES.x." >&2
  echo "Install or switch the toolkit first, then rerun this script." >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi failed in the current WSL session. CUDA userspace is installed, but torch.cuda.is_available() will stay false until GPU access is restored."
fi

if (( WITH_VLLM == 1 )); then
  require_command gcc
  require_command g++
  require_command uv
fi

if [[ -r /proc/meminfo && "$WITH_VLLM" -eq 1 ]]; then
  MEM_TOTAL_KB="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  SWAP_TOTAL_KB="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
  if [[ -n "$MEM_TOTAL_KB" && -n "$SWAP_TOTAL_KB" ]]; then
    if (( MEM_TOTAL_KB < 12582912 || SWAP_TOTAL_KB < 8388608 )); then
      warn "WSL currently has about $((MEM_TOTAL_KB / 1024 / 1024)) GiB RAM and $((SWAP_TOTAL_KB / 1024 / 1024)) GiB swap. vLLM source builds are likely to be OOM-killed with exit code 137 below roughly 12 GiB RAM or 8 GiB swap."
    fi
  fi
fi

mkdir -p "$PIP_CACHE_DIR"
mkdir -p "$UV_CACHE_DIR"
export PIP_CACHE_DIR
export UV_CACHE_DIR

info "Rebuilding primary local training environment against CUDA $CUDA_VERSION from $CUDA_HOME"
safe_rm_rf "$ROOT_DIR/.venv-rl"
"$PYTHON_BIN" -m venv "$ROOT_DIR/.venv-rl"
install_rl_stack "$ROOT_DIR/.venv-rl"
"$(venv_python "$ROOT_DIR/.venv-rl")" -c "import torch, transformers, trl, pyarrow, verl, sglang; print('rl-env', torch.__version__, torch.version.cuda, transformers.__version__, trl.__version__, verl.__version__, sglang.__version__)"

if (( WITH_VLLM == 1 )); then
  info "Rebuilding optional vLLM environment"
  safe_rm_rf "$ROOT_DIR/.venv-vllm"
  "$PYTHON_BIN" -m venv "$ROOT_DIR/.venv-vllm"
  install_base_tooling "$ROOT_DIR/.venv-vllm"
  install_torch_stack \
    "$ROOT_DIR/.venv-vllm" \
    "$VLLM_TORCH_VERSION" \
    "$VLLM_TORCHVISION_VERSION" \
    "$VLLM_TORCHAUDIO_VERSION"
  install_vllm_from_source "$ROOT_DIR/.venv-vllm"
  "$(venv_python "$ROOT_DIR/.venv-vllm")" -c "import torch, vllm; print('vllm-env', torch.__version__, torch.version.cuda, vllm.__version__)"
fi

info "Rebuild complete."
info "Primary path:"
info "  source .venv-rl/bin/activate && python -c \"import torch, verl, sglang; print(torch.cuda.is_available(), verl.__version__, sglang.__version__)\""
if (( WITH_VLLM == 1 )); then
  info "Optional vLLM path:"
  info "  source .venv-vllm/bin/activate && python -c \"import torch, vllm; print(torch.cuda.is_available(), vllm.__version__)\""
else
  info "Optional vLLM path not rebuilt. Rerun with --with-vllm only if you need baseline throughput experiments."
fi
