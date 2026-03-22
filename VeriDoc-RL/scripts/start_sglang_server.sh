#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${VERIDOC_RL_PYTHON_BIN:-$ROOT_DIR/.venv-rl/bin/python}}"
MODEL_REF="${MODEL_REF:-${VERIDOC_MODEL_REF:-${MODEL_PATH:-${VERIDOC_MODEL_PATH:-Qwen/Qwen3-1.7B}}}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
SERVER_BIN="${SERVER_BIN:-$(dirname "$PYTHON_BIN")/sglang}"
ENABLE_LORA="${ENABLE_LORA:-0}"
LORA_PATHS="${LORA_PATHS:-}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[start_sglang_server] Missing python executable: $PYTHON_BIN" >&2
  echo "[start_sglang_server] Run bash scripts/bootstrap_autodl_envs.sh first, or set VERIDOC_RL_PYTHON_BIN / PYTHON_BIN explicitly." >&2
  exit 1
fi

looks_like_local_model_ref() {
  local value="$1"
  case "$value" in
    /*|./*|../*|~*|models/*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

if looks_like_local_model_ref "$MODEL_REF" && [[ ! -e "$MODEL_REF" ]]; then
  echo "[start_sglang_server] Local model path does not exist: $MODEL_REF" >&2
  echo "[start_sglang_server] Set MODEL_REF / VERIDOC_MODEL_REF to a valid directory, or use a Hugging Face repo id." >&2
  exit 1
fi

SERVER_ARGS=(
  --model-path "$MODEL_REF"
  --host "$HOST"
  --port "$PORT"
  --attention-backend "$ATTENTION_BACKEND"
  --sampling-backend "$SAMPLING_BACKEND"
)

if [[ "$ENABLE_LORA" == "1" ]]; then
  SERVER_ARGS+=(--enable-lora)
  if [[ -n "$LORA_PATHS" ]]; then
    SERVER_ARGS+=(--lora-paths "$LORA_PATHS")
  fi
fi

if [[ "$DISABLE_CUDA_GRAPH" == "1" ]]; then
  SERVER_ARGS+=(--disable-cuda-graph)
fi

if [[ "$DISABLE_RADIX_CACHE" == "1" ]]; then
  SERVER_ARGS+=(--disable-radix-cache)
fi

if [[ -x "$SERVER_BIN" ]]; then
  exec "$SERVER_BIN" serve "${SERVER_ARGS[@]}" "$@"
fi

exec "$PYTHON_BIN" -m sglang.launch_server "${SERVER_ARGS[@]}" "$@"
