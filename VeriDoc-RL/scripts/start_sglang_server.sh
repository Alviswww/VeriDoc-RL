#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv-rl/bin/python}"
MODEL_PATH="${MODEL_PATH:-${VERIDOC_MODEL_PATH:-$ROOT_DIR/models/Qwen3-0.6B}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
SAMPLING_BACKEND="${SAMPLING_BACKEND:-pytorch}"
SERVER_BIN="${SERVER_BIN:-$(dirname "$PYTHON_BIN")/sglang}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[start_sglang_server] Missing python executable: $PYTHON_BIN" >&2
  echo "[start_sglang_server] Run bash scripts/bootstrap_autodl_envs.sh first." >&2
  exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "[start_sglang_server] Model path does not exist: $MODEL_PATH" >&2
  echo "[start_sglang_server] Set MODEL_PATH or VERIDOC_MODEL_PATH before launching." >&2
  exit 1
fi

if [[ -x "$SERVER_BIN" ]]; then
  exec "$SERVER_BIN" serve \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --attention-backend "$ATTENTION_BACKEND" \
    --sampling-backend "$SAMPLING_BACKEND" \
    "$@"
fi

exec "$PYTHON_BIN" -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --attention-backend "$ATTENTION_BACKEND" \
  --sampling-backend "$SAMPLING_BACKEND" \
  "$@"
