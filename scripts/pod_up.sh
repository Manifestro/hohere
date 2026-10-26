#!/usr/bin/env bash
# Bring a RunPod pod to "ready": deps installed, Qwen served by vLLM, healthcheck passing.
# Idempotent and fast on re-runs: everything (venvs, model weights, compile caches) lives
# on /workspace. Usage:  bash scripts/pod_up.sh   (from the repo root on the pod)
set -euo pipefail

MODEL="${HOHERE_QWEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${HOHERE_QWEN_PORT:-8000}"
GPU_UTIL="${HOHERE_VLLM_GPU_UTIL:-0.80}"
MAX_LEN="${HOHERE_VLLM_MAX_LEN:-16384}"
VLLM_ENV=/workspace/vllm-env

export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/workspace/uv-cache
export HF_HOME=/workspace/hf-cache
# Compile caches (torch.compile, FlashInfer JIT, CUDA graphs) survive pod restarts here.
export XDG_CACHE_HOME=/workspace/cache
export VLLM_CACHE_ROOT=/workspace/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/workspace/cache/inductor
mkdir -p "$XDG_CACHE_HOME"

log() { echo "[pod_up $(date -u +%T)] $*"; }

if ! command -v espeak-ng > /dev/null; then
  log "installing espeak-ng"
  apt-get update -qq && apt-get install -y -qq espeak-ng tmux > /dev/null
fi
if ! command -v uv > /dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

log "syncing project env"
uv sync --all-extras --quiet

if [ ! -x "$VLLM_ENV/bin/vllm" ]; then
  log "creating vLLM env (one-time, several minutes)"
  uv venv "$VLLM_ENV" --python 3.12 --quiet
  VIRTUAL_ENV="$VLLM_ENV" uv pip install --quiet vllm ninja
fi

if curl -sf -m 5 "localhost:$PORT/v1/models" > /dev/null; then
  log "vLLM already serving on :$PORT"
else
  log "starting vLLM: $MODEL"
  # The venv must be on PATH: FlashInfer JIT-builds kernels with the venv's ninja.
  PATH="$VLLM_ENV/bin:$PATH" setsid nohup vllm serve "$MODEL" \
    --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" \
    > /workspace/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do
    if curl -sf -m 5 "localhost:$PORT/v1/models" > /dev/null; then break; fi
    if ! pgrep -f "vllm serve" > /dev/null; then
      log "vLLM died; last log lines:"; tail -30 /workspace/vllm.log; exit 1
    fi
    sleep 10
  done
fi

curl -sf -m 5 "localhost:$PORT/v1/models" > /dev/null || { log "vLLM not ready"; exit 1; }
echo "export HOHERE_QWEN_BASE_URL=http://localhost:$PORT/v1" > /workspace/hohere.env
log "ready: HOHERE_QWEN_BASE_URL=http://localhost:$PORT/v1 (source /workspace/hohere.env)"
