#!/usr/bin/env bash
# Run by scripts/start.sh (the image entrypoint) after sshd is up.
# Starts vLLM on the Qwen weights stored on the global volume, in the background.
#
#   HOHERE_QWEN_MODEL_DIR  weights dir on the volume  (default /models/Qwen3-30B-A3B-Instruct-2507-FP8)
#   HOHERE_QWEN_MODEL      served model name          (default Qwen/Qwen3-30B-A3B-Instruct-2507-FP8)
#   HOHERE_START_VLLM=0    skip starting vLLM (e.g. weight-download pods)

MODEL_DIR="${HOHERE_QWEN_MODEL_DIR:-/models/Qwen3-30B-A3B-Instruct-2507-FP8}"
SERVED="${HOHERE_QWEN_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${HOHERE_QWEN_PORT:-8000}"
LOG=/var/log/vllm.log

if [ "${HOHERE_START_VLLM:-1}" != "1" ]; then
  echo "[post_start] HOHERE_START_VLLM=0, not starting vLLM"
  exit 0
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[post_start] no weights at $MODEL_DIR (is the global volume mounted?)"
  exit 0
fi

echo "[post_start] starting vLLM: $SERVED from $MODEL_DIR"
# The venv must be on PATH (ninja for any JIT kernels). The FlashInfer sampler is the
# only kernel that needed JIT compilation on L40S; the default sampler skips that wait.
PATH="/opt/hohere/.venv/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0 setsid nohup \
  vllm serve "$MODEL_DIR" --served-model-name "$SERVED" \
  --port "$PORT" \
  --max-model-len "${HOHERE_VLLM_MAX_LEN:-16384}" \
  --gpu-memory-utilization "${HOHERE_VLLM_GPU_UTIL:-0.80}" \
  > "$LOG" 2>&1 < /dev/null &
echo "[post_start] vLLM log: $LOG"
