#!/usr/bin/env bash
# One-time setup on a RunPod GPU pod. Idempotent.
set -euo pipefail

apt-get update -qq && apt-get install -y -qq espeak-ng tmux > /dev/null

if ! command -v uv > /dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

export UV_CACHE_DIR=/workspace/uv-cache
export HF_HOME=/workspace/hf-cache

# Project env: hohere + Kokoro + HF publishing.
uv sync --all-extras

# Separate env for the Qwen server so its torch/CUDA pins never clash with Kokoro.
if [ ! -x /workspace/vllm-env/bin/vllm ]; then
  uv venv /workspace/vllm-env --python 3.12
  VIRTUAL_ENV=/workspace/vllm-env uv pip install vllm ninja
fi

echo "done. next: start vLLM (docs/runpod.md), then: uv run hohere pipeline -c configs/smoke.yaml"
