# Running on RunPod

One GPU pod runs both the Qwen server (vLLM) and Kokoro. Kokoro-82M needs ~1–2 GB of VRAM,
so give vLLM most of the card and leave a little headroom.

## Pod

| Qwen model | GPU | Notes |
|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | L40S / A6000 48 GB | good default, fast MoE |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` (BF16) | A100 / H100 80 GB | |

Use a PyTorch/CUDA template, and put the project and caches on the network volume
(`/workspace`) so they survive pod restarts.

## Setup

```bash
cd /workspace
git clone https://github.com/Manifestro/hohere.git && cd hohere
bash scripts/runpod_setup.sh          # apt espeak-ng, uv, project env, separate vLLM env
```

## Start Qwen (separate venv, so vLLM's torch never conflicts with Kokoro's)

```bash
source /workspace/vllm-env/bin/activate
vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.80 \
  > /workspace/vllm.log 2>&1 &
deactivate
curl -s localhost:8000/v1/models   # wait until it answers
```

Set `dialogue.model` in the config to the exact model name you served.

## Generate

```bash
export HOHERE_QWEN_BASE_URL=http://localhost:8000/v1
export HF_HOME=/workspace/hf-cache
uv run hohere pipeline --config configs/smoke.yaml        # ~15 min of audio, sanity check
uv run hohere status   --config configs/smoke.yaml
```

Then the real run, inside `tmux` so it survives disconnects:

```bash
tmux new -s hohere
uv run hohere pipeline --config configs/production.yaml
```

If the pod restarts, just run the same command again — every stage resumes.

## Opt-in integration tests on the pod

```bash
HOHERE_RUN_EXTERNAL=1 uv run pytest -m external
```

## Publish

```bash
uv run hohere export  --config configs/production.yaml --out /workspace/exports/v0.1.0
HF_TOKEN=... uv run hohere publish --repo-id <namespace>/hohere --export-dir /workspace/exports/v0.1.0 --tag v0.1.0
```

Never put the token in a config file or the repository.
