# Running on RunPod

One GPU pod runs both the Qwen server (vLLM) and Kokoro. Kokoro-82M needs ~1–2 GB of VRAM,
so vLLM gets 80% of the card.

## Pod

| Qwen model | GPU | Notes |
|---|---|---|
| `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` | L40S / A6000 48 GB | default, fast MoE |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` (BF16) | A100 / H100 80 GB | set `dialogue.model` to match |

Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, SSH enabled, `/workspace` volume
of ~150 GB. Pass `HF_TOKEN` and a GitHub token as RunPod secrets
(`{{ RUNPOD_SECRET_HF_TOKEN }}`), never in files.

A stopped pod's disk is pinned to its host; if that host's GPU is taken, the pod cannot
start. A network volume avoids this: any pod in the same data center can mount it.

## One command to "ready"

```bash
cd /workspace
git clone https://github.com/Manifestro/hohere.git && cd hohere   # first time only
bash scripts/pod_up.sh
source /workspace/hohere.env
```

`pod_up.sh` is idempotent: it installs `espeak-ng` and uv if missing, syncs the project env,
creates a separate vLLM env (so vLLM's torch never clashes with Kokoro's), starts
`vllm serve` and waits for the healthcheck. Venvs, model weights and all compile caches
(torch.compile, FlashInfer JIT, CUDA graphs) live on `/workspace`, so the second start
skips installs and compilation. Override with `HOHERE_QWEN_MODEL`, `HOHERE_QWEN_PORT`,
`HOHERE_VLLM_GPU_UTIL`, `HOHERE_VLLM_MAX_LEN`.

## Generate

```bash
uv run hohere pipeline --config configs/smoke.yaml     # ~15 min of audio, sanity check
uv run hohere pipeline --config configs/stage1.yaml    # Stage 1: 30 min deliverable
uv run hohere status   --config configs/stage1.yaml
```

Long runs go inside `tmux`. If the pod restarts, run `pod_up.sh` and the same command
again: every stage resumes.

## Opt-in integration tests

```bash
HOHERE_RUN_EXTERNAL=1 uv run pytest -m external
```

## Publish

```bash
uv run hohere export  --config configs/production.yaml --out /workspace/exports/v0.1.0
uv run hohere publish --repo-id <namespace>/hohere --export-dir /workspace/exports/v0.1.0 --tag v0.1.0
```
