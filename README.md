# hohere

Toolkit for generating, synthesizing, validating and publishing **synchronized two-channel
conversational audio datasets** for full-duplex spoken dialogue models.

```
Scenario plan ─► Qwen dialogue generator ─► dialogue validator ─► Kokoro TTS
             ─► timeline composer ─► audio QC ─► manifest ─► local export / private HF repo
```

- Dialogues written by **Qwen** (any OpenAI-compatible server: vLLM, SGLang, DashScope, Ollama).
- Speech synthesized by **Kokoro-82M**, 24 kHz mono, stable voice per speaker.
- Composer places pauses, overlaps, backchannels, interruptions, cutoffs and self-corrections
  with **sample-accurate integer timestamps**, deterministic from the recorded seed.
- Output: `channel_a.wav` (speaker A only) + `channel_b.wav` (speaker B only), identical length,
  plus `events.json` and `metadata.json` per conversation.
- Every stage is resumable, rejected items are always kept with machine-readable reasons,
  splits are fixed per scenario family before synthesis.

Status: experimental (Stage 1 — functional pipeline).

## Dataset mix

Dialogues are planned by their **primary conversational skill** (other behaviours may still
appear inside them):

| Skill | Share | Default hours (of 20) |
|---|---:|---:|
| `simple_dialogue` — questions, answers, clarifications | 40% | 8 |
| `context_memory` — referring back several turns | 20% | 4 |
| `turn_taking` — pauses, hesitation, waiting | 15% | 3 |
| `backchannel` — "mm-hm", "right" during the other's speech | 10% | 2 |
| `interruption` — stopping, clarifying, redirecting | 10% | 2 |
| `incomplete_thoughts` — self-correction, resuming | 5% | 1 |

Shares live in `dataset.skill_mix` in the config.

## Install

```bash
uv sync                      # core (planning, validation, composer, mock providers)
uv sync --extra kokoro       # + Kokoro TTS (needs `espeak-ng` system package)
uv sync --extra hf           # + Hugging Face export/publishing
uv sync --all-extras         # everything (GPU pod)
```

## Quick start (offline, no models)

```bash
uv run hohere pipeline --config configs/mock.yaml
uv run hohere export   --config configs/mock.yaml --out exports/mock
```

## Real run

Start a Qwen server (see [docs/runpod.md](docs/runpod.md)), then:

```bash
export HOHERE_QWEN_BASE_URL=http://localhost:8000/v1
uv run hohere generate   --config configs/smoke.yaml
uv run hohere synthesize --config configs/smoke.yaml
uv run hohere compose    --config configs/smoke.yaml
uv run hohere validate   --config configs/smoke.yaml
# or all at once:
uv run hohere pipeline   --config configs/smoke.yaml
uv run hohere status     --config configs/smoke.yaml
```

Interrupt any stage at any time; re-running it continues where it stopped without duplicating
conversations.

## Export and publish

```bash
uv run hohere export  --config configs/default.yaml --out exports/v0.1.0
uv run hohere export  --config configs/default.yaml --out exports/v0.1.0-train --no-text
HF_TOKEN=... uv run hohere publish --repo-id namespace/dataset-name --export-dir exports/v0.1.0 --tag v0.1.0
```

Publishing is **private by default** and refuses to upload into an existing public repo.
The token comes from `$HF_TOKEN` (or `hf auth login`) and is never written to disk.

The export loads directly with 🤗 Datasets:

```python
from datasets import load_dataset
ds = load_dataset("namespace/dataset-name")   # or a local export dir
row = ds["train"][0]
row["channel_a"], row["channel_b"], row["events"]
```

## CLI

| Command | What it does |
|---|---|
| `hohere init [path]` | write a full default config |
| `hohere plan` | build/show the plan (conversation specs + immutable splits) |
| `hohere generate` | Qwen dialogues → validation → `dialogues/accepted` / `dialogues/rejected` |
| `hohere synthesize` | Kokoro segments with fragment cache and retries |
| `hohere compose` | two-channel timelines + audio QC |
| `hohere validate` | dataset-wide structural + audio checks → `reports/validation.json` |
| `hohere manifest` | release manifest with SHA-256 of every file |
| `hohere export` | HF-compatible parquet export with dataset card |
| `hohere publish` | upload to a private HF dataset repo, verify, tag |
| `hohere pipeline` | generate → synthesize → compose → validate |
| `hohere status` | progress of each stage |

## Python API

```python
from hohere.config import load_config
from hohere.pipeline import Pipeline

cfg = load_config("configs/smoke.yaml")
pipe = Pipeline(cfg)
pipe.generate(limit=10)
pipe.synthesize()
pipe.compose()
report = pipe.validate()
```

Every stage is also usable on its own — see [examples/](examples/) and
[docs/dataset_format.md](docs/dataset_format.md).

## Development

```bash
uv run pytest                    # unit + offline integration tests
HOHERE_RUN_EXTERNAL=1 uv run pytest -m external   # real Qwen / Kokoro / HF (opt-in)
uv run ruff check . && uv run ruff format --check .
```

Generated data (`data/`, `exports/`), caches and credentials are git-ignored.

## License

MIT
