"""Export to a Hugging Face Datasets layout and publish to a private Hub repository.

Export layout (loadable with `datasets.load_dataset("<dir or repo>")`):

    README.md                          dataset card (YAML header declares the splits)
    data/<split>-<NNNNN>-of-<MMMMM>.parquet
    manifest.json                      release manifest with per-file checksums
    splits.json                        immutable split manifest
    generation_config.yaml             full configuration used to generate the data
    reports/validation.json            latest dataset validation report

Requires `uv sync --extra hf`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from hohere.config.schema import HohereConfig, dump_config
from hohere.storage.local import DatasetStore, read_json, write_json
from hohere.storage.manifest import build_manifest, sha256_file

SPLITS = ("train", "validation", "test")


class PublishError(RuntimeError):
    pass


def _features():
    from datasets import Audio, Features, Value

    event = {
        "event_id": Value("string"),
        "speaker": Value("string"),
        "utterance_id": Value("string"),
        "event_type": Value("string"),
        "start_sample": Value("int64"),
        "end_sample": Value("int64"),
        "related_event_id": Value("string"),
        "cause_event_id": Value("string"),
        "text": Value("string"),
    }
    return Features(
        {
            "id": Value("string"),
            "channel_a": Audio(sampling_rate=24000),
            "channel_b": Audio(sampling_rate=24000),
            "duration_samples": Value("int64"),
            "duration_seconds": Value("float64"),
            "words_a": Value("int64"),
            "words_b": Value("int64"),
            "sample_rate": Value("int32"),
            "language": Value("string"),
            "skill": Value("string"),
            "scenario": Value("string"),
            "topic": Value("string"),
            "voice_a": Value("string"),
            "voice_b": Value("string"),
            "seed": Value("int64"),
            "events": [event],
            "metadata_json": Value("string"),
        }
    )


def _row(store: DatasetStore, cid: str, include_text: bool) -> dict:
    d = store.conversation_dir(cid)
    meta = store.load_metadata(cid)
    events = []
    for e in store.load_events(cid).events:
        ev = e.model_dump(mode="json")
        if not include_text:
            ev["text"] = None
        events.append(ev)
    return {
        "id": cid,
        "channel_a": {"bytes": (d / "channel_a.wav").read_bytes(), "path": f"{cid}_a.wav"},
        "channel_b": {"bytes": (d / "channel_b.wav").read_bytes(), "path": f"{cid}_b.wav"},
        "duration_samples": meta.duration_samples,
        "duration_seconds": meta.duration_seconds,
        "words_a": meta.words_a,
        "words_b": meta.words_b,
        "sample_rate": meta.sample_rate,
        "language": meta.language,
        "skill": meta.skill.value,
        "scenario": meta.scenario,
        "topic": meta.topic,
        "voice_a": meta.speakers["A"],
        "voice_b": meta.speakers["B"],
        "seed": meta.seed,
        "events": events,
        "metadata_json": meta.model_dump_json(),
    }


def export_dataset(
    store: DatasetStore,
    cfg: HohereConfig,
    out_dir: Path,
    include_text: bool = True,
    shard_size: int = 64,
) -> dict:
    """Write passed conversations as parquet shards per split plus card and manifests."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:
        raise PublishError("export needs the hf extra: `uv sync --extra hf`") from e

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir / "data", ignore_errors=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(store, cfg)
    by_split: dict[str, list[str]] = defaultdict(list)
    for conv in manifest["conversations"]:
        by_split[conv["split"]].append(conv["id"])
    if not by_split:
        raise PublishError("no passed conversations to export; run compose and validate first")

    features = _features()
    shard_files: dict[str, list[str]] = {}
    for split in SPLITS:
        ids = sorted(by_split.get(split, []))
        if not ids:
            continue
        n_shards = (len(ids) + shard_size - 1) // shard_size
        shard_files[split] = []
        for k in range(n_shards):
            chunk = ids[k * shard_size : (k + 1) * shard_size]
            rows = [_row(store, cid, include_text) for cid in chunk]
            name = f"data/{split}-{k:05d}-of-{n_shards:05d}.parquet"
            # Rows already hold encoded WAV bytes, so write Arrow directly with the HF schema
            # (Features metadata keeps `channel_a`/`channel_b` typed as Audio columns).
            table = pa.Table.from_pylist(rows, schema=features.arrow_schema)
            pq.write_table(table, out_dir / name)
            shard_files[split].append(name)

    manifest["export"] = {
        "include_text": include_text,
        "files": {
            name: {"sha256": sha256_file(out_dir / name), "bytes": (out_dir / name).stat().st_size}
            for names in shard_files.values()
            for name in names
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(store.manifest_path, manifest)
    shutil.copy2(store.splits_path, out_dir / "splits.json")
    (out_dir / "generation_config.yaml").write_text(dump_config(cfg))
    report = store.reports_dir / "validation.json"
    if report.exists():
        (out_dir / "reports").mkdir(exist_ok=True)
        shutil.copy2(report, out_dir / "reports" / "validation.json")
    validation = read_json(report)["summary"] if report.exists() else None
    (out_dir / "README.md").write_text(dataset_card(cfg, manifest, shard_files, validation))
    return manifest


def _size_category(n: int) -> str:
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return label
    return "100K<n<1M"


def dataset_card(
    cfg: HohereConfig,
    manifest: dict,
    shard_files: dict[str, list[str]],
    validation: dict | None = None,
) -> str:
    t = manifest["totals"]
    pub, sc = cfg.publishing, cfg.speech
    name = (pub.repository_id or cfg.project.name).split("/")[-1]
    words, hours = t["words"], t["hours"]
    configs = "\n".join(
        f"  - split: {split}\n    path: data/{split}-*.parquet" for split in shard_files
    )
    split_rows = "\n".join(
        f"| {k} | {t['words_by_split'].get(k, 0):,} | {v:.1f} |"
        for k, v in t["hours_by_split"].items()
    )
    skill_rows = "\n".join(
        f"| `{k}` | {t['words_by_skill'].get(k, 0):,} | {v:.1f} | "
        f"{t['words_by_skill'].get(k, 0) / max(words, 1) * 100:.0f}% | "
        f"{t['target_share_by_skill'].get(k, 0) * 100:.0f}% |"
        for k, v in t["hours_by_skill"].items()
    )
    events = (validation or {}).get("event_counts", {})
    per_k = 1000 / max(words, 1)
    event_rows = "\n".join(
        f"| `{k}` | {v:,} | {v * per_k:.1f} |"
        for k, v in sorted(events.items(), key=lambda x: -x[1])
    )
    event_table = (
        f"| Event | Count | Per 1,000 words |\n|---|---:|---:|\n{event_rows}"
        if event_rows
        else "See `events` in each row."
    )
    return f"""---
license: {pub.license}
language:
- {cfg.dataset.language}
pretty_name: "{name}: synthetic full-duplex spoken dialogues"
size_categories:
- {_size_category(t["conversations"])}
task_categories:
- audio-to-audio
- automatic-speech-recognition
tags:
- full-duplex
- spoken-dialogue
- conversational
- turn-taking
- voice-assistant
- synthetic
- two-channel
configs:
- config_name: default
  data_files:
{configs}
---

# {name} (v{manifest["version"]}): synthetic full-duplex spoken dialogues

> **Fully synthetic.** Every word was written by a language model and every voice is a
> text-to-speech voice. No real person was recorded.

Two-channel conversations between a **human user** (channel A) and an **AI voice
assistant** (channel B) for training and evaluating full-duplex spoken dialogue models:
models that listen and speak at the same time, take turns, backchannel, interrupt and get
interrupted.

| | |
|---|---:|
| Words, total | **{words:,}** |
| Words heard (user, channel A) | {t["words_heard_a"]:,} |
| Words spoken (assistant, channel B) | {t["words_spoken_b"]:,} |
| Conversations | {t["conversations"]:,} |
| Audio | {hours:.1f} h (two channels, {manifest["sample_rate"]} Hz) |
| Words per conversation (mean) | {words / max(t["conversations"], 1):.0f} |

## Format

Each row is one conversation:

| Field | Description |
|---|---|
| `channel_a` | Audio, user only, 16-bit PCM mono, {manifest["sample_rate"]} Hz |
| `channel_b` | Audio, assistant only, same length as `channel_a` |
| `events` | Sample-accurate events: type, speaker, start/end sample, text |
| `words_a`, `words_b` | Words in each channel's text |
| `skill`, `scenario`, `topic` | Primary conversational skill and scenario family |
| `voice_a`, `voice_b` | TTS voices |
| `metadata_json` | Full per-conversation metadata |

Channels are never mixed: overlapping speech appears at the same sample positions in
both channels. Outside its own speech a channel is digital silence.

```python
from datasets import load_dataset

ds = load_dataset("<this repo>", split="train", streaming=True)
row = next(iter(ds))
row["channel_a"]["array"], row["channel_b"]["array"], row["events"]
```

## Splits

Splits are fixed before synthesis and grouped by scenario family (skill x topic), so
near-duplicate conversations never cross splits.

| Split | Words | Hours |
|---|---:|---:|
{split_rows}

## Conversational skills

Each conversation is generated around one primary skill; others appear naturally too.

| Skill | Words | Hours | Share | Target |
|---|---:|---:|---:|---:|
{skill_rows}

## Full-duplex events

{event_table}

## How it was made

1. **Dialogue**: `{cfg.dialogue.model}` writes each conversation as structured JSON
   (speaker, text, intent, event, reference to the turn it reacts to). A validator rejects
   malformed, monologue-like, duplicate or off-skill dialogues; every rejection is logged.
2. **Speech**: Kokoro-82M. The assistant always uses one voice (`{sc.model_voice}`); the
   user voice is drawn per conversation from {len(sc.user_voices)} voices, with small
   speed variation.
3. **Timeline**: utterances are placed on two channels with natural pauses, overlaps,
   backchannels inside the other speaker's turn, interruptions that cut the interrupted
   speaker off, self-cutoffs and self-corrections.
4. **Quality control**: sample-accurate event checks, speaker isolation, clipping and
   silence checks on every conversation; only passed conversations are published.

Code: [Manifestro/hohere](https://github.com/Manifestro/hohere).

## Intended use and limitations

- **Use**: pre-training and fine-tuning full-duplex / turn-taking speech models,
  evaluation of interruption and backchannel handling, voice activity projection.
- **Synthetic speech**: limited voice diversity, clean studio-like audio, no background
  noise, TTS prosody. Expect a domain gap to real conversations.
- **English only**, everyday topics; dialogue style reflects the generating LLM.
- A cut-off utterance keeps its full source `text`, while its audio stops early.

## License and attribution

**{pub.license.upper()}**, by {pub.attribution}. Please credit "{pub.attribution}, hohere"
when you use this dataset.

Generated with Qwen3 (Apache-2.0) and Kokoro-82M (Apache-2.0).

## Provenance

`manifest.json` lists every source file with its SHA-256, software versions and models;
`generation_config.yaml` holds the full configuration (seed {manifest["seed"]},
fingerprint `{manifest["config_fingerprint"]}`); `splits.json` is the split assignment.
Schema version {manifest["schema_version"]}.

## Citation

```bibtex
@misc{{hohere{manifest["version"].replace(".", "")},
  title  = {{hohere: synthetic full-duplex spoken dialogues}},
  author = {{{pub.attribution}}},
  year   = {{2026}},
  note   = {{Version {manifest["version"]}}},
  url    = {{https://github.com/Manifestro/hohere}}
}}
```
"""


def _api(token_env: str):
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise PublishError("publishing needs the hf extra: `uv sync --extra hf`") from e
    # None falls back to the cached `hf auth login` token. Never logged or persisted.
    return HfApi(token=os.environ.get(token_env) or None)


def publish_dataset(
    export_dir: Path,
    repo_id: str,
    private: bool = True,
    tag: str | None = None,
    token_env: str = "HF_TOKEN",
    allow_public: bool = False,
) -> dict:
    """Create/update a dataset repo, upload the export, verify it, optionally tag it."""
    export_dir = Path(export_dir)
    manifest = read_json(export_dir / "manifest.json")
    if not private and not allow_public:
        raise PublishError("refusing to publish publicly without allow_public=True")

    api = _api(token_env)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    info = api.repo_info(repo_id, repo_type="dataset")
    if private and not info.private:
        raise PublishError(f"{repo_id} exists and is PUBLIC; refusing to upload private data")

    commit = api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Release {manifest['version']} ({manifest['config_fingerprint']})",
    )

    problems = verify_remote(api, repo_id, export_dir, manifest)
    if problems:
        raise PublishError("upload verification failed: " + "; ".join(problems[:10]))

    if tag:
        existing = {t.name for t in api.list_repo_refs(repo_id, repo_type="dataset").tags}
        if tag in existing:
            raise PublishError(f"tag {tag!r} already exists on {repo_id}")
        api.create_tag(repo_id, tag=tag, repo_type="dataset", tag_message=f"Release {tag}")
    return {
        "repo_id": repo_id,
        "private": private,
        "commit": getattr(commit, "oid", None) or str(commit),
        "tag": tag,
        "files": len(manifest["export"]["files"]),
    }


def verify_remote(api, repo_id: str, export_dir: Path, manifest: dict) -> list[str]:
    expected = dict(manifest["export"]["files"])
    for name in ("README.md", "manifest.json", "splits.json", "generation_config.yaml"):
        expected.setdefault(name, None)
    remote = set(api.list_repo_files(repo_id, repo_type="dataset"))
    problems = [f"missing remote file {n}" for n in expected if n not in remote]
    parquet = [n for n, v in expected.items() if v and n in remote]
    if parquet:
        for info in api.get_paths_info(repo_id, parquet, repo_type="dataset"):
            lfs = getattr(info, "lfs", None)
            want = expected[info.path]["sha256"]
            if lfs is not None and getattr(lfs, "sha256", want) != want:
                problems.append(f"checksum mismatch {info.path}")
    local_manifest = json.loads((export_dir / "manifest.json").read_text())
    if local_manifest["export"]["files"] != manifest["export"]["files"]:
        problems.append("local manifest changed during upload")
    return problems


# -- dialogue hand-off: text generated on one machine, audio synthesized on another ------

DIALOGUE_PATTERNS = ("plan.json", "splits.json", "reports/plan_summary.json", "dialogues/**")


def push_dialogues(
    store: DatasetStore, repo_id: str, prefix: str, token_env: str = "HF_TOKEN"
) -> dict:
    """Upload the plan, split manifest and all dialogues (accepted + rejected) under
    `prefix/` of a private dataset repo. No audio is uploaded."""
    if not store.plan_path.exists():
        raise PublishError(f"nothing to push: no plan in {store.root}")
    api = _api(token_env)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        raise PublishError(f"{repo_id} is PUBLIC; refusing to upload dialogues")
    n = len(store.accepted_dialogue_ids())
    api.upload_folder(
        folder_path=str(store.root),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=prefix,
        allow_patterns=list(DIALOGUE_PATTERNS),
        commit_message=f"{prefix}: {n} accepted dialogues",
    )
    return {"repo_id": repo_id, "prefix": prefix, "accepted_dialogues": n}


def pull_dialogues(
    store: DatasetStore, repo_id: str, prefix: str, token_env: str = "HF_TOKEN"
) -> dict:
    """Download `prefix/` from the dialogue repo into the local store. Refuses to merge
    into a store whose split manifest differs (splits are immutable)."""
    from huggingface_hub import snapshot_download

    _api(token_env)  # import check + auth
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=tmp,
            allow_patterns=[f"{prefix}/{p}" for p in DIALOGUE_PATTERNS],
            token=os.environ.get(token_env) or None,
        )
        src = Path(tmp) / prefix
        if not (src / "plan.json").exists():
            raise PublishError(f"{repo_id} has no {prefix}/plan.json")
        if store.splits_path.exists() and read_json(store.splits_path) != read_json(
            src / "splits.json"
        ):
            raise PublishError(f"{store.splits_path} differs from {repo_id}/{prefix}")
        store.root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, store.root, dirs_exist_ok=True)
    return {
        "repo_id": repo_id,
        "prefix": prefix,
        "accepted_dialogues": len(store.accepted_dialogue_ids()),
    }
