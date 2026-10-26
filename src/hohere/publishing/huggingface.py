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
from hohere.publishing.card_assets import write_card_assets
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
    ids = [c["id"] for c in manifest["conversations"]]
    assets = write_card_assets(store, ids, out_dir)
    card = dataset_card(cfg, manifest, shard_files, validation, assets)
    (out_dir / "README.md").write_text(card)
    return manifest


def _size_category(n: int) -> str:
    for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K")):
        if n < limit:
            return label
    return "100K<n<1M"


def _human(n: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suffix
    return f"{n:.0f}"


def _badge(label: str, value: str, color: str) -> str:
    def enc(t: str) -> str:
        return t.replace("-", "--").replace("_", "__").replace(" ", "%20")

    return (
        f'<img alt="{label}: {value}" src="https://img.shields.io/badge/'
        f'{enc(label)}-{enc(value)}-{color}?style=for-the-badge">'
    )


def _bar(frac: float, width: int = 20) -> str:
    n = max(0, min(width, round(frac * width)))
    return "█" * n + "░" * (width - n)


SKILL_LABELS = {
    "simple_dialogue": "💬 Simple dialogue",
    "context_memory": "🧠 Context & memory",
    "turn_taking": "⏳ Turn-taking & silence",
    "backchannel": "👂 Backchannels",
    "interruption": "✋ Interruptions",
    "incomplete_thoughts": "🔁 Incomplete thoughts",
}
EVENT_LABELS = {
    "normal_turn": "🗣️ Turn",
    "short_pause": "· Short pause",
    "long_pause": "⏸️ Long pause",
    "overlap": "🔀 Overlap",
    "backchannel": "👂 Backchannel",
    "interruption": "✋ Interruption",
    "cutoff": "✂️ Cutoff",
    "self_correction": "🔁 Self-correction",
}


def dataset_card(
    cfg: HohereConfig,
    manifest: dict,
    shard_files: dict[str, list[str]],
    validation: dict | None = None,
    assets: dict | None = None,
) -> str:
    t = manifest["totals"]
    pub, sc = cfg.publishing, cfg.speech
    repo = pub.repository_id or cfg.project.name
    name = repo.split("/")[-1]
    words, hours, convs = t["words"], t["hours"], t["conversations"]
    configs = "\n".join(
        f"  - split: {split}\n    path: data/{split}-*.parquet" for split in shard_files
    )
    badges = " ".join(
        [
            _badge("words", _human(words), "4C78A8"),
            _badge("audio", f"{hours:.0f} h", "F58518"),
            _badge("conversations", f"{convs:,}", "54A24B"),
            _badge("license", pub.license.upper(), "B279A2"),
            _badge("language", cfg.dataset.language.upper(), "72B7B2"),
            _badge("data", "100% synthetic", "E45756"),
        ]
    )
    listen = ""
    if assets:
        base = f"https://huggingface.co/datasets/{repo}/resolve/main"
        listen = f"""
## 🎧 Listen

<p align="center">
  <audio controls src="{base}/{assets["audio"]}"></audio><br>
  <sub>First {assets["seconds"]:.0f} s of <code>{assets["conversation_id"]}</code> ·
  🎧 headphones: <b>left = user</b>, <b>right = assistant</b></sub>
</p>

<p align="center"><img src="{assets["timeline"]}" alt="two-channel timeline" width="100%"></p>
"""
    skill_rows = "\n".join(
        f"| {SKILL_LABELS.get(k, k)} | `{_bar(t['words_by_skill'].get(k, 0) / max(words, 1))}` "
        f"| {t['words_by_skill'].get(k, 0) / max(words, 1) * 100:.0f}% "
        f"| {_human(t['words_by_skill'].get(k, 0))} | {v:.1f} h |"
        for k, v in sorted(t["hours_by_skill"].items(), key=lambda kv: -kv[1])
    )
    events = {k: v for k, v in (validation or {}).get("event_counts", {}).items()}
    top = max([v for k, v in events.items() if k != "normal_turn"] or [1])
    event_rows = "\n".join(
        f"| {EVENT_LABELS.get(k, k)} | `{_bar(min(v / top, 1.0), 16)}` | {v:,} "
        f"| {v * 1000 / max(words, 1):.1f} |"
        for k, v in sorted(events.items(), key=lambda kv: -kv[1])
        if k != "normal_turn"
    )
    events_section = (
        f"""
## ⚡ Full-duplex behaviour

| Event | | Count | per 1K words |
|:--|:--|--:|--:|
{event_rows}
"""
        if event_rows
        else ""
    )
    glance = (
        f"| **{_human(words)}** | {_human(t['words_heard_a'])} "
        f"| {_human(t['words_spoken_b'])} | {convs:,} | {hours:.0f} h |"
    )
    subtitle = (
        f"Two synchronized channels · {manifest['sample_rate'] // 1000} kHz · "
        f"sample-accurate turn-taking events · v{manifest['version']}"
    )
    split_rows = "\n".join(
        f"| **{k}** | {t['words_by_split'].get(k, 0):,} | {v:.1f} h |"
        for k, v in t["hours_by_split"].items()
    )
    return f"""---
license: {pub.license}
language:
- {cfg.dataset.language}
pretty_name: "{name}: synthetic full-duplex spoken dialogues"
size_categories:
- {_size_category(convs)}
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

<div align="center">

# 🗣️ {name}

### {words:,} words of full-duplex conversation between a user and an AI voice assistant

{badges}

<sub>{subtitle}</sub>

</div>

---

> 🧪 **Fully synthetic.** Every word was written by an LLM and every voice is TTS.
> No real person was recorded.
{listen}
## 📊 At a glance

<div align="center">

| 🔤 Words | 👂 Heard (user) | 🗣️ Spoken (assistant) | 💬 Conversations | ⏱️ Audio |
|:--:|:--:|:--:|:--:|:--:|
{glance}

</div>

## 🧩 Conversational skills

| Skill | | Share | Words | Audio |
|:--|:--|--:|--:|--:|
{skill_rows}
{events_section}
## 🚀 Quick start

```python
from datasets import load_dataset

ds = load_dataset("{repo}", split="train", streaming=True)
row = next(iter(ds))
user, assistant = row["channel_a"]["array"], row["channel_b"]["array"]
row["events"][:3]
```

<details>
<summary><b>📦 Row format</b></summary>

| Field | Content |
|:--|:--|
| `channel_a` | user audio, 16-bit mono, {manifest["sample_rate"]} Hz |
| `channel_b` | assistant audio, same length as `channel_a` |
| `events` | `event_type`, `speaker`, `start_sample`, `end_sample`, `text` |
| `words_a` / `words_b` | words heard / spoken |
| `skill`, `scenario`, `topic` | what the conversation is about |
| `voice_a` / `voice_b` | TTS voices |

Channels are never mixed: overlapping speech sits at the same sample positions in both.

</details>

<details>
<summary><b>🧪 How it was made</b></summary>

1. 🤖 **Dialogue**: `{cfg.dialogue.model}` writes structured conversations; a validator
   rejects broken, monologue-like or duplicate ones.
2. 🔊 **Voices**: Kokoro-82M. Assistant always `{sc.model_voice}`; user drawn from
   {len(sc.user_voices)} voices.
3. 🎚️ **Timeline**: pauses, overlaps, backchannels, interruptions and cutoffs placed on two
   channels with sample accuracy.
4. ✅ **QC**: every conversation checked for timing, speaker isolation, clipping, silence.

Code: [Manifestro/hohere](https://github.com/Manifestro/hohere)

</details>

<details>
<summary><b>🗂️ Splits</b></summary>

Grouped by scenario family, fixed before synthesis.

| Split | Words | Audio |
|:--|--:|--:|
{split_rows}

</details>

<details>
<summary><b>⚠️ Limitations</b></summary>

- Synthetic voices, clean audio, no background noise.
- English only.
- Cut-off utterances keep their full `text`; the audio stops early.

</details>

## 📜 License

**{pub.license.upper()}** · © {pub.attribution} · Built with Qwen3 (Apache-2.0) and Kokoro-82M
(Apache-2.0).

```bibtex
@misc{{{name.replace("-", "_")}_{manifest["version"].replace(".", "_")},
  title  = {{{name}: synthetic full-duplex spoken dialogues}},
  author = {{{pub.attribution}}},
  year   = {{2026}},
  note   = {{Version {manifest["version"]}}},
  url    = {{https://huggingface.co/datasets/{repo}}}
}}
```

<div align="center"><sub>Made with ❤️ by {pub.attribution}</sub></div>
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
