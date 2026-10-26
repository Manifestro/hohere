"""Versioned release manifests and provenance."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from hohere.config.schema import HohereConfig
from hohere.schemas.dialogue import SCHEMA_VERSION
from hohere.storage.local import DatasetStore, read_json

MANIFEST_FILES = ("channel_a.wav", "channel_b.wav", "events.json", "metadata.json")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def software_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for pkg in ("hohere", "numpy", "soundfile", "openai", "kokoro", "torch", "datasets",
                "huggingface-hub"):  # fmt: skip
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def build_manifest(store: DatasetStore, cfg: HohereConfig, include_failed: bool = False) -> dict:
    conversations = []
    hours_by_split: dict[str, float] = defaultdict(float)
    hours_by_skill: dict[str, float] = defaultdict(float)
    models: Counter[str] = Counter()
    for cid in store.conversation_ids():
        meta = store.load_metadata(cid)
        if meta.quality_status != "passed" and not include_failed:
            continue
        d = store.conversation_dir(cid)
        files = {
            name: {"sha256": sha256_file(d / name), "bytes": (d / name).stat().st_size}
            for name in MANIFEST_FILES
        }
        conversations.append(
            {
                "id": cid,
                "split": meta.split,
                "skill": meta.skill.value,
                "scenario": meta.scenario,
                "duration_samples": meta.duration_samples,
                "quality_status": meta.quality_status,
                "files": files,
            }
        )
        hours_by_split[meta.split] += meta.duration_seconds / 3600
        hours_by_skill[meta.skill.value] += meta.duration_seconds / 3600
        models[f"dialogue:{meta.generator.get('dialogue_model', '?')}"] += 1
        models[f"speech:{meta.generator.get('speech_model', '?')}"] += 1

    splits_hash = sha256_file(store.splits_path) if store.splits_path.exists() else None
    total_hours = sum(hours_by_split.values())
    return {
        "dataset": cfg.project.name,
        "version": cfg.project.dataset_version,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_fingerprint": cfg.fingerprint(),
        "seed": cfg.project.seed,
        "sample_rate": cfg.speech.sample_rate,
        "software": software_versions(),
        "models": dict(models),
        "splits_manifest_sha256": splits_hash,
        "totals": {
            "conversations": len(conversations),
            "hours": round(total_hours, 4),
            "hours_by_split": {k: round(v, 4) for k, v in sorted(hours_by_split.items())},
            "hours_by_skill": {k: round(v, 4) for k, v in sorted(hours_by_skill.items())},
            "target_hours": cfg.dataset.target_hours,
            "target_share_by_skill": {k.value: v.share for k, v in cfg.dataset.skill_mix.items()},
        },
        "conversations": conversations,
    }


def verify_manifest(root: Path, manifest: dict) -> list[str]:
    """Re-hash every file listed in a manifest. Returns a list of problems."""
    problems = []
    for conv in manifest["conversations"]:
        for name, info in conv["files"].items():
            p = root / conv["id"] / name
            if not p.exists():
                problems.append(f"missing {p}")
            elif sha256_file(p) != info["sha256"]:
                problems.append(f"checksum mismatch {p}")
    return problems


def load_manifest(store: DatasetStore) -> dict:
    return read_json(store.manifest_path)
