"""Dataset-level structural validation and quality reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime

import soundfile as sf
from pydantic import ValidationError

from hohere.config.schema import HohereConfig
from hohere.schemas.dialogue import RejectionReason
from hohere.storage.local import DatasetStore, read_json, write_json
from hohere.validation.audio import check_event_consistency, validate_audio


def validate_structure(store: DatasetStore, cid: str, cfg: HohereConfig) -> list[RejectionReason]:
    """File-level checks: decodable PCM WAVs, equal lengths, metadata and events in bounds."""
    out: list[RejectionReason] = []

    def bad(code: str, message: str) -> None:
        out.append(RejectionReason(code=code, message=message))

    d = store.conversation_dir(cid)
    infos = {}
    for name in ("channel_a.wav", "channel_b.wav"):
        p = d / name
        if not p.exists():
            bad("missing_file", name)
            continue
        try:
            infos[name] = sf.info(p)
        except Exception as e:
            bad("undecodable_wav", f"{name}: {e}")
    for name in ("metadata.json", "events.json"):
        if not (d / name).exists():
            bad("missing_file", name)
    if out:
        return out

    for name, info in infos.items():
        if info.samplerate != cfg.speech.sample_rate:
            bad("sample_rate", f"{name}: {info.samplerate}")
        if info.channels != 1:
            bad("not_mono", f"{name}: {info.channels} channels")
        if info.subtype != "PCM_16":
            bad("pcm_format", f"{name}: {info.subtype}")
    frames = {info.frames for info in infos.values()}
    if len(frames) != 1:
        bad("length_mismatch", f"channel frame counts differ: {sorted(frames)}")

    try:
        meta = store.load_metadata(cid)
        events = store.load_events(cid)
    except (ValidationError, ValueError) as e:
        bad("invalid_metadata", str(e)[:300])
        return out
    n = next(iter(frames))
    if meta.id != cid or events.conversation_id != cid:
        bad("id_mismatch", "metadata/events id does not match directory")
    if meta.duration_samples != n:
        bad("duration_mismatch", f"metadata says {meta.duration_samples}, files have {n}")
    if abs(meta.duration_seconds - n / meta.sample_rate) > 1e-3:
        bad("duration_mismatch", "duration_seconds inconsistent with duration_samples")
    if set(meta.speakers) != {"A", "B"}:
        bad("speakers", "metadata must map speakers A and B")
    out += check_event_consistency(events.events, n)
    return out


def validate_dataset(
    store: DatasetStore, cfg: HohereConfig, audio: bool = True, update_status: bool = True
) -> dict:
    """Validate every conversation on disk, write reports/validation.json, return the report."""
    results: dict[str, dict] = {}
    seen_ids: Counter[str] = Counter()
    hours: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    words: Counter[str] = Counter()
    words_by_split: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for cid in store.conversation_ids():
        reasons = validate_structure(store, cid, cfg)
        meta = None
        if not reasons:
            meta = store.load_metadata(cid)
            seen_ids[meta.id] += 1
            events = store.load_events(cid).events
            if audio and cfg.quality.validate_audio:
                a, b, sr = store.load_channels(cid)
                reasons += validate_audio(a, b, sr, events, cfg)
            event_counts.update(e.event_type.value for e in events)
        status = "passed" if not reasons else "failed"
        reason_counts.update(r.code for r in reasons)
        results[cid] = {"status": status, "reasons": [r.model_dump() for r in reasons]}
        if meta is not None:
            if status == "passed":
                hours[meta.split][meta.skill.value] += meta.duration_seconds / 3600
                words["A"] += meta.words_a or 0
                words["B"] += meta.words_b or 0
                words_by_split[meta.split] += (meta.words_a or 0) + (meta.words_b or 0)
            if update_status and meta.quality_status != status:
                meta.quality_status = status
                q_path = store.conversation_dir(cid) / "quality.json"
                quality = read_json(q_path) if q_path.exists() else {}
                quality["dataset_validation"] = results[cid]
                write_json(q_path, quality)
                write_json(
                    store.conversation_dir(cid) / "metadata.json", meta.model_dump(mode="json")
                )

    for cid, count in seen_ids.items():
        if count > 1:
            results[cid]["status"] = "failed"
            results[cid]["reasons"].append({"code": "duplicate_id", "message": f"{count} copies"})

    total_hours = sum(sum(v.values()) for v in hours.values())
    by_skill: Counter[str] = Counter()
    for split_hours in hours.values():
        by_skill.update(split_hours)
    distribution = {
        skill.value: {
            "hours": round(by_skill.get(skill.value, 0.0), 4),
            "share": round(by_skill.get(skill.value, 0.0) / total_hours, 4) if total_hours else 0,
            "target_share": entry.share,
        }
        for skill, entry in cfg.dataset.skill_mix.items()
    }
    rejections = store.all_rejections()
    rejection_codes: Counter[str] = Counter()
    for r in rejections:
        rejection_codes.update(reason["code"] for reason in r.get("reasons", []))

    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config_fingerprint": cfg.fingerprint(),
        "summary": {
            "conversations": len(results),
            "passed": sum(r["status"] == "passed" for r in results.values()),
            "failed": sum(r["status"] == "failed" for r in results.values()),
            "passed_hours": round(total_hours, 4),
            "passed_words": words["A"] + words["B"],
            "words_heard_a": words["A"],
            "words_spoken_b": words["B"],
            "target_words": cfg.dataset.target_words,
            "words_by_split": dict(sorted(words_by_split.items())),
            "target_hours": cfg.dataset.target_hours,
            "hours_by_split": {s: round(sum(v.values()), 4) for s, v in sorted(hours.items())},
            "skill_distribution": distribution,
            "event_counts": dict(event_counts),
            "failure_reasons": dict(reason_counts),
            "upstream_rejections": len(rejections),
            "upstream_rejection_reasons": dict(rejection_codes),
        },
        "conversations": results,
    }
    write_json(store.reports_dir / "validation.json", report)
    return report
