"""Frame-level labels for codec-token models: speech activity per channel and turn events.

Frames follow the Mimi codec: 24 kHz audio, 12.5 frames/s, so one frame = 1920 samples.
Frame k covers samples [k * 1920, (k + 1) * 1920). A frame is active for a speaker when
at least half of it is covered by that speaker's speech events. Event frames are
`start_sample // 1920` (first frame touched) and `ceil(end_sample / 1920)` (exclusive).

Event table rows (all frames are frame indices, `gap_ms` may be negative = overlap):

    turn_end        a floor-holding turn ends; `next_speaker` is who holds the floor next
    response_onset  B starts a floor-holding turn right after an A turn (A -> B transition)
    user_onset      A starts a floor-holding turn right after a B turn (B -> A transition)
    interruption    `speaker` cuts into `other`; `yield_ms` = how long `other` kept talking
    backchannel     short reaction by `speaker` while `other` holds the floor (or in a gap)
    cutoff          `speaker` stops mid-utterance (fade-out region)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

import numpy as np

from hohere.schemas.dataset import Event, EventType

FRAME_SAMPLES = 1920
# Speech that holds the floor; backchannels do not.
FLOOR_EVENTS = {
    EventType.NORMAL_TURN,
    EventType.OVERLAP,
    EventType.INTERRUPTION,
    EventType.SELF_CORRECTION,
}
SPEECH = FLOOR_EVENTS | {EventType.BACKCHANNEL}


def num_frames(num_samples: int, frame: int = FRAME_SAMPLES) -> int:
    return -(-num_samples // frame)


def frame_activity(
    events: Iterable[Event], num_samples: int, frame: int = FRAME_SAMPLES
) -> dict[str, np.ndarray]:
    """uint8 activity per frame for A and B."""
    n = num_frames(num_samples, frame)
    covered = {s: np.zeros(n * frame, np.bool_) for s in ("A", "B")}
    for e in events:
        if e.event_type in SPEECH and e.speaker is not None:
            covered[e.speaker][e.start_sample : e.end_sample] = True
    return {
        s: (c.reshape(n, frame).sum(axis=1) * 2 >= frame).astype(np.uint8)
        for s, c in covered.items()
    }


def _f0(sample: int, frame: int) -> int:
    return sample // frame


def _f1(sample: int, frame: int) -> int:
    return -(-sample // frame)


def event_table(
    events: list[Event], sample_rate: int = 24000, frame: int = FRAME_SAMPLES
) -> list[dict]:
    """Turn-transition events with frame indices, sorted by start frame."""
    ms = 1000 / sample_rate
    by_id = {e.event_id: e for e in events}
    floor = sorted(
        (e for e in events if e.event_type in FLOOR_EVENTS and e.speaker),
        key=lambda e: (e.start_sample, e.event_id),
    )
    rows: list[dict] = []

    def add(kind: str, e: Event, **extra) -> None:
        rows.append(
            {
                "type": kind,
                "speaker": e.speaker,
                "start_frame": _f0(e.start_sample, frame),
                "end_frame": _f1(e.end_sample, frame),
                "start_sample": e.start_sample,
                "end_sample": e.end_sample,
                "event_id": e.event_id,
                **extra,
            }
        )

    for prev, nxt in zip(floor, floor[1:], strict=False):
        gap = round((nxt.start_sample - prev.end_sample) * ms)
        add("turn_end", prev, next_speaker=nxt.speaker, gap_ms=gap)
    if floor:
        add("turn_end", floor[-1], next_speaker=None, gap_ms=None)

    for prev, nxt in zip(floor, floor[1:], strict=False):
        if prev.speaker == nxt.speaker or nxt.event_type is EventType.INTERRUPTION:
            continue
        kind = "response_onset" if nxt.speaker == "B" else "user_onset"
        add(kind, nxt, other=prev.speaker, gap_ms=round((nxt.start_sample - prev.end_sample) * ms))

    for e in events:
        if e.event_type is EventType.INTERRUPTION and e.speaker:
            host = by_id.get(e.related_event_id or "")
            yield_ms = round((host.end_sample - e.start_sample) * ms) if host else None
            add("interruption", e, other=host.speaker if host else None, yield_ms=yield_ms)
        elif e.event_type is EventType.BACKCHANNEL and e.speaker:
            host = by_id.get(e.related_event_id or "")
            inside = bool(host and host.start_sample <= e.start_sample < host.end_sample)
            add("backchannel", e, other=host.speaker if host else None, inside_host=inside)
        elif e.event_type is EventType.CUTOFF and e.speaker:
            cause = by_id.get(e.cause_event_id or "")
            add("cutoff", e, cause_speaker=cause.speaker if cause else None)

    rows.sort(key=lambda r: (r["start_frame"], r["type"], r["event_id"]))
    return rows


def _pct(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, float)
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 1),
        **{f"p{q}": round(float(np.percentile(a, q)), 1) for q in (5, 25, 50, 75, 95)},
        "share_negative": round(float((a < 0).mean()), 3),
    }


def transition_stats(tables: Iterable[tuple[str, list[dict]]]) -> dict:
    """Corpus statistics over (skill, event_table) pairs."""
    counts: Counter[str] = Counter()
    by_skill: dict[str, Counter[str]] = defaultdict(Counter)
    gaps: dict[str, list[float]] = defaultdict(list)
    yields: dict[str, list[float]] = defaultdict(list)
    conversations = 0
    for skill, rows in tables:
        conversations += 1
        for r in rows:
            key = r["type"]
            if r["type"] in ("backchannel", "interruption", "cutoff"):
                key = f"{r['type']}_{r['speaker']}"
            counts[key] += 1
            by_skill[skill][key] += 1
            if r["type"] in ("response_onset", "user_onset"):
                gaps[r["type"]].append(r["gap_ms"])
            if r["type"] == "interruption" and r.get("yield_ms") is not None:
                yields[f"{r['other']}_yields_to_{r['speaker']}"].append(r["yield_ms"])
    bc_a, bc_b = counts["backchannel_A"], counts["backchannel_B"]
    int_a, int_b = counts["interruption_A"], counts["interruption_B"]
    return {
        "conversations": conversations,
        "counts": dict(sorted(counts.items())),
        "backchannel_share_assistant": round(bc_b / (bc_a + bc_b), 3) if bc_a + bc_b else None,
        "interruption_share_user_interrupts": (
            round(int_a / (int_a + int_b), 3) if int_a + int_b else None
        ),
        "gap_ms": {k: _pct(v) for k, v in sorted(gaps.items())},
        "yield_ms": {k: _pct(v) for k, v in sorted(yields.items())},
        "by_skill": {s: dict(sorted(c.items())) for s, c in sorted(by_skill.items())},
    }


# Columns of the labels companion file (one row per conversation). Event rows share one
# struct; fields that do not apply to a row type are null.
EVENT_FIELDS = {
    "type": "string",
    "speaker": "string",
    "other": "string",
    "next_speaker": "string",
    "cause_speaker": "string",
    "start_frame": "int32",
    "end_frame": "int32",
    "start_sample": "int64",
    "end_sample": "int64",
    "gap_ms": "int32",
    "yield_ms": "int32",
    "inside_host": "bool",
    "event_id": "string",
}


def labels_schema():
    import pyarrow as pa

    event = pa.struct([(k, getattr(pa, v)()) for k, v in EVENT_FIELDS.items()])
    return pa.schema(
        [
            ("id", pa.string()),
            ("skill", pa.string()),
            ("sample_rate", pa.int32()),
            ("frame_samples", pa.int32()),
            ("num_frames", pa.int32()),
            ("activity_a", pa.list_(pa.uint8())),
            ("activity_b", pa.list_(pa.uint8())),
            ("events", pa.list_(event)),
        ]
    )


def labels_row(
    cid: str, skill: str, events: list[Event], num_samples: int, sample_rate: int
) -> dict:
    act = frame_activity(events, num_samples)
    table = event_table(events, sample_rate)
    return {
        "id": cid,
        "skill": skill,
        "sample_rate": sample_rate,
        "frame_samples": FRAME_SAMPLES,
        "num_frames": num_frames(num_samples),
        "activity_a": act["A"].tolist(),
        "activity_b": act["B"].tolist(),
        "events": [{k: r.get(k) for k in EVENT_FIELDS} for r in table],
    }


def write_labels(rows: list[dict], out_dir, source: dict) -> dict:
    """labels/labels-00000-of-00001.parquet + labels/stats.json; returns the stats."""
    import json
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["id"])
    pq.write_table(
        pa.Table.from_pylist(rows, schema=labels_schema()),
        out / "labels-00000-of-00001.parquet",
    )
    stats = {
        "source": source,
        "frame_samples": FRAME_SAMPLES,
        **transition_stats((r["skill"], r["events"]) for r in rows),
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def store_labels(store, ids: list[str] | None = None) -> list[dict]:
    """Label rows for conversations of a local DatasetStore (default: all composed)."""
    rows = []
    for cid in ids if ids is not None else store.conversation_ids():
        meta = store.load_metadata(cid)
        rows.append(
            labels_row(
                cid,
                meta.skill.value,
                store.load_events(cid).events,
                meta.duration_samples,
                meta.sample_rate,
            )
        )
    return rows
