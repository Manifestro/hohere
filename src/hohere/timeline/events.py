"""Build the sample-accurate event list from final placements."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hohere.schemas.dataset import Event, EventType

if TYPE_CHECKING:
    from hohere.timeline.composer import Placement

_ORDER = {t: i for i, t in enumerate(EventType)}


def silence_gaps(placements: list[Placement], min_len: int) -> list[tuple[int, int]]:
    """Intervals where *both* channels are silent, between the first and last speech."""
    spans = sorted((p.start, p.end) for p in placements)
    gaps: list[tuple[int, int]] = []
    if not spans:
        return gaps
    cur_end = spans[0][1]
    for start, end in spans[1:]:
        if start - cur_end >= min_len:
            gaps.append((cur_end, start))
        cur_end = max(cur_end, end)
    return gaps


def annotate(
    conversation_id: str,
    placements: list[Placement],
    sample_rate: int,
    min_pause: int,
    long_pause: int,
) -> list[Event]:
    """Speech events (one per utterance), cutoff events and pause events.

    Speech events cover exactly the audio placed in the channel. Cutoff events cover
    the fade-out where an utterance was stopped early. Pauses are global silences.
    """
    raw: list[dict] = []
    for p in placements:
        raw.append(
            {
                "key": ("speech", p.turn.id),
                "speaker": p.speaker,
                "utterance_id": p.turn.id,
                "event_type": p.event_type,
                "start_sample": p.start,
                "end_sample": p.end,
                "related": ("speech", p.related_turn) if p.related_turn else None,
                "cause": None,
                "text": p.turn.text,
            }
        )
        if p.cut:
            raw.append(
                {
                    "key": ("cutoff", p.turn.id),
                    "speaker": p.speaker,
                    "utterance_id": p.turn.id,
                    "event_type": EventType.CUTOFF,
                    "start_sample": p.end - p.fade_out,
                    "end_sample": p.end,
                    "related": ("speech", p.turn.id),
                    "cause": ("speech", p.cut_cause) if p.cut_cause else None,
                    "text": None,
                }
            )
    for i, (start, end) in enumerate(silence_gaps(placements, min_pause)):
        raw.append(
            {
                "key": ("pause", str(i)),
                "speaker": None,
                "utterance_id": None,
                "event_type": EventType.LONG_PAUSE
                if end - start >= long_pause
                else EventType.SHORT_PAUSE,
                "start_sample": start,
                "end_sample": end,
                "related": None,
                "cause": None,
                "text": None,
            }
        )

    raw.sort(key=lambda r: (r["start_sample"], _ORDER[r["event_type"]], r["key"]))
    ids = {r["key"]: f"{conversation_id}_e{i:04d}" for i, r in enumerate(raw, start=1)}
    return [
        Event(
            event_id=ids[r["key"]],
            speaker=r["speaker"],
            utterance_id=r["utterance_id"],
            event_type=r["event_type"],
            start_sample=r["start_sample"],
            end_sample=r["end_sample"],
            related_event_id=ids.get(r["related"]) if r["related"] else None,
            cause_event_id=ids.get(r["cause"]) if r["cause"] else None,
            text=r["text"],
        )
        for r in raw
    ]
