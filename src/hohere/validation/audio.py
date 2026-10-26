"""Audio-level quality checks on a composed two-channel conversation."""

from __future__ import annotations

import itertools

import numpy as np

from hohere.config.schema import HohereConfig
from hohere.schemas.dataset import SPEECH_EVENTS, Event, EventType
from hohere.schemas.dialogue import RejectionReason


def _db(x: float) -> float:
    return 20 * np.log10(max(x, 1e-10))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def validate_audio(
    channel_a: np.ndarray,
    channel_b: np.ndarray,
    sample_rate: int,
    events: list[Event],
    cfg: HohereConfig,
) -> list[RejectionReason]:
    q = cfg.quality
    out: list[RejectionReason] = []

    def bad(code: str, message: str, ref: str | None = None) -> None:
        out.append(RejectionReason(code=code, message=message, turn_id=ref))

    if sample_rate != cfg.speech.sample_rate:
        bad("sample_rate", f"{sample_rate} != {cfg.speech.sample_rate}")
    if channel_a.ndim != 1 or channel_b.ndim != 1:
        bad("not_mono", "channels must be 1-D mono arrays")
        return out
    if channel_a.shape[0] != channel_b.shape[0]:
        bad("length_mismatch", f"{channel_a.shape[0]} vs {channel_b.shape[0]} samples")
        return out
    n = channel_a.shape[0]
    channels = {"A": channel_a, "B": channel_b}

    for name, ch in channels.items():
        if not np.all(np.isfinite(ch)):
            bad("non_finite", f"channel {name} has NaN/inf samples")
            return out
        if np.max(np.abs(ch), initial=0.0) > 1.0:
            bad("invalid_amplitude", f"channel {name} exceeds full scale")
        clipped = float(np.mean(np.abs(ch) >= q.clipping_threshold)) if n else 0.0
        if clipped > q.max_clipped_fraction:
            bad("clipping", f"channel {name}: {clipped:.4%} samples clipped")

    duration = n / sample_rate
    dc = cfg.dialogue
    if not dc.min_duration_seconds <= duration <= dc.max_duration_seconds:
        bad(
            "duration",
            f"{duration:.1f}s outside [{dc.min_duration_seconds}, {dc.max_duration_seconds}]",
        )

    out += check_event_consistency(events, n)

    speech = [e for e in events if e.event_type in SPEECH_EVENTS]
    masks = {s: np.zeros(n, dtype=bool) for s in channels}
    for e in speech:
        if e.speaker is None or not 0 <= e.start_sample < e.end_sample <= n:
            continue
        seg = channels[e.speaker][e.start_sample : e.end_sample]
        masks[e.speaker][e.start_sample : e.end_sample] = True
        if _db(_rms(seg)) < q.silence_rms_db:
            bad("silent_utterance", f"rms {_db(_rms(seg)):.1f} dB", e.utterance_id)
        edge = max(abs(float(seg[0])), abs(float(seg[-1])))
        if edge > q.edge_amplitude_max:
            bad("abrupt_edge", f"edge amplitude {edge:.3f}", e.utterance_id)

    cut_ids = {e.related_event_id for e in events if e.event_type is EventType.CUTOFF}
    tail = max(1, sample_rate // 200)  # 5 ms
    for e in speech:
        if e.event_id in cut_ids or e.speaker is None or e.end_sample - e.start_sample <= tail:
            continue
        seg = channels[e.speaker][e.end_sample - tail : e.end_sample]
        if np.max(np.abs(seg)) > 4 * q.edge_amplitude_max:
            bad("truncated_segment", "utterance ends at high amplitude", e.utterance_id)

    for name, ch in channels.items():
        outside = ch[~masks[name]]
        if outside.size and _db(_rms(outside)) > q.leakage_rms_db:
            bad("speaker_leakage", f"channel {name} has audio outside its own utterances")
    return out


def check_event_consistency(events: list[Event], num_samples: int) -> list[RejectionReason]:
    """Temporal relationships between events; independent of the audio content."""
    out: list[RejectionReason] = []
    by_id = {e.event_id: e for e in events}
    if len(by_id) != len(events):
        out.append(RejectionReason(code="duplicate_event_id", message="event ids not unique"))

    for e in events:
        if not 0 <= e.start_sample < e.end_sample <= num_samples:
            out.append(
                RejectionReason(
                    code="event_bounds",
                    message=f"[{e.start_sample}, {e.end_sample}) outside [0, {num_samples}]",
                    turn_id=e.event_id,
                )
            )
        if e.related_event_id and e.related_event_id not in by_id:
            out.append(
                RejectionReason(
                    code="dangling_relation", message="related event missing", turn_id=e.event_id
                )
            )

    for s in ("A", "B"):
        spans = sorted(
            (e.start_sample, e.end_sample, e.event_id)
            for e in events
            if e.speaker == s and e.event_type in SPEECH_EVENTS
        )
        for (_, end1, _), (start2, _, id2) in itertools.pairwise(spans):
            if start2 < end1:
                out.append(
                    RejectionReason(
                        code="same_speaker_overlap",
                        message=f"speaker {s} overlaps itself",
                        turn_id=id2,
                    )
                )

    for e in events:
        rel = by_id.get(e.related_event_id) if e.related_event_id else None
        if rel is None:
            continue
        if e.event_type in (EventType.INTERRUPTION, EventType.OVERLAP) and not (
            rel.speaker != e.speaker and rel.start_sample < e.start_sample < rel.end_sample
        ):
            out.append(
                RejectionReason(
                    code=f"{e.event_type.value}_timing",
                    message="must start inside the other speaker's utterance",
                    turn_id=e.event_id,
                )
            )
        if e.event_type is EventType.INTERRUPTION and not any(
            c.event_type is EventType.CUTOFF and c.related_event_id == rel.event_id for c in events
        ):
            out.append(
                RejectionReason(
                    code="interruption_without_cutoff",
                    message="interrupted utterance is not cut off",
                    turn_id=e.event_id,
                )
            )
        if e.event_type is EventType.CUTOFF and not (
            rel.start_sample <= e.start_sample and e.end_sample == rel.end_sample
        ):
            out.append(
                RejectionReason(
                    code="cutoff_timing",
                    message="cutoff must end its utterance",
                    turn_id=e.event_id,
                )
            )
    return out
