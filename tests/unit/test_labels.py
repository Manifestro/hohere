from __future__ import annotations

from hohere.labels import (
    FRAME_SAMPLES,
    event_table,
    frame_activity,
    labels_row,
    num_frames,
    transition_stats,
)
from hohere.schemas.dataset import Event, EventType

F = FRAME_SAMPLES


def ev(eid, speaker, etype, start, end, related=None, cause=None):
    return Event(
        event_id=eid,
        speaker=speaker,
        utterance_id=eid if speaker else None,
        event_type=etype,
        start_sample=start,
        end_sample=end,
        related_event_id=related,
        cause_event_id=cause,
        text="x" if speaker and etype is not EventType.CUTOFF else None,
    )


def conversation():
    # A asks (0-10F), B answers after 3 frames (13F-30F) while A backchannels (18F-20F),
    # A interrupts B at 40F, B keeps talking until 43F and is cut off.
    return [
        ev("a1", "A", EventType.NORMAL_TURN, 0, 10 * F),
        ev("p1", None, EventType.SHORT_PAUSE, 10 * F, 13 * F),
        ev("b1", "B", EventType.NORMAL_TURN, 13 * F, 30 * F),
        ev("a2", "A", EventType.BACKCHANNEL, 18 * F, 20 * F, related="b1"),
        ev("b2", "B", EventType.NORMAL_TURN, 32 * F, 43 * F),
        ev("a3", "A", EventType.INTERRUPTION, 40 * F, 50 * F, related="b2"),
        ev("c1", "B", EventType.CUTOFF, 42 * F, 43 * F, related="b2", cause="a3"),
    ]


def test_frame_activity_half_frame_rule():
    events = [ev("a", "A", EventType.NORMAL_TURN, F // 2, 2 * F + F // 2 - 1)]
    act = frame_activity(events, 4 * F)
    # frame 0: exactly half covered -> active; frame 2: one sample short of half -> silent
    assert act["A"].tolist() == [1, 1, 0, 0]
    assert act["B"].tolist() == [0, 0, 0, 0]
    assert num_frames(4 * F + 1) == 5


def test_activity_counts_backchannels_and_ignores_pauses():
    act = frame_activity(conversation(), 52 * F)
    assert act["A"][18:20].tolist() == [1, 1]
    assert act["A"][10:13].tolist() == [0, 0, 0]
    assert act["B"][13:30].all() and not act["B"][30:32].any()


def test_event_table_transitions():
    rows = event_table(conversation())
    kinds = [(r["type"], r["speaker"]) for r in rows]
    assert ("response_onset", "B") in kinds
    onset = next(r for r in rows if r["type"] == "response_onset" and r["event_id"] == "b1")
    assert onset["start_frame"] == 13 and onset["gap_ms"] == 240
    # the interrupting turn is an interruption, not a user_onset
    assert not any(r["type"] == "user_onset" and r["event_id"] == "a3" for r in rows)
    intr = next(r for r in rows if r["type"] == "interruption")
    assert intr["speaker"] == "A" and intr["other"] == "B"
    assert intr["yield_ms"] == round(3 * F * 1000 / 24000)
    bc = next(r for r in rows if r["type"] == "backchannel")
    assert bc["speaker"] == "A" and bc["other"] == "B" and bc["inside_host"]
    cut = next(r for r in rows if r["type"] == "cutoff")
    assert cut["speaker"] == "B" and cut["cause_speaker"] == "A"
    ends = [r for r in rows if r["type"] == "turn_end"]
    assert ends[0]["next_speaker"] == "B" and ends[-1]["next_speaker"] is None


def test_stats_and_row_schema():
    row = labels_row("c1", "interruption", conversation(), 52 * F, 24000)
    assert row["num_frames"] == 52 == len(row["activity_a"]) == len(row["activity_b"])
    stats = transition_stats([(row["skill"], row["events"])])
    assert stats["counts"]["backchannel_A"] == 1
    assert stats["backchannel_share_assistant"] == 0.0
    assert stats["interruption_share_user_interrupts"] == 1.0
    assert stats["gap_ms"]["response_onset"]["p50"] == 240.0
    assert stats["yield_ms"]["B_yields_to_A"]["n"] == 1
