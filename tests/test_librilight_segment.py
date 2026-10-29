from __future__ import annotations

import random

import pytest

from hohere.librilight.segment import (
    SegmentConfig,
    merge_intervals,
    runs,
    segment,
    voiced_fraction,
)

CFG = SegmentConfig()
EPS = 1e-9
VAD = list[tuple[float, float]]


def spans(segments) -> list[tuple[float, float]]:
    return [(s.start, s.end) for s in segments]


def test_rule1_intervals_are_clipped_and_merged() -> None:
    vad = [(-1.0, 2.0), (2.0, 5.0), (5.5, 7.0), (7.2, 7.3), (30.0, 40.0)]
    assert merge_intervals(vad, 6.0) == [(0.0, 5.0), (5.5, 6.0)]


def test_rule2_runs_break_only_on_gaps_of_cut_gap() -> None:
    assert runs([(0.0, 2.0), (2.29, 4.0)], CFG.cut_gap) == [(0.0, 4.0)]
    assert runs([(0.0, 2.0), (2.31, 4.0)], CFG.cut_gap) == [(0.0, 2.0), (2.31, 4.0)]


def test_rule3_segment_closes_as_soon_as_it_reaches_min_len() -> None:
    vad = [(s, s + 2.5) for s in (2, 5.5, 9, 12.5, 16.5, 20, 23.5, 27, 30.5, 34, 37.5, 41, 44.5,
                                  48, 51.5, 55)]  # fmt: skip
    assert spans(segment(vad, 60.0)) == [
        (1.5, 8.5),
        (8.5, 15.5),
        (16.0, 23.0),
        (23.0, 30.0),
        (30.0, 37.0),
        (37.0, 44.0),
        (44.0, 51.0),
        (51.0, 58.0),
    ]


def test_rule3_edges_stop_at_half_of_the_neighbouring_gap() -> None:
    vad = [(1.0, 5.0), (5.4, 12.0)]
    assert spans(segment(vad, 20.0)) == [(0.5, 12.5)]


def test_rule3_a_segment_never_exceeds_max_len() -> None:
    # adding the second run would span 31 s, so the first one is dropped (too short)
    vad = [(10.0, 12.0), (13.0, 26.0), (26.25, 40.0)]
    assert spans(segment(vad, 60.0)) == [(12.5, 40.5)]


def test_rule4_a_run_longer_than_max_len_is_dropped() -> None:
    vad = [(1.0, 4.0), (4.5, 9.0), (20.0, 60.0), (65.0, 68.0), (68.5, 73.0)]
    assert spans(segment(vad, 100.0)) == [(0.5, 9.5), (64.5, 73.5)]


def test_rule5_short_tail_is_merged_into_the_previous_segment() -> None:
    vad = [(1.0, 4.0), (4.5, 9.0), (12.0, 14.0)]
    assert spans(segment(vad, 60.0)) == [(0.5, 14.5)]


def test_rule5_short_tail_is_dropped_when_merging_breaks_a_limit() -> None:
    too_long = [(1.0, 4.0), (4.5, 9.0), (20.0, 23.0), (23.5, 28.0), (55.0, 57.0), (57.5, 59.0)]
    assert spans(segment(too_long, 80.0)) == [(0.5, 9.5), (19.5, 28.5)]
    not_adjacent = [(1.0, 4.0), (4.5, 9.0), (20.0, 60.0), (65.0, 66.0), (66.5, 68.0)]
    assert spans(segment(not_adjacent, 80.0)) == [(0.5, 9.5)]


def test_rule6_a_segment_needs_a_pause_away_from_its_edges() -> None:
    assert segment([(1.0, 20.0)], 40.0) == []  # continuous speech, no internal pause
    too_early = [(1.0, 2.0), (2.5, 20.0)]  # the gap starts 1.5 s in, min_a is 3
    assert segment(too_early, 30.0) == []


def test_segments_are_empty_for_degenerate_input() -> None:
    assert segment([], 10.0) == []
    assert segment([(0.0, 1.0)], 0.0) == []
    assert segment([(5.0, 4.0)], 10.0) == []


def test_voiced_fraction_counts_merged_speech_only() -> None:
    assert voiced_fraction([(1.0, 3.0), (2.0, 4.0), (10.0, 11.0)], 20.0) == pytest.approx(0.2)
    assert voiced_fraction([(1.0, 3.0)], 0.0) == 0.0


def random_vad(rng: random.Random, duration: float) -> VAD:
    vad: VAD = []
    t = 0.0
    while t < duration and len(vad) < 12:
        t = round(t + rng.uniform(0.05, 3.0), 2)
        if t >= duration:
            break
        end = round(min(t + rng.uniform(0.1, 8.0), duration), 2)
        vad.append((t, end))
        t = end
    return vad


def test_segment_invariants_on_random_vads() -> None:
    for seed in range(6):
        rng = random.Random(seed)
        duration = round(rng.uniform(10.0, 200.0), 2)
        vad = random_vad(rng, duration)
        intervals = merge_intervals(vad, duration)
        previous_end = 0.0
        for s in segment(vad, duration):
            assert CFG.min_len - EPS <= s.duration <= CFG.max_len + EPS
            assert s.start >= -EPS and s.end <= duration + EPS
            assert s.start >= previous_end - EPS
            previous_end = s.end
            for a, b in intervals:
                assert not a + EPS < s.start < b - EPS
                assert not a + EPS < s.end < b - EPS
            inside = [(a, b) for a, b in intervals if b > s.start and a < s.end]
            assert inside
            assert inside[0][0] - s.start <= CFG.edge_pad + EPS
            assert s.end - inside[-1][1] <= CFG.edge_pad + EPS
