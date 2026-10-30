"""Chunking invariants over random VAD spans (the review's property test)."""

from __future__ import annotations

import itertools
import random

from hohere.chunk import MAX_S, MIN_S, plan


def random_spans(rng: random.Random, duration: float) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    cursor = rng.uniform(0.0, 2.0)
    while cursor < duration:
        length = rng.choice([0.2, 1.0, 3.0, 7.5, 12.0, 25.0, 40.0, 65.0])
        gap = rng.choice([0.05, 0.15, 0.25, 0.5, 1.0, 2.0, 5.0])
        spans.append((cursor, min(cursor + length, duration)))
        cursor += length + gap
    return spans


def in_a_gap(spans, duration, value) -> bool:
    """True when `value` (seconds) falls in a silence between two spans."""
    edges = [0.0]
    for start, end in spans:
        edges.extend([start, end])
    edges.append(duration)
    gaps = [(edges[i], edges[i + 1]) for i in range(0, len(edges) - 1, 2)]
    return any(low - 1e-9 <= value <= high + 1e-9 for low, high in gaps)


def test_every_chunk_is_between_30_and_60_seconds_in_order_and_disjoint():
    rng = random.Random(1234)
    for _ in range(200):
        duration = rng.uniform(10.0, 900.0)
        vad = random_spans(rng, duration)
        result = plan(vad, duration)
        for start, end in result.chunks:
            assert MIN_S - 1e-6 <= end - start <= MAX_S + 1e-6, (start, end, vad)
            assert 0.0 <= start < end <= duration + 1e-6
        for (_s1, end1), (start2, _e2) in itertools.pairwise(result.chunks):
            assert end1 <= start2 + 1e-9, result.chunks
        for start, end in result.chunks:
            assert in_a_gap(vad, duration, start), (start, vad)
            assert in_a_gap(vad, duration, end), (end, vad)
        assert result.kept_seconds <= duration + 1e-6


def test_a_span_that_overshoots_the_maximum_drops_the_leading_spans():
    # 25 s of speech, then a single 40 s span: 25 + 40 > 60, so the leading spans go
    result = plan([(0.0, 25.0), (25.5, 65.5)], 70.0)
    assert result.dropped_lead == 1
    assert len(result.chunks) == 1
    start, end = result.chunks[0]
    assert 30.0 <= end - start <= 60.0
    assert start >= 25.0  # the chunk starts inside the 40 s span, not before it


def test_an_overshoot_never_emits_a_chunk_over_sixty_seconds():
    result = plan([(0.0, 25.0), (25.0, 65.0)], 70.0)
    assert all(end - start <= 60.0 for start, end in result.chunks)


def test_dropped_lead_is_zero_when_nothing_overshoots():
    result = plan([(0.0, 40.0), (45.0, 90.0)], 100.0)
    assert result.dropped_lead == 0
