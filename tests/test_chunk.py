"""Chunking on the VAD: the 30-60 s rules and the edge cases."""

from __future__ import annotations

import itertools

import pytest

from hohere.chunk import MAX_S, MIN_S, chunks, normalize, plan


def test_a_single_span_becomes_one_chunk_with_half_second_edges():
    result = plan([(10.0, 50.0)], 60.0)
    assert result.chunks == [(9.5, 50.5)]
    assert (result.dropped_long, result.dropped_tail, result.merged_tail) == (0, 0, 0)


def test_file_edges_are_clipped_to_the_recording():
    result = plan([(0.0, 40.0)], 40.0)
    assert result.chunks == [(0.0, 40.0)]


def test_tiny_recordings_are_dropped_and_counted():
    result = plan([(1.0, 3.0)], 10.0)
    assert result.chunks == []
    assert result.dropped_tail == 1
    assert plan([], 10.0).chunks == []


def test_a_span_longer_than_the_maximum_is_dropped_and_counted():
    result = plan([(0.0, 61.0), (70.0, 100.0)], 120.0)
    assert result.dropped_long == 1
    assert result.chunks == [(69.5, 100.5)]


def test_many_short_spans_give_30_to_60_second_chunks_that_never_overlap():
    spans = [(i * 6.0, i * 6.0 + 5.0) for i in range(40)]
    result = plan(spans, 240.0)
    assert len(result.chunks) > 2
    for start, end in result.chunks:
        assert MIN_S <= end - start <= MAX_S
    for (_, end), (start, _) in itertools.pairwise(result.chunks):
        assert end <= start
    # every chunk is contiguous with the next one, the gaps are exactly halved
    assert result.chunks[0][0] == 0.0


def test_the_cut_falls_in_the_widest_gap_that_fits():
    # 30 s of speech, then a 5 s gap, then 20 s more, then only a 1 s gap
    spans = [(0.0, 30.0), (35.0, 55.0), (56.0, 70.0)]
    result = plan(spans, 90.0)
    assert result.chunks[0] == (0.0, 30.5)  # the 5 s gap is the widest one that fits
    assert result.chunks[1][0] == 34.5  # half of that gap is on either side


def test_gaps_exactly_at_the_edge_limit_keep_chunks_from_overlapping():
    spans = [(0.0, 35.0), (36.0, 80.0), (81.0, 120.0)]
    result = plan(spans, 130.0)
    first, second = result.chunks[0], result.chunks[1]
    assert first[1] == 35.5  # tail = min(gap / 2, 0.5) = 0.5
    assert second[0] == 35.5  # lead = the same 0.5, so they touch but do not overlap
    assert first[1] <= second[0]


def test_a_short_tail_is_merged_into_the_previous_chunk():
    result = plan([(0.0, 30.0), (35.0, 50.0)], 55.0)
    assert result.chunks == [(0.0, 50.5)]
    assert result.merged_tail == 1
    assert result.dropped_tail == 0


def test_a_short_tail_that_would_not_fit_is_dropped_and_counted():
    result = plan([(0.0, 30.0), (35.0, 50.0), (60.0, 65.0)], 70.0)
    assert result.chunks == [(0.0, 50.5)]
    assert result.merged_tail == 0
    assert result.dropped_tail == 1


def test_normalize_clips_sorts_and_merges_overlapping_spans():
    assert normalize([(5, 8), (1, 3), (-2, 0.5), (7, 9), (20, 30)], 25) == [
        (0.0, 0.5),
        (1.0, 3.0),
        (5.0, 9.0),
        (20.0, 25.0),
    ]
    assert normalize([(2, 2), (3, 2)], 10) == []


def test_chunks_is_the_boundaries_only():
    spans = [(0.0, 40.0)]
    assert chunks(spans, 40.0) == plan(spans, 40.0).chunks


def test_kept_seconds_sums_the_chunks():
    result = plan([(10.0, 50.0)], 60.0)
    assert result.kept_seconds == pytest.approx(41.0)
