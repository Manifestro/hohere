"""The m-stream builder: variants, collisions and overflow."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import record

from hohere import units


def test_variant_a_is_all_pad():
    stream, stats = units.build_m_with_stats(record(), "A", 4096)
    assert stream.shape == (32,)
    assert stream.dtype == np.int16
    assert set(stream.tolist()) == {4097}
    assert stats.placed == stats.shifted == stats.dropped == 0


def test_variant_b_sil_only_holds_the_pauses():
    stream, stats = units.build_m_with_stats(record(), "B-sil", 4096)
    assert 4096 in stream.tolist()  # SIL = k
    assert (stream == 4096).sum() == 1
    assert stream[3] == 4096  # 12 WavLM frames // 4
    assert stats.placed == 1


def test_variant_b_places_units_and_pauses_in_time_order():
    stream, stats = units.build_m_with_stats(record(), "B", 4096)
    assert stream[:5].tolist() == [0, 1, 2, 4096, 4097]
    assert stats.placed == 4
    assert stats.shifted == 0
    assert stats.dropped == 0


def test_the_k8192_vocabulary_is_used_for_k_8192():
    stream, _ = units.build_m_with_stats(record(), "B", 8192)
    assert stream[:3].tolist() == [100, 101, 102]
    assert units.build_m(record(), "B-sil", 8192)[3] == 8192


def test_collisions_shift_to_the_next_free_frame():
    rec = record(unit_starts=(0, 1, 2, 8))
    stream, stats = units.build_m_with_stats(rec, "B", 4096)
    # the first three units all want frame 0, so they land on 0, 1, 2
    assert stream[:6].tolist() == [0, 1, 2, 3, 4096, 4097]
    assert stats.placed == 5
    assert stats.shifted == 4  # only the unit that wanted frame 0 kept it


def test_events_past_the_last_frame_are_dropped():
    rec = record(frames=2, unit_starts=(0, 8), sil_starts=())
    stream, stats = units.build_m_with_stats(rec, "B", 4096)
    assert stream.shape == (2,)
    assert stats.placed == 1
    assert stats.dropped == 1


def test_out_of_range_tokens_are_dropped():
    rec = record(unit_starts=(0, 8), sil_starts=())
    rec["unit_k4096"] = np.asarray([5, 9999], dtype=np.int16)
    stream, stats = units.build_m_with_stats(rec, "B", 4096)
    assert stats.placed == 1
    assert stats.dropped == 1
    assert 9999 not in stream.tolist()


def test_pair_frames_stacks_the_m_stream_with_codebook_zero():
    rec = record()
    pairs = units.pair_frames(rec, "B", 4096)
    assert pairs.shape == (32, 2)
    assert pairs.dtype == np.int16
    assert pairs[:, 1].tolist() == rec["codes"][0].tolist()
    assert pairs[:4, 0].tolist() == [0, 1, 2, 4096]


def test_unknown_variants_and_vocabularies_raise():
    with pytest.raises(ValueError):
        units.build_m(record(), "C", 4096)
    with pytest.raises(ValueError):
        units.build_m(record(), "B", 1024)


def test_an_empty_chunk_stays_empty():
    stream, stats = units.build_m_with_stats(record(), "B", 4096, frames=0)
    assert stream.shape == (0,)
    assert stats.dropped == 4
