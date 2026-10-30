"""Silero helpers: the WavLM speech mask and the internal pauses."""

from __future__ import annotations

from hohere import vad


def test_mask_is_per_wavlm_frame_centred_and_sample_accurate():
    # one span covering samples 160..480: frame centres are 160, 480, 800
    spans = [(160, 480)]
    mask = vad.speech_mask50(spans, 3)
    assert mask.tolist() == [True, False, False]  # centre 160 inside, 480 is not


def test_mask_handles_many_spans_and_frames():
    spans = [(0, 400), (640, 1000)]
    mask = vad.speech_mask50(spans, 4)  # centres 160, 480, 800, 1120
    assert mask.tolist() == [True, False, True, False]
    assert mask.dtype == bool


def test_mask_is_empty_for_no_spans_or_no_frames():
    assert vad.speech_mask50([], 5).tolist() == [False] * 5
    assert vad.speech_mask50([(0, 1000)], 0).shape == (0,)


def test_pauses_only_look_at_internal_gaps():
    # 0.0-0.2 s speech, 0.3 s silence, 0.5-0.7 s speech; leading/trailing silence ignored
    spans = [(0, 3200), (8000, 11200)]
    assert vad.pauses(spans) == [3200 // 320]


def test_a_gap_under_200_ms_is_not_a_pause():
    spans = [(0, 3200), (3200 + 3199, 9000)]
    assert vad.pauses(spans) == []
    assert vad.pauses([(0, 3200), (6400, 9600)]) == [10]


def test_pauses_are_sorted_and_unsorted_input_is_fine():
    spans = [(6400, 9600), (0, 3200)]
    assert vad.pauses(spans) == [10]


def test_to_samples_rounds_to_samples():
    assert vad.to_samples([(0.5, 1.0)]) == [(8000, 16000)]
