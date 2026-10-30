"""ZeroSyl post-processing: boundaries, speech filtering and unit extents (no torch)."""

from __future__ import annotations

import numpy as np
import pytest

from hohere import zerosyl


def features(frames: int, dim: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(frames, dim)).astype(np.float32)


def test_boundaries_are_zero_then_peaks_then_the_frame_count():
    boundary = features(40)
    bounds = zerosyl.boundaries_from_features(boundary, prominence=0.0)
    assert bounds[0] == 0
    assert bounds[-1] == 40
    assert bounds == sorted(bounds)
    assert len(bounds) >= 2


def test_a_flat_signal_has_no_internal_boundary():
    flat = np.zeros((30, 4), dtype=np.float32)
    flat[:, 0] = 1.0  # zero variance in the norms -> no peaks
    assert zerosyl.boundaries_from_features(flat) == [0, 30]


def test_normalisation_matches_layer_norm():
    audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = zerosyl.layer_norm(audio)
    assert abs(float(out.mean())) < 1e-6
    assert abs(float(out.std()) - 1.0) < 1e-5


def test_segment_features_keeps_only_speech_frames():
    boundary = features(40)
    semantic = features(40, dim=16, seed=1)
    mask = np.zeros(40, dtype=bool)
    mask[5:30] = True
    segments = zerosyl.segment_features(boundary, semantic, mask)
    assert len(segments) >= 1
    for start, end in zip(segments.unit_start, segments.unit_end, strict=True):
        assert mask[start : end].all(), (start, end)  # no silent frame inside a unit
        assert start < end
    assert segments.embeddings.shape == (len(segments), 16)
    assert segments.embeddings.dtype == np.float32


def test_a_segment_without_speech_is_dropped_and_counted():
    boundary = features(40)
    semantic = features(40, dim=16, seed=2)
    segments = zerosyl.segment_features(boundary, semantic, np.zeros(40, dtype=bool))
    assert len(segments) == 0
    assert segments.dropped == len(segments.boundaries) - 1
    assert segments.embeddings.shape == (0, 16)


def test_all_speech_reproduces_the_reference_boundaries_and_mean_pooling():
    boundary = features(40)
    semantic = features(40, dim=16, seed=3)
    segments = zerosyl.segment_features(boundary, semantic, np.ones(40, dtype=bool))
    bounds = segments.boundaries
    assert segments.unit_start.tolist() == bounds[:-1]
    assert segments.unit_end.tolist() == bounds[1:]
    first = semantic[bounds[0] : bounds[1]].mean(axis=0)
    assert np.allclose(segments.embeddings[0], first, atol=1e-6)


def test_mismatched_frame_counts_are_an_error():
    with pytest.raises(ValueError, match="disagree"):
        zerosyl.segment_features(features(10), features(11), np.ones(10, dtype=bool))


def test_smoothing_uses_edge_padding():
    values = np.arange(6, dtype=np.float64)
    smoothed = zerosyl.smooth(values, 3)
    assert np.allclose(smoothed, [1 / 3, 1.0, 2.0, 3.0, 4.0, 14 / 3])


def test_frame_count_matches_the_padded_reference():
    encoder = zerosyl.Encoder(model=None, device="cpu")
    assert encoder.n_frames(16000) == (16000 + 80) // 320
