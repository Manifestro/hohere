"""The connection is reused, and batching keeps every chunk with its own result."""

from __future__ import annotations

import io
import types

import numpy as np

from hohere import pipeline, zerosyl
from hohere.remote import RangeReader


class FakeClient:
    def __init__(self) -> None:
        self.resets = 0

    def get(self, offset, length):
        response = io.BytesIO(b"x" * length)
        response.status = 206
        return response

    def reset(self):
        self.resets += 1


def test_a_complete_read_keeps_the_connection():
    reader = RangeReader("https://example.test/a.tar", size=1 << 20)
    client = FakeClient()
    reader._client = lambda: client
    assert reader.read(100, 4096) == b"x" * 4096
    assert reader.read(9000, 512) == b"x" * 512
    assert client.resets == 0


def test_an_aborted_read_drops_the_connection():
    reader = RangeReader("https://example.test/a.tar", size=1 << 30)
    client = FakeClient()
    reader._client = lambda: client
    stream = reader.stream(0, 1 << 22, chunk=1024)
    next(stream)
    stream.close()
    assert client.resets == 1


def _segments(start, end):
    return zerosyl.Segments(
        unit_start=np.array([start], dtype=np.int32),
        unit_end=np.array([end], dtype=np.int32),
        embeddings=np.zeros((1, 4), dtype=np.float32),
        boundaries=[0, end],
    )


def test_a_chunk_is_cut_at_pauses_on_frame_boundaries():
    second = 16000
    spans = [(0, 12 * second), (13 * second, 25 * second), (26 * second, 40 * second)]
    pieces = zerosyl.split_chunk(40 * second, spans, 20.0)
    assert pieces == [
        (0, 12 * second + second // 2),
        (12 * second + second // 2, 25 * second + second // 2),
        (25 * second + second // 2, 40 * second),
    ]
    assert all(start % 320 == 0 for start, _end in pieces)


def test_speech_without_a_pause_stays_whole():
    assert zerosyl.split_chunk(960000, [(0, 960000)], 20.0) == [(0, 960000)]


def test_spans_of_a_piece_are_relative_and_clipped():
    assert zerosyl.spans_of_piece([(0, 100), (200, 400), (500, 600)], 300, 550) == [
        (0, 100),
        (200, 250),
    ]


def test_encode_many_merges_pieces_in_order_and_batches_by_length():
    sizes = []

    def batch(wavs, spans):
        sizes.append([w.size for w in wavs])
        return [_segments(1, w.size // 320) for w in wavs]

    stages = types.SimpleNamespace(encode_units_batch=batch, wavlm_batch=2, wavlm_max_s=20.0)
    second = 16000
    long_spans = [(0, 15 * second), (16 * second, 31 * second)]  # cut at 15.5 s
    short_spans = [(0, 10 * second)]
    wavs = [np.zeros(31 * second, np.float32), np.zeros(10 * second, np.float32)]
    first, second_chunk = pipeline.encode_many(stages, wavs, [long_spans, short_spans])
    assert first.unit_start.tolist() == [1, 1 + 15 * 50 + 25]  # the second piece is shifted
    assert second_chunk.unit_start.tolist() == [1]
    assert sizes[0][0] <= sizes[0][-1]  # shortest first, close lengths share a batch
