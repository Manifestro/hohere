"""The connection is reused, and batching keeps every chunk with its own result."""

from __future__ import annotations

import io
import types

import numpy as np

from hohere import pipeline
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


def test_encode_many_batches_by_length_and_keeps_the_order():
    sizes = []

    def batch(wavs, spans):
        sizes.append([w.size for w in wavs])
        return [(w.size, s) for w, s in zip(wavs, spans, strict=True)]

    stages = types.SimpleNamespace(encode_units_batch=batch, wavlm_batch=2)
    wavs = [np.zeros(n) for n in (30, 10, 20, 40, 15)]
    spans = [[i] for i in range(5)]
    out = pipeline.encode_many(stages, wavs, spans)
    assert out == [(30, [0]), (10, [1]), (20, [2]), (40, [3]), (15, [4])]
    assert sizes == [[10, 15], [20, 30], [40]]
