"""RangeReader: exact reads, retries and the stall guard."""

from __future__ import annotations

import pytest

from hohere.remote import RangeError, RangeReader

DATA = bytes(range(256)) * 4  # 1024 bytes


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def reader(opener, size: int = len(DATA), **kwargs) -> RangeReader:
    kwargs.setdefault("sleep", lambda _s: None)
    kwargs.setdefault("backoff", 0.0)
    return RangeReader("http://example/test.tar", size=size, opener=opener, **kwargs)


def test_read_returns_exact_bytes_and_never_passes_the_end():
    seen: list[tuple[int, int]] = []

    def opener(offset: int, length: int):
        seen.append((offset, length))
        yield DATA[offset : offset + length]

    handle = reader(opener)
    assert handle.read(0, 16) == DATA[:16]
    assert handle.read(1000, 100) == DATA[1000:]
    assert seen == [(0, 16), (1000, 24)]
    assert handle.read(2000, 8) == b""


def test_stream_chunks_and_clamps():
    def opener(offset: int, length: int):
        for start in range(offset, offset + length, 128):
            yield DATA[start : start + 128]

    handle = reader(opener)
    pieces = list(handle.stream(900, 500))
    assert b"".join(pieces) == DATA[900:]
    assert all(len(piece) <= 128 for piece in pieces)


def test_retries_a_failing_opener_and_resumes_at_the_same_offset():
    calls: list[tuple[int, int]] = []

    def opener(offset: int, length: int):
        calls.append((offset, length))
        if len(calls) < 3:
            raise ConnectionError("boom")
        yield DATA[offset : offset + length]

    handle = reader(opener)
    assert handle.read(0, len(DATA)) == DATA
    assert calls == [(0, 1024), (0, 1024), (0, 1024)]


def test_gives_up_after_the_retry_budget():
    def opener(offset: int, length: int):
        raise ConnectionError("down")
        yield b""  # pragma: no cover - makes this a generator

    handle = reader(opener, retries=2)
    with pytest.raises(RangeError, match="after 3 attempts"):
        handle.read(0, 16)


def test_short_read_is_retried_from_the_last_byte_received():
    calls: list[int] = []

    def opener(offset: int, length: int):
        calls.append(offset)
        if len(calls) == 1:
            yield DATA[offset : offset + 100]  # ends early: no exception, just short
            return
        yield DATA[offset : offset + length]

    handle = reader(opener)
    assert handle.read(0, 200) == DATA[:200]
    assert calls == [0, 100]


def test_stall_guard_aborts_a_trickle_and_resumes_from_the_last_byte():
    clock = Clock()
    calls: list[tuple[int, int]] = []
    data = DATA * 16  # 16 KiB is enough to keep the stream alive past one chunk

    def opener(offset: int, length: int):
        calls.append((offset, length))
        if len(calls) == 1:
            for start in range(offset, offset + min(length, 4096), 1024):
                clock.t += 25.0  # 25 s for 1 KiB: ~40 B/s, far below 256 KiB/s
                yield data[start : start + 1024]
            return
        yield data[offset : offset + length]

    handle = reader(opener, size=len(data), clock=clock, min_speed=256 * 1024, stall_window=20.0)
    assert handle.read(0, 5000) == data[:5000]
    assert calls[0][0] == 0
    assert calls[1][0] == 1024  # resumed, not restarted


def test_size_from_head_is_cached_and_reused():
    handle = RangeReader("http://example/test.tar", size=123)
    assert handle.size() == 123
    assert handle.size() == 123


def test_negative_ranges_are_rejected():
    handle = reader(lambda offset, length: iter([DATA[offset : offset + length]]))
    with pytest.raises(ValueError):
        handle.read(-1, 10)
