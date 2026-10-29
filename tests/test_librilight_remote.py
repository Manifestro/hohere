from __future__ import annotations

import pytest

from hohere.librilight.remote import RangeError, read_with_retry

PAYLOAD = bytes(range(256)) * 40


def chunks(data: bytes, size: int = 1000):
    for start in range(0, len(data), size):
        yield data[start : start + size]


def test_reads_the_whole_range_across_chunks() -> None:
    sleeps: list[float] = []
    data = read_with_retry(lambda o, n: chunks(PAYLOAD), 0, len(PAYLOAD), sleep=sleeps.append)
    assert data == PAYLOAD
    assert sleeps == []


def test_retries_a_transport_error_then_succeeds() -> None:
    calls = {"n": 0}

    def stream(offset: int, n: int):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("the read timed out")
        return chunks(PAYLOAD)

    sleeps: list[float] = []
    assert read_with_retry(stream, 0, len(PAYLOAD), sleep=sleeps.append) == PAYLOAD
    assert calls["n"] == 3
    assert sleeps == [2.0, 4.0]  # exponential backoff


def test_retries_a_short_read() -> None:
    calls = {"n": 0}

    def stream(offset: int, n: int):
        calls["n"] += 1
        return chunks(PAYLOAD[: n // 2] if calls["n"] == 1 else PAYLOAD)

    data = read_with_retry(stream, 0, len(PAYLOAD), sleep=lambda _: None)
    assert data == PAYLOAD
    assert calls["n"] == 2


def test_aborts_a_stream_that_trickles_below_one_megabyte_per_second() -> None:
    clock = {"t": 0.0}

    def stall(offset: int, n: int):
        while True:  # 1000 bytes every 5 s = 200 B/s
            clock["t"] += 5.0
            yield b"\x00" * 1000

    with pytest.raises(RangeError, match="trickling"):
        read_with_retry(stall, 0, 10_000_000, retries=0, clock=lambda: clock["t"])

    clock["t"] = 0.0

    def fast(offset: int, n: int):
        while True:  # 1 MB every 0.1 s
            clock["t"] += 0.1
            yield b"\x00" * 1_000_000

    data = read_with_retry(fast, 0, 2_500_000, retries=0, clock=lambda: clock["t"])
    assert len(data) == 2_500_000


def test_gives_up_after_the_configured_retries() -> None:
    calls = {"n": 0}

    def broken(offset: int, n: int):
        calls["n"] += 1
        raise ConnectionResetError("peer reset")

    with pytest.raises(RangeError, match="after 3 tries"):
        read_with_retry(broken, 512, 10, retries=2, sleep=lambda _: None)
    assert calls["n"] == 3
