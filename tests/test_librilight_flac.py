from __future__ import annotations

import pytest

from hohere.librilight.flac import parse_streaminfo

RATE = 24000


def streaminfo(sample_rate: int = RATE, channels: int = 1, bits: int = 16, total: int = 0) -> bytes:
    """The first metadata block of a FLAC file, hand-packed."""
    packed = (sample_rate << 44) | ((channels - 1) << 41) | ((bits - 1) << 36) | total
    fixed = b"\x10\x00\x10\x00"  # min/max block size
    frames = b"\x00\x00\x00" * 2  # min/max frame size
    body = fixed + frames + packed.to_bytes(8, "big") + b"\x00" * 16  # 16-byte md5
    return b"fLaC" + bytes([0x80]) + len(body).to_bytes(3, "big") + body


def test_parses_sample_rate_and_total_samples() -> None:
    info = parse_streaminfo(streaminfo(total=RATE * 3600))
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (RATE, 1, 16)
    assert info.total_samples == RATE * 3600
    assert info.duration == pytest.approx(3600.0)


def test_parses_odd_values_a_zero_length_stream_and_a_4_kib_slice() -> None:
    info = parse_streaminfo(streaminfo(sample_rate=16000, channels=2, bits=24, total=0))
    assert (info.sample_rate, info.channels, info.bits_per_sample) == (16000, 2, 24)
    assert info.total_samples == 0 and info.duration == 0.0
    assert parse_streaminfo(streaminfo(total=RATE * 90) + b"\xff" * 4000).duration == 90.0


def test_rejects_a_file_that_is_not_flac() -> None:
    with pytest.raises(ValueError, match="fLaC"):
        parse_streaminfo(b"OggS" + b"\x00" * 40)
    wrong_block = b"fLaC" + bytes([0x04]) + (34).to_bytes(3, "big") + b"\x00" * 34
    with pytest.raises(ValueError, match="STREAMINFO"):
        parse_streaminfo(wrong_block)


def test_rejects_a_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated"):
        parse_streaminfo(streaminfo()[:30])
