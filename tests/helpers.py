"""Shared test helpers: an in-memory archive and a range reader over bytes."""

from __future__ import annotations

import io
import json
import tarfile
from collections.abc import Iterable


class FakeReader:
    """The minimum of `RangeReader`: `size()` and `read(offset, length)` over bytes."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.requests: list[tuple[int, int]] = []

    def size(self) -> int:
        return len(self.data)

    def read(self, offset: int, length: int) -> bytes:
        self.requests.append((offset, length))
        return self.data[offset : offset + length]


def make_tar(
    members: Iterable[tuple[str, bytes]], fmt: int = tarfile.GNU_FORMAT
) -> bytes:
    """A tar in memory; GNU format keeps long names as `L` headers, PAX as `x`."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=fmt) as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def flac_head(sample_rate: int = 16000, samples: int = 16000, channels: int = 1) -> bytes:
    """The first 42 bytes of a FLAC stream: `fLaC`, STREAMINFO header, STREAMINFO."""
    packed = (sample_rate << 44) | ((channels - 1) << 41) | (15 << 36) | samples
    streaminfo = b"\x00" * 10 + packed.to_bytes(8, "big") + b"\x00" * 16
    assert len(streaminfo) == 34
    return b"fLaC" + bytes([0x80, 0, 0, 34]) + streaminfo


def flac_member(sample_rate: int = 16000, samples: int = 16000) -> bytes:
    """A fake FLAC payload: a real STREAMINFO head followed by junk frames."""
    return flac_head(sample_rate, samples) + b"\xff" * 500


def vad_json(*spans: tuple[float, float]) -> bytes:
    return json.dumps({"voice_activity": [list(span) for span in spans]}).encode()


def librilight_tar(split: str = "small", count: int = 3) -> bytes:
    members: list[tuple[str, bytes]] = []
    for index in range(count):
        stem = f"book_{index}-chapter-{index}"
        members.append((f"{split}/speaker/some book/{stem}.flac", flac_member()))
        members.append(
            (f"{split}/speaker/some book/{stem}.json", vad_json((0.5, 3.5), (4.0, 6.0)))
        )
    return make_tar(members)


def record(
    frames: int = 32,
    unit_starts: tuple[int, ...] = (0, 4, 8),
    sil_starts: tuple[int, ...] = (12,),
    k: int = 4096,
    codebooks: int = 32,
) -> dict:
    """A valid chunk record ready for `write_record`/`build_m`."""
    import numpy as np

    n = len(unit_starts)
    codes = np.zeros((codebooks, frames), dtype=np.int16)
    codes[0] = np.arange(frames, dtype=np.int16)
    return {
        "codes": codes,
        "unit_start": np.asarray(unit_starts, dtype=np.int32),
        "unit_end": np.asarray(unit_starts, dtype=np.int32) + 2,
        "unit_k4096": np.arange(n, dtype=np.int16),
        "unit_k8192": np.arange(n, dtype=np.int16) + 100,
        "sil_start": np.asarray(sil_starts, dtype=np.int32),
    }
