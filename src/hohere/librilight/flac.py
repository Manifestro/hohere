"""FLAC STREAMINFO: sample rate and length of one source file, from its first bytes.

The archive keeps a 321 GB tar on the other side of an HTTPS Range request, so the
length of a file comes from the 34-byte STREAMINFO block at its start instead of from
decoding it. The first 4 KiB of every FLAC are enough.
"""

from __future__ import annotations

from dataclasses import dataclass

MARKER = b"fLaC"
STREAMINFO = 0
STREAMINFO_LEN = 34


@dataclass(frozen=True)
class StreamInfo:
    sample_rate: int
    channels: int
    bits_per_sample: int
    total_samples: int

    @property
    def duration(self) -> float:
        """Seconds; 0.0 when the encoder could not know the length (total_samples 0)."""
        return self.total_samples / self.sample_rate if self.sample_rate else 0.0


def parse_streaminfo(data: bytes) -> StreamInfo:
    """Parse the STREAMINFO of a FLAC file from at least its first 42 bytes."""
    if data[:4] != MARKER:
        raise ValueError("not a FLAC stream: missing fLaC marker")
    if len(data) < 8:
        raise ValueError("truncated FLAC metadata header")
    kind = data[4] & 0x7F
    length = int.from_bytes(data[5:8], "big")
    if kind != STREAMINFO:
        raise ValueError(f"first FLAC metadata block is type {kind}, expected STREAMINFO")
    if length < STREAMINFO_LEN or len(data) < 8 + STREAMINFO_LEN:
        raise ValueError(f"truncated STREAMINFO: {length} bytes")
    packed = int.from_bytes(data[18:26], "big")
    return StreamInfo(
        sample_rate=packed >> 44,
        channels=((packed >> 41) & 0x07) + 1,
        bits_per_sample=((packed >> 36) & 0x1F) + 1,
        total_samples=packed & ((1 << 36) - 1),
    )
