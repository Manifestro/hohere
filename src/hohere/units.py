"""The m-stream: one token per Mimi frame, built from the sparse record events.

Mimi runs at 12.5 Hz, WavLM (which produced the syllable units and the pause
events) at 50 Hz, so an event that starts at WavLM frame `f` belongs to Mimi frame
`f // 4` (equivalently `int(start_s * 12.5)`).

Vocabulary: units are `0..k-1`, `SIL = k`, `PAD = k+1`. Events are placed in time
order; if the frame an event asks for is already taken it moves to the next free
frame, and an event that would land past the end of the chunk is dropped. Both
`build_m` (array only) and `build_m_with_stats` (array + counters) are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

WAVLM_HZ = 50
MIMI_HZ = 12.5
FRAMES_PER_MIMI = WAVLM_HZ // int(MIMI_HZ)  # 4
VARIANTS = ("A", "B", "B-sil")
DTYPE = np.int16


@dataclass(frozen=True)
class MStats:
    """What happened while placing the events of one chunk."""

    placed: int = 0  # events written into the stream
    shifted: int = 0  # of those, how many had to move to a later frame
    dropped: int = 0  # past the last frame, or with an out-of-range token


def token_bounds(k: int) -> tuple[int, int]:
    """`(SIL, PAD)` for a unit vocabulary of size `k`."""
    if k <= 0:
        raise ValueError("k must be positive")
    return k, k + 1


def _unit_key(k: int) -> str:
    if k == 4096:
        return "unit_k4096"
    if k == 8192:
        return "unit_k8192"
    raise ValueError(f"k must be 4096 or 8192, got {k}")


def _events(record: dict[str, Any], variant: str, k: int) -> list[tuple[int, int, int, str]]:
    """`(target Mimi frame, kind, index, token)` events, in placement priority order."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    if variant == "A":
        return []
    sil, _ = token_bounds(k)
    events: list[tuple[int, int, int, str]] = []
    if variant == "B":
        starts = np.asarray(record[_unit_key(k)]).reshape(-1)
        frames = np.asarray(record["unit_start"]).reshape(-1)
        for index, (frame, token) in enumerate(zip(frames, starts, strict=True)):
            events.append((int(frame) // FRAMES_PER_MIMI, 0, index, int(token)))
    if "sil_start" in record:
        frames = np.asarray(record["sil_start"]).reshape(-1)
        for index, frame in enumerate(frames):
            events.append((int(frame) // FRAMES_PER_MIMI, 1, index, sil))
    # Time order; at the same frame units come before pauses.
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    return events


def build_m_with_stats(
    record: dict[str, Any],
    variant: str = "B",
    k: int = 4096,
    *,
    frames: int | None = None,
) -> tuple[np.ndarray, MStats]:
    """Build the m stream of one chunk and report how the events were placed.

    `record` needs `codes` (or an explicit `frames`) and, depending on `variant`,
    `unit_start` with `unit_k4096`/`unit_k8192`, and `sil_start`.
    """
    if frames is None:
        codes = np.asarray(record["codes"])
        if codes.ndim != 2:
            raise ValueError("codes must be 2-D [codebooks, T]")
        frames = int(codes.shape[1])
    if frames < 0:
        raise ValueError("frames must be >= 0")
    _, pad = token_bounds(k)
    stream = np.full(frames, pad, dtype=DTYPE)
    placed = shifted = dropped = 0
    cursor = -1  # last frame that was written
    for frame, _kind, _index, token in _events(record, variant, k):
        if token != k and not 0 <= token < k:
            dropped += 1
            continue
        target = max(frame, cursor + 1)
        if target >= frames:
            dropped += 1
            continue
        stream[target] = token
        placed += 1
        if target != frame:
            shifted += 1
        cursor = target
    return stream, MStats(placed=placed, shifted=shifted, dropped=dropped)


def build_m(
    record: dict[str, Any],
    variant: str = "B",
    k: int = 4096,
    *,
    frames: int | None = None,
) -> np.ndarray:
    """The m stream alone: `int16[T]` with units `0..k-1`, `SIL = k`, `PAD = k+1`."""
    return build_m_with_stats(record, variant, k, frames=frames)[0]


def pair_frames(
    record: dict[str, Any],
    variant: str = "B",
    k: int = 4096,
) -> np.ndarray:
    """Training pairs: `(T, 2)` with the m token and Mimi codebook 0 at every frame."""
    codes = np.asarray(record["codes"])
    if codes.ndim != 2 or codes.shape[0] < 1:
        raise ValueError("codes must be [codebooks >= 1, T]")
    frames = int(codes.shape[1])
    m = build_m(record, variant, k, frames=frames)
    return np.stack([m, codes[0].astype(DTYPE)], axis=1)
