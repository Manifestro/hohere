"""Frame-level speech activity of one segment, and the pause rule it has to pass.

A Mimi frame is 1920 samples at 24 kHz (12.5 frames/s). A frame counts as speech when
its RMS over the zero-padded audio is above -45 dBFS, which is the same threshold the
composed datasets use, so a segment without any active frame is digital silence and a
segment without a long internal pause has no place to put a turn boundary.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
THRESHOLD_DB = -45.0
AMPLITUDE = 10 ** (THRESHOLD_DB / 20)

# Codes Mimi produces for digital silence; `encode` refuses to run if they change.
SILENCE_CODES = (1049, 1700, 1626, 546, 306, 1443, 1871, 2008)

# A usable pause: 3 inactive frames (0.24 s) that start at frame 38 (3 s) and end at
# most 25 frames (2 s) before the end of the segment.
PAUSE_FRAMES = 3
PAUSE_START = 38
PAUSE_END = 25


def num_frames(num_samples: int, frame: int = FRAME_SAMPLES) -> int:
    """Frames of `num_samples`, the last one zero-padded."""
    return -(-num_samples // frame)


def activity(
    audio: np.ndarray, frame: int = FRAME_SAMPLES, amplitude: float = AMPLITUDE
) -> np.ndarray:
    """uint8 activity of every zero-padded frame."""
    n = num_frames(audio.shape[0], frame)
    if n == 0:
        return np.zeros(0, np.uint8)
    padded = np.zeros(n * frame, np.float32)
    padded[: audio.shape[0]] = audio
    rms = np.sqrt(np.mean(np.square(padded, dtype=np.float64).reshape(n, frame), axis=1))
    return (rms > amplitude).astype(np.uint8)


def pause_runs(act: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of inactive frames as `[start, end)` frame indices."""
    total = int(act.shape[0])
    if total == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([1], act.astype(np.int8), [1]))))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def has_pause(act: np.ndarray) -> bool:
    """True if a pause of >= 3 frames starts at frame >= 38 and ends at <= T - 25.

    `T` is the frame count of the segment, so the run leaves 2 s of room after it for
    the next turn and 3 s before it for the previous one.
    """
    total = int(act.shape[0])
    return any(
        end - start >= PAUSE_FRAMES and start >= PAUSE_START and end <= total - PAUSE_END
        for start, end in pause_runs(act)
    )


def is_silence(codes: np.ndarray) -> bool:
    """True if `codes` (shape `[8, frames]`) is the silence codebook everywhere."""
    expected = np.asarray(SILENCE_CODES, np.int64)[:, None]
    return bool(np.all(codes.astype(np.int64) == expected))
