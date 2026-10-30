"""Silero VAD: speech spans, the WavLM speech mask and the internal pauses.

Everything here except `speech` is a pure function of the spans, so the mask and the
pause rules are tested without a model. Silero is imported lazily and loaded once per
worker; `speech` is the only function that touches it.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence

import numpy as np

SAMPLE_RATE = 16000
WAVLM_HOP = 320  # 50 Hz: frames are 320 samples apart
WAVLM_WIN = 400  # 25 ms analysis window
MIN_PAUSE_S = 0.2  # a gap has to be at least 200 ms long to become a SIL
MIN_PAUSE_SAMPLES = int(MIN_PAUSE_S * SAMPLE_RATE)

Span = tuple[int, int]

_lock = threading.Lock()
_model = None


def load_model():
    """The Silero model, loaded once per process (one process == one worker)."""
    global _model
    with _lock:
        if _model is None:
            from silero_vad import load_silero_vad

            _model = load_silero_vad()
        return _model


def speech(wav16: np.ndarray, model=None) -> list[Span]:
    """Speech spans in samples of a 16 kHz float32 mono waveform."""
    import torch
    from silero_vad import get_speech_timestamps

    model = model if model is not None else load_model()
    audio = np.asarray(wav16, dtype=np.float32).reshape(-1)
    timestamps = get_speech_timestamps(
        torch.from_numpy(audio),
        model,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=200,
        speech_pad_ms=0,
    )
    spans: list[Span] = []
    for item in timestamps:
        start, end = int(item["start"]), int(item["end"])
        if end > start:
            spans.append((start, end))
    spans.sort()
    return spans


def speech_mask50(spans: Sequence[Span], n_frames: int) -> np.ndarray:
    """One bool per WavLM frame: the input is padded by 40 samples, so frame `i` covers
    [i*320-40, i*320+360) of the chunk.

    A frame is speech when its centre sample `i*320 + 160` lies inside a span.
    """
    frames = int(n_frames)
    mask = np.zeros(max(0, frames), dtype=bool)
    if not spans or frames <= 0:
        return mask
    starts = np.array([int(start) for start, _end in spans], dtype=np.int64)
    ends = np.array([int(end) for _start, end in spans], dtype=np.int64)
    centres = np.arange(frames, dtype=np.int64) * WAVLM_HOP + WAVLM_WIN // 2 - 40
    index = np.searchsorted(starts, centres, side="right") - 1
    inside = index >= 0
    safe = np.where(inside, index, 0)
    return inside & (centres < ends[safe])


def pauses(spans: Sequence[Span]) -> list[int]:
    """`sil_start` frames of the internal pauses, as 50 Hz WavLM frames.

    Only the gaps *between* two speech spans count: the leading and trailing silence
    of a chunk is not a pause, so a chunk starts and ends with speech.
    """
    ordered = sorted((s, e) for s, e in spans)
    out: list[int] = []
    for index in range(len(ordered) - 1):
        end = int(ordered[index][1])
        next_start = int(ordered[index + 1][0])
        if next_start - end >= MIN_PAUSE_SAMPLES:
            out.append(end // WAVLM_HOP)
    return out


def to_samples(spans: Sequence[Sequence[float]]) -> list[Span]:
    """Libri-Light VAD spans in seconds -> sample spans."""
    return [(round(s * SAMPLE_RATE), round(e * SAMPLE_RATE)) for s, e in spans]
