"""Speech synthesizer interface, fragment cache and silence trimming."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import soundfile as sf


class SynthesisError(RuntimeError):
    """The TTS provider failed or returned unusable audio."""


@runtime_checkable
class SpeechSynthesizer(Protocol):
    name: str
    model: str
    sample_rate: int

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        """Return mono float32 audio in [-1, 1] at `sample_rate`."""
        ...


def cache_key(provider: str, model: str, voice: str, speed: float, text: str) -> str:
    payload = f"{provider}|{model}|{voice}|{speed:.3f}|{text}"
    return hashlib.sha256(payload.encode()).hexdigest()


class AudioCache:
    """Content-addressed cache of synthesized fragments (lossless float32 WAV)."""

    def __init__(self, root: Path, sample_rate: int) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.wav"

    def get(self, key: str) -> np.ndarray | None:
        p = self.path(key)
        if not p.exists():
            return None
        audio, sr = sf.read(p, dtype="float32")
        if sr != self.sample_rate:
            return None
        return audio

    def put(self, key: str, audio: np.ndarray) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp{os.getpid()}.wav")
        sf.write(tmp, audio, self.sample_rate, subtype="FLOAT")
        tmp.replace(p)


def trim_silence(
    audio: np.ndarray, sample_rate: int, threshold_db: float, padding_ms: int
) -> np.ndarray:
    """Trim leading/trailing silence using 10 ms RMS frames, keeping `padding_ms`."""
    if audio.size == 0:
        return audio
    frame = max(1, sample_rate // 100)
    n = len(audio) // frame
    if n == 0:
        return audio
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    voiced = np.flatnonzero(db > threshold_db)
    if voiced.size == 0:
        return audio[:0]
    pad = int(sample_rate * padding_ms / 1000)
    start = max(0, voiced[0] * frame - pad)
    end = min(len(audio), (voiced[-1] + 1) * frame + pad)
    return audio[start:end]
