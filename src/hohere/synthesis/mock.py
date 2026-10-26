"""Deterministic speech-like tone generator for tests and offline dry runs."""

from __future__ import annotations

import hashlib

import numpy as np


class MockSynthesizer:
    name = "mock"
    model = "mock-tts-v1"
    version = "1"

    def __init__(self, sample_rate: int = 24000, seconds_per_word: float = 0.32) -> None:
        self.sample_rate = sample_rate
        self.seconds_per_word = seconds_per_word

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(f"{voice}|{speed}|{text}".encode()).digest()[:4])
        rng = np.random.default_rng(seed)
        n_words = max(1, len(text.split()))
        dur = (n_words * self.seconds_per_word + 0.15) / speed
        n = int(dur * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        pitch = 110 + (int(hashlib.md5(voice.encode()).hexdigest(), 16) % 120)
        tone = sum(np.sin(2 * np.pi * pitch * h * t) / h for h in (1, 2, 3))
        syllables = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3.5, 5.0) * t) ** 2
        audio = 0.25 * tone * syllables + 0.01 * rng.standard_normal(n)
        ramp = min(n // 4, int(0.02 * self.sample_rate))
        if ramp:
            env = np.ones(n)
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
            audio *= env
        return audio.astype(np.float32)
