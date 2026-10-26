"""Per-conversation synthesis: stable voice mapping, caching, retries, trimming."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from hohere.config.schema import SpeechConfig
from hohere.planning import derive_seed
from hohere.schemas.dialogue import Dialogue, Speaker
from hohere.synthesis.base import (
    AudioCache,
    SpeechSynthesizer,
    SynthesisError,
    cache_key,
    trim_silence,
)

log = logging.getLogger(__name__)

PEAK_LIMIT = 0.95


@dataclass
class Segment:
    utterance_id: str
    speaker: Speaker
    text: str
    voice: str
    speed: float
    cache_key: str
    audio: np.ndarray = field(repr=False)
    gain: float = 1.0

    @property
    def num_samples(self) -> int:
        return int(self.audio.shape[0])

    def meta(self) -> dict:
        return {
            "utterance_id": self.utterance_id,
            "speaker": self.speaker,
            "text": self.text,
            "voice": self.voice,
            "speed": self.speed,
            "cache_key": self.cache_key,
            "num_samples": self.num_samples,
            "gain": self.gain,
        }


@dataclass
class VoiceAssignment:
    voices: dict[Speaker, str]
    speeds: dict[Speaker, float]


def assign_voices(dialogue: Dialogue, cfg: SpeechConfig) -> VoiceAssignment:
    """Deterministic, stable speaker -> voice/speed mapping for one conversation."""
    rng = np.random.default_rng(derive_seed(dialogue.seed, "voices"))
    pool = sorted(set(cfg.voices))
    a, b = rng.choice(len(pool), size=2, replace=False)
    speeds = rng.uniform(cfg.speed_min, cfg.speed_max, size=2)
    return VoiceAssignment(
        voices={"A": pool[int(a)], "B": pool[int(b)]},
        speeds={"A": round(float(speeds[0]), 3), "B": round(float(speeds[1]), 3)},
    )


class SynthesisRunner:
    def __init__(self, cfg: SpeechConfig, synthesizer: SpeechSynthesizer, cache: AudioCache):
        if synthesizer.sample_rate != cfg.sample_rate:
            raise SynthesisError(
                f"synthesizer rate {synthesizer.sample_rate} != config {cfg.sample_rate}"
            )
        self.cfg = cfg
        self.tts = synthesizer
        self.cache = cache
        self.cache_hits = 0
        self.cache_misses = 0

    def _utterance_speed(self, base: float, utterance_id: str, seed: int) -> float:
        jitter = self.cfg.utterance_speed_jitter
        if jitter <= 0:
            return base
        rng = np.random.default_rng(derive_seed(seed, "speed", utterance_id))
        return round(float(base * (1 + rng.uniform(-jitter, jitter))), 3)

    def _synthesize_with_retries(self, text: str, voice: str, speed: float) -> np.ndarray:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                audio = np.asarray(self.tts.synthesize(text, voice, speed), dtype=np.float32)
                if audio.ndim != 1 or audio.size == 0:
                    raise SynthesisError("empty or non-mono audio")
                if not np.all(np.isfinite(audio)):
                    raise SynthesisError("non-finite samples")
                return audio
            except Exception as e:
                last = e
                log.warning("synthesis attempt %d failed for %r: %s", attempt + 1, text[:40], e)
                if attempt < self.cfg.max_retries:
                    time.sleep(min(8.0, 0.5 * 2**attempt))
        raise SynthesisError(f"synthesis failed after retries: {last}") from last

    def synthesize_dialogue(self, dialogue: Dialogue) -> tuple[VoiceAssignment, list[Segment]]:
        assignment = assign_voices(dialogue, self.cfg)
        segments: list[Segment] = []
        for turn in dialogue.turns:
            voice = assignment.voices[turn.speaker]
            speed = self._utterance_speed(assignment.speeds[turn.speaker], turn.id, dialogue.seed)
            key = cache_key(self.tts.name, self.tts.model, voice, speed, turn.text)
            raw = self.cache.get(key)
            if raw is None:
                self.cache_misses += 1
                raw = self._synthesize_with_retries(turn.text, voice, speed)
                self.cache.put(key, raw)
            else:
                self.cache_hits += 1
            audio = raw
            if self.cfg.trim_silence:
                audio = trim_silence(
                    audio,
                    self.cfg.sample_rate,
                    self.cfg.trim_threshold_db,
                    self.cfg.trim_padding_ms,
                )
            if audio.size == 0:
                raise SynthesisError(f"utterance {turn.id} is silent after trimming")
            peak = float(np.max(np.abs(audio)))
            gain = PEAK_LIMIT / peak if peak > PEAK_LIMIT else 1.0
            segments.append(
                Segment(
                    utterance_id=turn.id,
                    speaker=turn.speaker,
                    text=turn.text,
                    voice=voice,
                    speed=speed,
                    cache_key=key,
                    audio=(audio * gain).astype(np.float32),
                    gain=round(gain, 6),
                )
            )
        return assignment, segments
