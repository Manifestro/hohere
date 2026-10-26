from __future__ import annotations

from hohere.config.schema import HohereConfig
from hohere.synthesis.base import AudioCache, SpeechSynthesizer, SynthesisError, trim_silence
from hohere.synthesis.runner import Segment, SynthesisRunner, VoiceAssignment, assign_voices


def make_synthesizer(cfg: HohereConfig) -> SpeechSynthesizer:
    if cfg.speech.provider == "mock":
        from hohere.synthesis.mock import MockSynthesizer

        return MockSynthesizer(cfg.speech.sample_rate)
    from hohere.synthesis.kokoro import KokoroSynthesizer

    return KokoroSynthesizer(cfg.speech)


__all__ = [
    "AudioCache",
    "Segment",
    "SpeechSynthesizer",
    "SynthesisError",
    "SynthesisRunner",
    "VoiceAssignment",
    "assign_voices",
    "make_synthesizer",
    "trim_silence",
]
