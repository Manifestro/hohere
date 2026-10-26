from __future__ import annotations

from hohere.config.schema import HohereConfig
from hohere.synthesis.base import AudioCache, SpeechSynthesizer, SynthesisError, trim_silence
from hohere.synthesis.runner import Segment, SynthesisRunner, VoiceAssignment, assign_voices


def _make(provider: str, cfg: HohereConfig) -> SpeechSynthesizer:
    sc = cfg.speech
    if provider == "mock":
        from hohere.synthesis.mock import MockSynthesizer

        return MockSynthesizer(sc.sample_rate)
    if provider.startswith("chatterbox"):
        from hohere.synthesis.chatterbox import ChatterboxSynthesizer

        return ChatterboxSynthesizer(
            sc.model_reference_audio,  # type: ignore[arg-type]  # validated in config
            variant="turbo" if provider == "chatterbox_turbo" else "orig",
            python=sc.chatterbox_python,
            device=sc.device,
        )
    from hohere.synthesis.kokoro import KokoroSynthesizer

    return KokoroSynthesizer(sc)


def make_synthesizer(cfg: HohereConfig) -> SpeechSynthesizer:
    """User voices use `speech.provider`; the model voice may use its own provider."""
    sc = cfg.speech
    user_tts = _make(sc.provider, cfg)
    if sc.model_provider == "same":
        return user_tts
    from hohere.synthesis.router import VoiceRouter

    return VoiceRouter(sc.model_voice, _make(sc.model_provider, cfg), user_tts)


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
