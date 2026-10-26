"""Route synthesis by voice: the model (assistant) voice to its own TTS, users to another."""

from __future__ import annotations

import numpy as np

from hohere.synthesis.base import SpeechSynthesizer, SynthesisError


class VoiceRouter:
    def __init__(
        self, model_voice: str, model_tts: SpeechSynthesizer, user_tts: SpeechSynthesizer
    ) -> None:
        if model_tts.sample_rate != user_tts.sample_rate:
            raise SynthesisError(
                f"sample rates differ: {model_tts.sample_rate} vs {user_tts.sample_rate}"
            )
        self.model_voice = model_voice
        self.model_tts = model_tts
        self.user_tts = user_tts
        self.sample_rate = user_tts.sample_rate
        self.name = f"{model_tts.name}+{user_tts.name}"
        self.model = f"{model_tts.model}+{user_tts.model}"

    def backend(self, voice: str) -> SpeechSynthesizer:
        return self.model_tts if voice == self.model_voice else self.user_tts

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        return self.backend(voice).synthesize(text, voice, speed)
