"""Kokoro-82M speech synthesis (https://huggingface.co/hexgrad/Kokoro-82M).

Install with `uv sync --extra kokoro`; requires the `espeak-ng` system package.
"""

from __future__ import annotations

import threading

import numpy as np

from hohere.config.schema import SpeechConfig
from hohere.synthesis.base import SynthesisError

# Kokoro voice prefix -> pipeline language code ("a" American, "b" British English).
_LANG_CODES = {"a", "b"}


class KokoroSynthesizer:
    name = "kokoro"
    sample_rate = 24000

    def __init__(self, cfg: SpeechConfig) -> None:
        try:
            import kokoro  # noqa: F401
        except ImportError as e:
            raise SynthesisError("kokoro is not installed; run `uv sync --extra kokoro`") from e
        self.cfg = cfg
        self.model = cfg.model
        self.device = cfg.device or self._auto_device()
        self._pipelines: dict[str, object] = {}
        self._model_obj = None
        self._lock = threading.Lock()
        for v in cfg.voices:
            if v[:1] not in _LANG_CODES:
                raise SynthesisError(f"voice {v!r} is not an English Kokoro voice")

    @staticmethod
    def _auto_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _pipeline(self, lang_code: str):
        with self._lock:
            if lang_code not in self._pipelines:
                from kokoro import KModel, KPipeline

                if self._model_obj is None:
                    self._model_obj = KModel(repo_id=self.model).to(self.device).eval()
                self._pipelines[lang_code] = KPipeline(
                    lang_code=lang_code, repo_id=self.model, model=self._model_obj
                )
            return self._pipelines[lang_code]

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        pipeline = self._pipeline(voice[0])
        chunks: list[np.ndarray] = []
        for result in pipeline(text, voice=voice, speed=speed, split_pattern=None):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        if not chunks:
            raise SynthesisError(f"kokoro returned no audio for {text!r}")
        return np.concatenate(chunks)

    @property
    def version(self) -> str:
        try:
            from importlib.metadata import version

            return version("kokoro")
        except Exception:
            return "unknown"
