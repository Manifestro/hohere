from __future__ import annotations

from hohere.config.schema import HohereConfig
from hohere.generators.base import (
    DialogueGenerationError,
    DialogueGenerator,
    normalize_raw_dialogue,
)


def make_generator(cfg: HohereConfig) -> DialogueGenerator:
    if cfg.dialogue.provider == "mock":
        from hohere.generators.mock import MockGenerator

        return MockGenerator(cfg.dataset.language, cfg.dialogue.max_words_per_utterance)
    from hohere.generators.qwen import QwenGenerator

    return QwenGenerator(cfg.dialogue, cfg.dataset.language)


__all__ = [
    "DialogueGenerationError",
    "DialogueGenerator",
    "make_generator",
    "normalize_raw_dialogue",
]
