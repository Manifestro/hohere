from __future__ import annotations

import os
from pathlib import Path

import pytest

from hohere.config.schema import HohereConfig
from hohere.generators.mock import MockGenerator
from hohere.planning import build_plan
from hohere.schemas.dialogue import ConversationSpec, Dialogue, Skill
from hohere.synthesis import AudioCache, SynthesisRunner
from hohere.synthesis.mock import MockSynthesizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path: Path) -> HohereConfig:
    c = HohereConfig()
    c.dataset.target_hours = 0.1
    c.dataset.output_dir = tmp_path / "out"
    c.dataset.cache_dir = tmp_path / "cache"
    c.dialogue.provider = "mock"
    c.speech.provider = "mock"
    return c


def make_spec(skill: Skill, seed: int = 7, turns: int = 8, cid: str = "conv_000001"):
    return ConversationSpec(
        conversation_id=cid,
        skill=skill,
        topic="planning a weekend trip",
        scenario_family=f"{skill.value}/planning_a_weekend_trip",
        split="train",
        seed=seed,
        target_turns=turns,
    )


@pytest.fixture
def mock_dialogue_factory():
    gen = MockGenerator()

    def make(skill: Skill = Skill.SIMPLE_DIALOGUE, seed: int = 7, turns: int = 8) -> Dialogue:
        return gen.generate(make_spec(skill, seed, turns))

    return make


@pytest.fixture
def synth_runner(cfg: HohereConfig) -> SynthesisRunner:
    return SynthesisRunner(
        cfg.speech, MockSynthesizer(), AudioCache(cfg.dataset.cache_dir, cfg.speech.sample_rate)
    )


@pytest.fixture
def plan(cfg: HohereConfig):
    return build_plan(cfg)


def external_enabled() -> bool:
    return os.environ.get("HOHERE_RUN_EXTERNAL") == "1"
