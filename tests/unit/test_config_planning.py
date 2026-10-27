from __future__ import annotations

from collections import Counter, defaultdict

import pytest
from pydantic import ValidationError

from hohere.config.schema import HohereConfig, load_config
from hohere.planning import build_plan, conversations_per_skill, split_manifest
from hohere.schemas.dialogue import Skill
from tests.conftest import ROOT


@pytest.mark.parametrize(
    "name", ["default", "smoke", "production", "mock", "v0.2", "v0.3", "v0.4"]
)
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.speech.sample_rate == 24000
    assert cfg.publishing.private is True


def test_bad_split_ratios_rejected():
    with pytest.raises(ValidationError):
        HohereConfig.model_validate({"dataset": {"train_ratio": 0.9}})


def test_bad_skill_mix_rejected():
    with pytest.raises(ValidationError):
        HohereConfig.model_validate({"dataset": {"skill_mix": {"simple_dialogue": {"share": 0.5}}}})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        HohereConfig.model_validate({"timeline": {"typo_field": 1}})


def test_plan_follows_skill_mix():
    cfg = HohereConfig()
    counts = conversations_per_skill(cfg)
    total = sum(counts.values())
    for skill, entry in cfg.dataset.skill_mix.items():
        assert counts[skill] / total == pytest.approx(entry.share, abs=0.01)


def test_plan_is_deterministic():
    cfg = HohereConfig()
    a = [s.model_dump() for s in build_plan(cfg)]
    b = [s.model_dump() for s in build_plan(cfg)]
    assert a == b
    cfg.project.seed = 43
    c = [s.model_dump() for s in build_plan(cfg)]
    assert a != c


def test_families_never_cross_splits():
    specs = build_plan(HohereConfig())
    fam_splits: dict[str, set[str]] = defaultdict(set)
    for s in specs:
        fam_splits[s.scenario_family].add(s.split)
    assert all(len(v) == 1 for v in fam_splits.values())
    manifest = split_manifest(specs)
    assert len(manifest["conversations"]) == len(specs)


def test_split_ratios_roughly_respected():
    specs = build_plan(HohereConfig())
    c = Counter(s.split for s in specs)
    n = len(specs)
    assert c["train"] / n == pytest.approx(0.8, abs=0.05)
    assert c["validation"] / n == pytest.approx(0.1, abs=0.04)
    assert c["test"] / n == pytest.approx(0.1, abs=0.04)


def test_unique_ids_and_all_skills():
    specs = build_plan(HohereConfig())
    assert len({s.conversation_id for s in specs}) == len(specs)
    assert {s.skill for s in specs} == set(Skill)
