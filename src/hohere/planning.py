"""Dataset planning: turn the skill mix into concrete conversation specs with fixed splits.

Splits are assigned per scenario family (skill x topic) *before* any generation, so
near-duplicate conversations from the same family never straddle splits.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Literal

import numpy as np

from hohere.config.schema import HohereConfig
from hohere.schemas.dialogue import ConversationSpec, Skill

Split = Literal["train", "validation", "test"]

# Rough turn-count preference per skill, as a fraction of [min_turns, max_turns].
_TURN_BIAS: dict[Skill, tuple[float, float]] = {
    Skill.SIMPLE_DIALOGUE: (0.0, 0.6),
    Skill.CONTEXT_MEMORY: (0.5, 1.0),
    Skill.TURN_TAKING: (0.2, 0.8),
    Skill.BACKCHANNEL: (0.3, 0.9),
    Skill.INTERRUPTION: (0.3, 0.9),
    Skill.INCOMPLETE_THOUGHTS: (0.2, 0.8),
}


def derive_seed(*parts: object) -> int:
    """Deterministic 31-bit seed from arbitrary parts."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def family_id(skill: Skill, topic: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in topic.lower()).strip("_")
    return f"{skill.value}/{slug}"


def conversations_per_skill(cfg: HohereConfig) -> dict[Skill, int]:
    ds = cfg.dataset
    out: dict[Skill, int] = {}
    for skill, entry in ds.skill_mix.items():
        seconds = ds.target_hours * 3600 * entry.share
        per_conversation = entry.estimated_seconds or ds.estimated_conversation_seconds
        out[skill] = math.ceil(seconds / per_conversation * ds.overprovision)
    return out


def assign_family_splits(families: list[str], cfg: HohereConfig, seed: int) -> dict[str, Split]:
    """Shuffle families deterministically and cut them by the configured ratios."""
    ds = cfg.dataset
    order = sorted(families)
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    n = len(order)
    n_val = round(n * ds.validation_ratio)
    n_test = round(n * ds.test_ratio)
    # Keep every non-zero split represented once there are enough families.
    if n >= 3:
        n_val = max(n_val, 1 if ds.validation_ratio > 0 else 0)
        n_test = max(n_test, 1 if ds.test_ratio > 0 else 0)
    n_train = n - n_val - n_test
    splits: dict[str, Split] = {}
    for i, fam in enumerate(order):
        if i < n_train:
            splits[fam] = "train"
        elif i < n_train + n_val:
            splits[fam] = "validation"
        else:
            splits[fam] = "test"
    return splits


def build_plan(cfg: HohereConfig) -> list[ConversationSpec]:
    seed = cfg.project.seed
    topics = list(dict.fromkeys(cfg.dataset.topics))
    lo, hi = cfg.dialogue.min_turns, cfg.dialogue.max_turns
    specs: list[ConversationSpec] = []
    index = 0
    for skill in Skill:
        entry = cfg.dataset.skill_mix.get(skill)
        if entry is None or entry.share <= 0:
            continue
        count = conversations_per_skill(cfg)[skill]
        rng = np.random.default_rng(derive_seed(seed, "plan", skill.value))
        skill_topics = list(topics)
        rng.shuffle(skill_topics)
        skill_topics = skill_topics[: max(1, min(len(skill_topics), count))]
        families = [family_id(skill, t) for t in skill_topics]
        splits = assign_family_splits(families, cfg, derive_seed(seed, "splits", skill.value))
        b_lo, b_hi = _TURN_BIAS[skill]
        for k in range(count):
            topic = skill_topics[k % len(skill_topics)]
            fam = family_id(skill, topic)
            t_lo = lo + round((hi - lo) * b_lo)
            t_hi = lo + round((hi - lo) * b_hi)
            index += 1
            conv_id = f"conv_{index:06d}"
            specs.append(
                ConversationSpec(
                    conversation_id=conv_id,
                    skill=skill,
                    topic=topic,
                    scenario_family=fam,
                    split=splits[fam],
                    seed=derive_seed(seed, conv_id),
                    target_turns=int(rng.integers(t_lo, t_hi + 1)),
                )
            )
    return specs


def plan_summary(specs: list[ConversationSpec]) -> dict:
    return {
        "conversations": len(specs),
        "by_skill": dict(Counter(s.skill.value for s in specs)),
        "by_split": dict(Counter(s.split for s in specs)),
        "families": len({s.scenario_family for s in specs}),
    }


def split_manifest(specs: list[ConversationSpec]) -> dict:
    """Immutable mapping written once per dataset: family -> split and conversation -> split."""
    return {
        "families": {s.scenario_family: s.split for s in specs},
        "conversations": {s.conversation_id: s.split for s in specs},
    }
