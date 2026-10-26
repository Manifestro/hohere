"""Datasets sized by words instead of hours."""

from __future__ import annotations

import pytest

from hohere.pipeline import Pipeline
from hohere.planning import conversations_per_skill
from hohere.validation.dialogue import dialogue_words


def test_plan_sized_by_words(cfg):
    cfg.dataset.target_words = 50_000
    cfg.dataset.overprovision = 1.0
    counts = conversations_per_skill(cfg)
    words_per_conv = cfg.dataset.estimated_conversation_words
    assert sum(counts.values()) == pytest.approx(50_000 / words_per_conv, rel=0.02)
    cfg.dataset.target_words = 100_000
    assert sum(conversations_per_skill(cfg).values()) == pytest.approx(
        2 * sum(counts.values()), rel=0.02
    )


def test_generation_stops_at_word_target(cfg):
    cfg.dataset.target_words = 2_000
    cfg.dataset.overprovision = 3.0  # plan far more dialogues than the target needs
    pipe = Pipeline(cfg, show_progress=False)
    report = pipe.generate()
    ids = pipe.store.accepted_dialogue_ids()
    total = sum(sum(dialogue_words(pipe.store.load_dialogue(c)).values()) for c in ids)
    assert total >= 2_000
    biggest = max(sum(dialogue_words(pipe.store.load_dialogue(c)).values()) for c in ids)
    # stops right after crossing the target (parallel workers may overshoot a little)
    assert total < 2_000 + cfg.dialogue.concurrency * biggest
    assert report["counts"]["skipped_target_reached"] > 0
    assert report["accepted_words"] == total
    # resuming does not generate more once the target is met
    again = pipe.generate()
    assert again["counts"].get("accepted", 0) == 0


def test_words_flow_into_metadata_and_report(cfg):
    cfg.dataset.target_words = 1_500
    pipe = Pipeline(cfg, show_progress=False)
    pipe.run()
    metas = [pipe.store.load_metadata(c) for c in pipe.store.conversation_ids()]
    assert all(m.words_a and m.words_b for m in metas)
    summary = pipe.validate()["summary"]
    assert summary["passed_words"] == sum(m.words_a + m.words_b for m in metas)
    assert summary["words_heard_a"] + summary["words_spoken_b"] == summary["passed_words"]
    manifest = pipe.manifest()
    assert manifest["totals"]["words"] == summary["passed_words"]


def test_downstream_rejections_do_not_count_towards_target(cfg):
    from hohere.schemas.dialogue import Rejection, RejectionReason

    cfg.dataset.target_words = 2_000
    cfg.dataset.overprovision = 3.0
    pipe = Pipeline(cfg, show_progress=False)
    pipe.generate()
    first = pipe.store.accepted_dialogue_ids()
    pipe.store.save_rejection(
        Rejection(
            conversation_id=first[0],
            stage="audio",
            reasons=[RejectionReason(code="duration", message="too long")],
        )
    )
    report = pipe.generate()  # tops up to replace the rejected dialogue's words
    assert report["counts"].get("accepted", 0) >= 1
    kept = [c for c in pipe.store.accepted_dialogue_ids() if c != first[0]]
    assert sum(sum(dialogue_words(pipe.store.load_dialogue(c)).values()) for c in kept) >= 2_000
