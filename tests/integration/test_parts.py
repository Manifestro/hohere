"""A second part of a dataset: own id prefix, own personas, no duplicates of part 1."""

from __future__ import annotations

import json

from hohere.generators.prompts import build_user_prompt
from hohere.pipeline import Pipeline
from hohere.planning import build_plan
from hohere.validation.dialogue import dialogue_text, load_dedup_texts


def test_id_prefix(cfg):
    cfg.dataset.id_prefix = "conv2"
    ids = [s.conversation_id for s in build_plan(cfg)]
    assert ids and all(i.startswith("conv2_") for i in ids)


def test_personas_from_config(cfg):
    spec = build_plan(cfg)[0]
    prompt = build_user_prompt(spec, personas=["a pirate who only talks about ships"])
    assert "a pirate who only talks about ships" in prompt


def test_dedup_against_earlier_part(cfg, tmp_path):
    cfg.dataset.target_hours = 0.02
    first = Pipeline(cfg, show_progress=False)
    first.generate()
    texts = load_dedup_texts(first.store.root)
    assert texts
    # The same dialogues under other ids, as if they had been published in part 1.
    part1 = tmp_path / "part1.jsonl"
    part1.write_text(
        "".join(json.dumps({"id": f"old_{cid}", "text": t}) + "\n" for cid, t in texts.items())
    )
    assert load_dedup_texts(part1) == {f"old_{c}": t for c, t in texts.items()}

    cfg.dataset.output_dir = tmp_path / "part2"
    cfg.dialogue.dedup_against = [part1]
    second = Pipeline(cfg, show_progress=False)
    report = second.generate()
    reasons = report["rejection_reasons"]
    assert reasons.get("exact_duplicate", 0) + reasons.get("near_duplicate", 0) > 0
    for cid in second.store.accepted_dialogue_ids():
        assert dialogue_text(second.store.load_dialogue(cid)) not in texts.values()


def test_dedup_against_rejects_shared_ids(cfg, tmp_path):
    cfg.dataset.target_hours = 0.02
    first = Pipeline(cfg, show_progress=False)
    first.generate()
    cfg.dataset.output_dir = tmp_path / "part2"
    cfg.dialogue.dedup_against = [first.store.root]
    try:
        Pipeline(cfg, show_progress=False).generate()
    except ValueError as e:
        assert "id_prefix" in str(e)
    else:
        raise AssertionError("shared conversation ids were not detected")
