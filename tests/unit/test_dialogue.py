from __future__ import annotations

import json

import pytest

from hohere.generators.base import normalize_raw_dialogue
from hohere.generators.qwen import QwenGenerator, extract_json
from hohere.schemas.dialogue import Skill, TurnEvent
from hohere.validation.dialogue import DuplicateIndex, validate_dialogue
from tests.conftest import make_spec


def codes(reasons):
    return {r.code for r in reasons}


@pytest.mark.parametrize("skill", list(Skill))
def test_mock_dialogues_pass_validation(cfg, mock_dialogue_factory, skill):
    d = mock_dialogue_factory(skill, turns=10)
    assert validate_dialogue(d, cfg) == []


def _raw(turns):
    return {"objective": "x", "roles": {"A": "a", "B": "b"}, "turns": turns}


LONG = "well I was thinking that we could go to the lake on saturday and bring some food"


def _base_turns():
    return [
        {"id": "t1", "speaker": "A", "text": LONG},
        {
            "id": "t2",
            "speaker": "B",
            "text": "That sounds great, which lake were you thinking about?",
        },
        {"id": "t3", "speaker": "A", "text": "The one near the old mill, about an hour north."},
        {
            "id": "t4",
            "speaker": "B",
            "text": "Okay, I can drive if you bring the sandwiches and drinks.",
        },
        {"id": "t5", "speaker": "A", "text": "Deal, let's leave at nine so we get a good spot."},
    ]


def _dialogue(turns, skill=Skill.SIMPLE_DIALOGUE):
    return normalize_raw_dialogue(_raw(turns), make_spec(skill), {"provider": "test"})


def test_refs_are_remapped(cfg):
    turns = _base_turns()
    turns.insert(
        1, {"id": "t9", "speaker": "B", "text": "mm-hm", "event": "backchannel", "ref": "t1"}
    )
    d = _dialogue(turns)
    assert d.turns[1].ref == d.turns[0].id == "conv_000001_u001"


def test_one_speaker_rejected(cfg):
    turns = [dict(t, speaker="A") for t in _base_turns()]
    assert "speaker_count" in codes(validate_dialogue(_dialogue(turns), cfg))


def test_unknown_and_forward_refs(cfg):
    turns = _base_turns()
    turns[2]["ref"] = "t99"
    turns[3]["ref"] = "t5"
    c = codes(validate_dialogue(_dialogue(turns), cfg))
    assert {"unknown_ref", "forward_ref"} <= c


def test_backchannel_rules(cfg):
    turns = _base_turns()
    turns.insert(
        1,
        {
            "id": "bc",
            "speaker": "A",
            "text": "yeah right okay sure fine",
            "event": "backchannel",
            "ref": "t1",
        },
    )
    c = codes(validate_dialogue(_dialogue(turns), cfg))
    assert {"backchannel_same_speaker", "backchannel_too_long"} <= c


def test_interruption_must_target_previous_turn(cfg):
    turns = _base_turns()
    turns[4] = {
        "id": "t5",
        "speaker": "A",
        "text": "Wait, sorry, let me stop you there.",
        "event": "interruption",
        "ref": "t2",
    }
    c = codes(validate_dialogue(_dialogue(turns, Skill.INTERRUPTION), cfg))
    assert "interruption_not_adjacent" in c


def test_skill_requirement(cfg):
    c = codes(validate_dialogue(_dialogue(_base_turns(), Skill.BACKCHANNEL), cfg))
    assert "skill_requirement" in c


def test_monologue_with_backchannels_rejected(cfg):
    # Speaker A holds the floor the whole time; B only says "mm-hm".
    turns = []
    for i in range(5):
        turns.append({"id": f"a{i}", "speaker": "A", "text": f"{LONG} and then part {i}"})
        turns.append(
            {"id": f"b{i}", "speaker": "B", "text": "mm-hm", "event": "backchannel", "ref": f"a{i}"}
        )
    c = codes(validate_dialogue(_dialogue(turns, Skill.BACKCHANNEL), cfg))
    assert {"speaker_imbalance", "monologue_run"} <= c


def test_balance_thresholds_are_configurable(cfg):
    turns = _base_turns()
    turns[1]["text"] = "Which lake?"
    turns[3]["text"] = "Sure."
    assert "speaker_imbalance" in codes(validate_dialogue(_dialogue(turns), cfg))
    cfg.dialogue.min_speaker_word_share = 0.0
    assert "speaker_imbalance" not in codes(validate_dialogue(_dialogue(turns), cfg))


def test_markup_and_length(cfg):
    turns = _base_turns()
    turns[1]["text"] = "*laughs* " + " ".join(["word"] * 80)
    c = codes(validate_dialogue(_dialogue(turns), cfg))
    assert {"non_spoken_text", "utterance_too_long"} <= c


def test_duplicate_index(mock_dialogue_factory):
    idx = DuplicateIndex(0.6)
    a = mock_dialogue_factory(seed=1)
    idx.add(a)
    b = a.model_copy(update={"conversation_id": "conv_000002"})
    assert idx.check(b).code == "exact_duplicate"
    near = b.model_copy(deep=True)
    near.turns[0].text = near.turns[0].text + " indeed"
    assert idx.check(near).code == "near_duplicate"
    assert idx.check(mock_dialogue_factory(seed=2)) is None


def test_extract_json_tolerates_wrappers():
    payload = {"turns": [{"id": "t1"}]}
    text = "<think>hmm</think>\n```json\n" + json.dumps(payload) + "\n```"
    assert extract_json(text) == payload
    assert extract_json("Sure! " + json.dumps(payload) + " hope it helps") == payload


class _FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        msg = type("M", (), {"content": self.content})
        choice = type("C", (), {"message": msg})
        return type("R", (), {"choices": [choice], "model": "Qwen/test"})


def test_qwen_generator_parses_response(cfg, monkeypatch):
    monkeypatch.setenv("HOHERE_QWEN_BASE_URL", "http://localhost:1/v1")
    gen = QwenGenerator(cfg.dialogue)
    fake = _FakeCompletions(json.dumps(_raw(_base_turns())))
    gen.client = type("Client", (), {"chat": type("Chat", (), {"completions": fake})})
    d = gen.generate(make_spec(Skill.SIMPLE_DIALOGUE), attempt=1)
    assert d.generator == {"provider": "qwen", "model": "Qwen/test", "attempt": "1"}
    assert len(d.turns) == 5 and d.turns[0].event is TurnEvent.NORMAL
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert validate_dialogue(d, cfg) == []
