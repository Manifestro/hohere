from __future__ import annotations

import numpy as np
import pytest

from hohere.planning import split_manifest
from hohere.schemas.dataset import Event, EventType
from hohere.storage.local import DatasetStore, SplitManifestConflict
from hohere.synthesis import assign_voices, trim_silence
from hohere.validation.audio import check_event_consistency, validate_audio

SR = 24000


def _tone(n, amp=0.3):
    t = np.arange(n) / SR
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    ramp = min(240, n // 2)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return x


def _conv(cfg_seconds=25):
    n = SR * cfg_seconds
    a, b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    a[SR : 3 * SR] = _tone(2 * SR)
    b[4 * SR : 6 * SR] = _tone(2 * SR)
    events = [
        Event(event_id="e1", speaker="A", utterance_id="u1", event_type=EventType.NORMAL_TURN,
              start_sample=SR, end_sample=3 * SR),
        Event(event_id="e2", speaker=None, utterance_id=None, event_type=EventType.LONG_PAUSE,
              start_sample=3 * SR, end_sample=4 * SR),
        Event(event_id="e3", speaker="B", utterance_id="u2", event_type=EventType.NORMAL_TURN,
              start_sample=4 * SR, end_sample=6 * SR),
    ]  # fmt: skip
    return a, b, events


def codes(r):
    return {x.code for x in r}


def test_clean_conversation_passes(cfg):
    a, b, events = _conv()
    assert validate_audio(a, b, SR, events, cfg) == []


def test_detects_leakage_clipping_and_length(cfg):
    a, b, events = _conv()
    b[SR + 100 : SR + 2000] = 0.2  # B audio during A's turn, unannotated
    a[2 * SR : 2 * SR + 5000] = 1.0  # 0.8% of samples at full scale
    c = codes(validate_audio(a, b, SR, events, cfg))
    assert {"speaker_leakage", "clipping"} <= c
    assert "length_mismatch" in codes(validate_audio(a, b[:-1], SR, events, cfg))


def test_detects_silent_utterance_and_bounds(cfg):
    a, b, events = _conv()
    b[:] = 0
    events.append(Event(event_id="e4", speaker="A", utterance_id="u3",
                        event_type=EventType.NORMAL_TURN, start_sample=len(a) - 10,
                        end_sample=len(a) + 10))  # fmt: skip
    c = codes(validate_audio(a, b, SR, events, cfg))
    assert {"silent_utterance", "event_bounds"} <= c


def test_same_speaker_overlap_detected():
    events = [
        Event(event_id="e1", speaker="A", utterance_id="u1", event_type=EventType.NORMAL_TURN,
              start_sample=0, end_sample=100),
        Event(event_id="e2", speaker="A", utterance_id="u2", event_type=EventType.BACKCHANNEL,
              start_sample=50, end_sample=150),
    ]  # fmt: skip
    assert "same_speaker_overlap" in codes(check_event_consistency(events, 200))


def test_trim_silence_keeps_padding():
    x = np.concatenate([np.zeros(SR), _tone(SR), np.zeros(SR)]).astype(np.float32)
    y = trim_silence(x, SR, -45, 20)
    assert SR <= len(y) <= SR + 2 * int(0.03 * SR)
    assert trim_silence(np.zeros(SR, np.float32), SR, -45, 20).size == 0


def test_model_voice_fixed_user_voice_varies(cfg, mock_dialogue_factory):
    d = mock_dialogue_factory()
    v1, v2 = assign_voices(d, cfg.speech), assign_voices(d, cfg.speech)
    assert v1 == v2
    assignments = [assign_voices(mock_dialogue_factory(seed=s), cfg.speech) for s in range(40)]
    assert {a.voices["B"] for a in assignments} == {"af_heart"}
    assert {a.speeds["B"] for a in assignments} == {cfg.speech.model_speed}
    user_voices = {a.voices["A"] for a in assignments}
    assert "af_heart" not in user_voices and len(user_voices) > 5
    assert all(cfg.speech.speed_min <= a.speeds["A"] <= cfg.speech.speed_max for a in assignments)


def test_model_voice_never_in_user_pool():
    from pydantic import ValidationError

    from hohere.config.schema import HohereConfig

    with pytest.raises(ValidationError):
        HohereConfig.model_validate({"speech": {"user_voices": ["af_heart", "am_eric"]}})


def test_model_utterances_have_constant_speed(cfg, synth_runner, mock_dialogue_factory):
    _, segments = synth_runner.synthesize_dialogue(mock_dialogue_factory(turns=10))
    assert {s.speed for s in segments if s.speaker == "B"} == {cfg.speech.model_speed}
    assert {s.voice for s in segments if s.speaker == "B"} == {"af_heart"}


def test_split_manifest_is_immutable(tmp_path, plan):
    store = DatasetStore(tmp_path)
    store.save_plan(plan, split_manifest(plan))
    store.save_plan(plan, split_manifest(plan))  # identical: fine
    changed = [s.model_copy(update={"split": "test"}) for s in plan]
    with pytest.raises(SplitManifestConflict):
        store.save_plan(changed, split_manifest(changed))
