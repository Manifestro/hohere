from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from hohere.config.schema import HohereConfig
from hohere.pipeline import Pipeline
from hohere.synthesis import AudioCache, SynthesisError, SynthesisRunner
from hohere.synthesis.chatterbox import ChatterboxSynthesizer
from hohere.synthesis.mock import MockSynthesizer
from hohere.synthesis.router import VoiceRouter

FAKE_WORKER = Path(__file__).parents[1] / "fixtures" / "fake_tts_worker.py"


@pytest.fixture
def ref_wav(tmp_path):
    import soundfile as sf

    p = tmp_path / "ref.wav"
    sf.write(p, np.zeros(24000, np.float32), 24000)
    return p


def test_chatterbox_worker_protocol(ref_wav):
    tts = ChatterboxSynthesizer(ref_wav, worker_cmd=[sys.executable, str(FAKE_WORKER)])
    a = tts.synthesize("hello there friend", "dauys", 1.0)
    b = tts.synthesize("hi", "dauys", 1.0)
    assert a.dtype == np.float32 and a.shape[0] == 3 * 2400 and b.shape[0] == 2400
    with pytest.raises(SynthesisError, match="boom"):
        tts.synthesize("FAIL", "dauys", 1.0)
    assert tts.synthesize("still alive", "dauys", 1.0).size > 0  # worker survives errors
    tts.close()


def test_missing_reference_is_an_error(tmp_path):
    with pytest.raises(SynthesisError, match="reference audio not found"):
        ChatterboxSynthesizer(tmp_path / "nope.wav", worker_cmd=["true"])


class _Tagged(MockSynthesizer):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.calls: list[str] = []

    def synthesize(self, text, voice, speed):
        self.calls.append(voice)
        return super().synthesize(text, voice, speed)


def test_router_sends_model_voice_to_model_tts(cfg, mock_dialogue_factory):
    model_tts, user_tts = _Tagged("model-tts"), _Tagged("user-tts")
    router = VoiceRouter(cfg.speech.model_voice, model_tts, user_tts)
    runner = SynthesisRunner(cfg.speech, router, AudioCache(cfg.dataset.cache_dir, 24000))
    _, segments = runner.synthesize_dialogue(mock_dialogue_factory(turns=8))
    assert set(model_tts.calls) == {cfg.speech.model_voice}
    assert cfg.speech.model_voice not in user_tts.calls and user_tts.calls
    # cache keys are namespaced per backend
    b = next(s for s in segments if s.speaker == "B")
    a = next(s for s in segments if s.speaker == "A")
    assert b.cache_key != a.cache_key


def test_chatterbox_config_validation(ref_wav):
    with pytest.raises(ValidationError, match="model_reference_audio"):
        HohereConfig.model_validate({"speech": {"model_provider": "chatterbox_turbo"}})
    with pytest.raises(ValidationError, match="model_speed"):
        HohereConfig.model_validate(
            {"speech": {"model_provider": "chatterbox_turbo", "model_reference_audio": str(ref_wav),
                        "model_speed": 1.1}}
        )  # fmt: skip
    cfg = HohereConfig.model_validate(
        {"speech": {"model_provider": "chatterbox_turbo", "model_reference_audio": str(ref_wav)}}
    )
    assert cfg.speech.model_provider == "chatterbox_turbo"


def test_synthesis_shards_are_disjoint_and_complete(cfg):
    pipe = Pipeline(cfg, show_progress=False)
    pipe.generate()
    accepted = set(pipe.store.accepted_dialogue_ids())
    pipe.synthesize(shard=(0, 2))
    first = {c for c in accepted if pipe.store.has_segments(c)}
    pipe.synthesize(shard=(1, 2))
    second = {c for c in accepted if pipe.store.has_segments(c)} - first
    assert first and second and not first & second
    assert first | second == accepted
