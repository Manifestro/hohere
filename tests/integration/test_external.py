"""Opt-in tests against real services. Run on the GPU pod with:

    HOHERE_RUN_EXTERNAL=1 HOHERE_QWEN_BASE_URL=http://localhost:8000/v1 \
    HOHERE_TEST_HF_REPO=<namespace>/hohere-test uv run pytest -m external
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hohere.schemas.dialogue import Skill
from hohere.validation.dialogue import validate_dialogue
from tests.conftest import external_enabled, make_spec

pytestmark = [
    pytest.mark.external,
    pytest.mark.skipif(not external_enabled(), reason="set HOHERE_RUN_EXTERNAL=1"),
]


@pytest.mark.parametrize("skill", [Skill.BACKCHANNEL, Skill.INTERRUPTION])
def test_qwen_generates_valid_dialogue(cfg, skill):
    from hohere.generators.qwen import QwenGenerator

    if not (cfg.dialogue.base_url or os.environ.get("HOHERE_QWEN_BASE_URL")):
        pytest.skip("no Qwen endpoint configured")
    cfg.dialogue.model = os.environ.get("HOHERE_QWEN_MODEL", cfg.dialogue.model)
    gen = QwenGenerator(cfg.dialogue)
    reasons = []
    for attempt in range(3):
        d = gen.generate(make_spec(skill, turns=10), attempt=attempt)
        reasons = validate_dialogue(d, cfg)
        if not reasons:
            break
    assert reasons == []


@pytest.mark.kokoro
def test_kokoro_synthesizes_24k_mono(cfg):
    pytest.importorskip("kokoro")
    from hohere.synthesis.kokoro import KokoroSynthesizer

    tts = KokoroSynthesizer(cfg.speech)
    audio = tts.synthesize("Hello there, this is a quick test.", "af_heart", 1.0)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert 0.8 * 24000 < audio.shape[0] < 6 * 24000
    assert np.max(np.abs(audio)) > 0.01


@pytest.mark.hf
def test_private_publish_roundtrip(cfg, tmp_path):
    pytest.importorskip("huggingface_hub")
    repo = os.environ.get("HOHERE_TEST_HF_REPO")
    if not repo:
        pytest.skip("set HOHERE_TEST_HF_REPO to an isolated test repository")
    from hohere.pipeline import Pipeline
    from hohere.publishing.huggingface import publish_dataset

    pipe = Pipeline(cfg, show_progress=False)
    pipe.generate(limit=3)
    pipe.run()
    pipe.export(tmp_path / "export")
    result = publish_dataset(tmp_path / "export", repo, private=True)
    assert result["private"] is True
