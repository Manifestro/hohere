from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest
import soundfile as sf

from hohere.pipeline import Pipeline
from hohere.publishing.card_assets import timeline_svg, write_card_assets


def test_card_assets_and_card(cfg, tmp_path):
    pytest.importorskip("datasets")
    cfg.publishing.repository_id = "org/hohere-test"
    pipe = Pipeline(cfg, show_progress=False)
    pipe.run()
    out = tmp_path / "export"
    pipe.export(out)
    svg = (out / "assets" / "timeline.svg").read_text()
    root = ET.fromstring(svg)  # well-formed
    assert root.tag.endswith("svg") and svg.count("<rect") > 10
    info = sf.info(out / "assets" / "sample_stereo.wav")
    assert info.channels == 2 and info.samplerate == 24000 and info.duration <= 40.5
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0")
    assert "<audio controls" in card and "assets/timeline.svg" in card
    assert "img.shields.io" in card and "hohere-test" in card


def test_timeline_svg_without_events():
    svg = timeline_svg([], 24000, 10.0)
    ET.fromstring(svg)


def test_no_assets_without_passed_conversations(cfg, tmp_path):
    pipe = Pipeline(cfg, show_progress=False)
    assert write_card_assets(pipe.store, [], tmp_path) is None
