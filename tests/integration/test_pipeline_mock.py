"""End-to-end pipeline on mock providers: offline, fast, always run."""

from __future__ import annotations

import io
import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from hohere.pipeline import Pipeline
from hohere.storage.manifest import verify_manifest


@pytest.fixture
def pipe(cfg):
    return Pipeline(cfg, show_progress=False)


def test_full_pipeline_produces_valid_dataset(pipe):
    results = pipe.run()
    summary = results["validate"]
    assert summary["conversations"] > 0
    assert summary["failed"] == 0
    store = pipe.store
    for cid in store.conversation_ids():
        a, b, sr = store.load_channels(cid)
        assert sr == 24000 and a.shape == b.shape
        meta = store.load_metadata(cid)
        assert meta.duration_samples == a.shape[0]
        assert meta.quality_status == "passed"
    kinds = set(summary["event_counts"])
    assert {"normal_turn", "backchannel", "interruption", "cutoff", "short_pause"} <= kinds


def test_resume_does_not_duplicate(pipe):
    pipe.generate(limit=3)
    pipe.synthesize(limit=2)
    first = set(pipe.store.accepted_dialogue_ids())
    report = pipe.generate()
    assert report["counts"]["skipped"] == len(first)
    assert set(pipe.store.accepted_dialogue_ids()) >= first
    pipe.synthesize()
    pipe.compose()
    again = pipe.run()
    for stage in ("generate", "synthesize", "compose"):
        counts = again[stage]["counts"]
        assert counts.get("accepted", 0) == counts.get("synthesized", 0) == 0
        assert counts.get("passed", 0) == 0
    ids = pipe.store.conversation_ids()
    assert len(ids) == len(set(ids)) == len(pipe.store.accepted_dialogue_ids())


def test_recompose_is_bit_identical(pipe):
    pipe.run()
    store = pipe.store
    cid = store.conversation_ids()[0]
    d = store.conversation_dir(cid)
    before = {n: (d / n).read_bytes() for n in ("channel_a.wav", "channel_b.wav", "events.json")}
    shutil.rmtree(d)
    pipe.compose()
    after = {n: (d / n).read_bytes() for n in before}
    assert before == after


def test_manifest_checksums(pipe):
    pipe.run()
    manifest = pipe.manifest()
    assert manifest["totals"]["conversations"] == len(pipe.store.conversation_ids())
    assert verify_manifest(pipe.store.root / "conversations", manifest) == []
    # tamper with one file
    cid = manifest["conversations"][0]["id"]
    p = pipe.store.conversation_dir(cid) / "events.json"
    p.write_text(p.read_text() + " ")
    assert verify_manifest(pipe.store.root / "conversations", manifest)


def test_rejections_are_persisted(pipe, cfg):
    cfg.dialogue.max_duration_seconds = 21  # forces most mock dialogues to be rejected
    pipe.generate()
    rejected = list((pipe.store.root / "dialogues" / "rejected").glob("*.json"))
    assert rejected
    data = json.loads(rejected[0].read_text())
    assert len(data) == cfg.dialogue.max_attempts
    assert all(r["reasons"] and r["payload"] for r in data)
    # exhausted conversations are not retried on resume
    assert pipe.generate()["counts"].get("rejected", 0) == 0


@pytest.mark.hf
def test_export_is_loadable_hf_dataset(pipe, tmp_path):
    datasets = pytest.importorskip("datasets")
    pipe.run()
    out = tmp_path / "export"
    manifest = pipe.export(out, include_text=False)
    ds = datasets.load_dataset(str(out))
    assert set(ds) <= {"train", "validation", "test"}
    assert sum(len(s) for s in ds.values()) == manifest["totals"]["conversations"]
    split = next(iter(ds.values()))
    assert isinstance(split.features["channel_a"], datasets.Audio)
    row = split.cast_column("channel_a", datasets.Audio(decode=False)).cast_column(
        "channel_b", datasets.Audio(decode=False)
    )[0]
    a, sr = sf.read(io.BytesIO(row["channel_a"]["bytes"]), dtype="int16")
    b, _ = sf.read(io.BytesIO(row["channel_b"]["bytes"]), dtype="int16")
    assert sr == 24000 and a.shape == b.shape == (row["duration_samples"],)
    assert np.any(a) and np.any(b)
    assert all(e["text"] is None for e in row["events"])
    card = (out / "README.md").read_text()
    assert "configs:" in card and "data/train-*.parquet" in card
