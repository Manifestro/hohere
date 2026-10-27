"""hohere-bench suites E2/E4/E5/E6 and frame labels, built on a mock pipeline store."""

from __future__ import annotations

import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from hohere.bench.suites import (
    EVAL_SEEN,
    EVAL_UNSEEN,
    bench_card,
    interruption_rows,
    long_turn_rows,
    memory_items,
    memory_rows,
    single_turn_rows,
    write_suite,
)
from hohere.config.schema import SpeechConfig, load_config
from hohere.labels import store_labels, write_labels
from hohere.pipeline import Pipeline
from hohere.synthesis.mock import MockSynthesizer
from tests.conftest import ROOT


def _audio(cell):
    a, _ = sf.read(io.BytesIO(cell["bytes"]), dtype="float32")
    return a


@pytest.fixture
def e1_store(cfg):
    cfg.dataset.target_hours = 0.15
    pipe = Pipeline(cfg, show_progress=False)
    pipe.generate()
    pipe.synthesize()
    pipe.compose()
    return pipe.store


def test_eval_voices_are_held_out():
    sc = SpeechConfig()
    assert set(EVAL_SEEN) <= set(sc.user_voices)
    assert not set(EVAL_UNSEEN) & set(sc.all_voices)
    for name in ("v0.3", "v0.4"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert not set(EVAL_UNSEEN) & set(cfg.speech.all_voices)
        e1 = load_config(ROOT / "configs" / "bench-e1.yaml")
        assert not set(e1.dataset.topics) & set(cfg.dataset.topics)


def test_store_labels(e1_store, tmp_path):
    rows = store_labels(e1_store)
    assert rows and all(len(r["activity_a"]) == r["num_frames"] for r in rows)
    stats = write_labels(rows, tmp_path / "labels", {"test": True})
    assert stats["conversations"] == len(rows)
    t = pq.read_table(tmp_path / "labels" / "labels-00000-of-00001.parquet")
    assert t.num_rows == len(rows)
    assert json.loads((tmp_path / "labels" / "stats.json").read_text())["counts"]


def test_single_turn(e1_store, tmp_path):
    rows = single_turn_rows(e1_store)
    assert rows
    for r in rows:
        a, b = _audio(r["channel_a"]), _audio(r["channel_b"])
        assert len(a) == len(b) == r["total_samples"]
        assert r["user_start_frame"] < r["answer_start_frame"] < r["answer_end_frame"]
        assert not np.any(a[r["user_end_frame"] * 1920 :])  # only the first user turn
        assert np.any(b[r["answer_start_frame"] * 1920 : r["answer_end_frame"] * 1920])
    s = write_suite(rows, "e2_single_turn", tmp_path)
    assert s["rows"] == len(rows)


def test_interruptions(e1_store, tmp_path):
    rows = interruption_rows(e1_store, MockSynthesizer(24000), control_share=0.5)
    assert rows
    assert {r["condition"] for r in rows} <= {"interrupt", "backchannel_control"}
    for r in rows:
        a = _audio(r["channel_a"])
        assert r["assistant_start_frame"] < r["interrupt_start_frame"] < r["assistant_end_frame"]
        start, end = r["interrupt_start_frame"] * 1920, r["interrupt_end_frame"] * 1920
        assert np.any(a[start:end])
        assert not np.any(a[r["assistant_start_frame"] * 1920 : start])
        assert not np.any(a[end:])
    write_suite(rows, "e5_interruptions", tmp_path)


def test_memory_items_are_consistent():
    items = memory_items(per_cell=4)
    assert items == memory_items(per_cell=4)
    for it in items:
        assert len(set(it["candidates"])) == 4
        final = it["candidate_locations"][it["correct_index"]]
        texts = [t for _, t in it["turns"]]
        if it["update"]:
            stale = it["candidate_locations"][it["stale_index"]]
            assert stale in texts[0] and final not in texts[0]
            assert any(final in t for t in texts[1:-1])
        else:
            assert final in texts[0] and it["stale_index"] is None
        n_exchanges = (len(texts) - 1) // 2
        assert n_exchanges == 1 + it["distance"] + int(it["update"])
        assert texts[-1].endswith("?") and it["turns"][-1][0] == "A"
        assert "{" not in " ".join(texts + it["candidates"])


def test_memory_rows_layout(tmp_path):
    items = memory_items(per_cell=1)[:3]
    rows = memory_rows(items, MockSynthesizer(24000))
    assert len(rows) == 6
    assert {r["voice_a_seen_in_training"] for r in rows} == {True, False}
    for r in rows:
        b = _audio(r["channel_b"])
        assert not np.any(b[r["answer_start_sample"] :])  # answer slot is empty
        for i in range(4):
            assert len(_audio(r[f"candidate_{i}_audio"])) > 0
        assert r["answer_start_sample"] + max(
            len(_audio(r[f"candidate_{i}_audio"])) for i in range(4)
        ) <= r["total_samples"]
    write_suite(rows, "e4_memory", tmp_path)


def test_long_turns_and_card(tmp_path):
    items = [
        {"item_id": f"e6-{k:03d}", "topic": "t", "persona": "p", "model": "m",
         "sentences": [f"This is sentence number {i} of a fairly long story." for i in range(8)]}
        for k in range(3)
    ]
    rows = long_turn_rows(items, MockSynthesizer(24000), EVAL_SEEN + EVAL_UNSEEN)
    assert rows
    for r in rows:
        assert len(r["pause_start_frames"]) == 7
        assert not np.any(_audio(r["channel_b"]))
        a = _audio(r["channel_a"])
        for s, e in zip(r["pause_start_frames"], r["pause_end_frames"], strict=True):
            assert not np.any(a[(s + 1) * 1920 : (e - 1) * 1920])
    write_suite(rows, "e6_long_turns", tmp_path)
    readme = bench_card(tmp_path, "manifestro/hohere-bench")
    assert "config_name: e6_long_turns" in readme and "config_name: e4_memory" not in readme
