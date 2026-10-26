from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from hohere.bench.build import (
    SEEN_VOICES,
    UNSEEN_VOICES,
    build_rows,
    compose,
    write_bench,
)
from hohere.bench.probes import ECHO, INFER, build_probes
from hohere.synthesis.mock import MockSynthesizer


def test_probe_set_is_deterministic_and_unique():
    probes = build_probes()
    assert probes == build_probes()
    assert len(probes) == 70 * len(ECHO) + 9 * len(INFER)
    assert len({p.pair_id for p in probes}) == len(probes)
    keys = Counter((p.question_x, p.question_y) for p in probes)
    assert max(keys.values()) == 1


def test_echo_pairs_differ_only_in_the_slot():
    for p in build_probes():
        if p.kind != "echo":
            continue
        assert p.slot_x != p.slot_y
        assert p.question_x != p.question_y and p.answer_x != p.answer_y
        assert p.question_x.replace(p.slot_x, "#").replace(p.slot_x.capitalize(), "#") == (
            p.question_y.replace(p.slot_y, "#").replace(p.slot_y.capitalize(), "#")
        )
        assert p.slot_x.lower() in p.answer_x.lower() and p.slot_y.lower() in p.answer_y.lower()


def test_voices_held_out():
    from hohere.config.schema import SpeechConfig

    sc = SpeechConfig()
    assert set(SEEN_VOICES) <= set(sc.user_voices)
    assert not set(UNSEEN_VOICES) & set(sc.all_voices)


def test_rows_share_timing_across_conditions():
    probes = build_probes(pairs_per_echo_category=2)[:3]
    rows = build_rows(probes, MockSynthesizer(24000), ["af_sarah", "af_river"])
    assert len(rows) == 6
    import io

    import soundfile as sf

    row = dict(rows[0])
    for k in ("question_x", "question_y", "answer_x", "answer_y"):
        row[f"{k}_audio"], _ = sf.read(io.BytesIO(row[f"{k}_audio"]["bytes"]), dtype="float32")
    start = row["answer_start_sample"]
    for q in "xy":
        for a in "xy":
            ch_a, ch_b = compose(row, q, a)
            assert len(ch_a) == len(ch_b) == row["total_samples"]
            assert not np.any(ch_b[:start]), "assistant must be silent during the question"
            assert not np.any(ch_a[start:]), "user must be silent during the answer"
            assert np.any(ch_b[start : start + 2400])


def test_write_bench(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("datasets")
    probes = build_probes(pairs_per_echo_category=2)
    rows = build_rows(probes, MockSynthesizer(24000), ["af_sarah"])
    summary = write_bench(rows, tmp_path, "org/bench")
    assert summary["pairs"] == len(probes)
    assert (tmp_path / "data" / "test-00000-of-00001.parquet").exists()
    assert "Evaluation only" in (tmp_path / "README.md").read_text()
