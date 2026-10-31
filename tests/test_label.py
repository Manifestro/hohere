"""`hohere label` and `hohere listen` with fake models: real files, no torch."""

from __future__ import annotations

import json

import numpy as np
import soundfile
from test_pipeline import FakeModels, source

from hohere import label, listen, records, units, vad

SR = vad.SAMPLE_RATE


def _stages(version=True, **kwargs):
    stages = FakeModels().stages(mimi_batch=1)
    if version:
        stages.units_version = {"units_k4096": "x", "units_k8192": "y"}
    return stages.with_overrides(**kwargs)


def _speech_in_the_middle(wav):
    n = wav.shape[0]
    return [(n // 4, n // 2), (n // 2 + SR // 2, 3 * n // 4)] if n >= 4 * SR else [(0, n)]


def _write(path, seconds, rate=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, np.linspace(-0.5, 0.5, int(seconds * rate), dtype=np.float32), rate)


def test_plan_pieces_trims_short_files_and_chunks_long_ones():
    wav = np.zeros(10 * SR, dtype=np.float32)
    assert label.plan_pieces(wav, [(3 * SR, 4 * SR)]) == [(2.5, 4.5)]
    assert label.plan_pieces(wav, [(SR // 4, 10 * SR)]) == [(0.0, 10.0)]
    assert label.plan_pieces(wav, []) == [(0.0, 10.0)]
    assert label.plan_pieces(np.zeros(0, dtype=np.float32), []) == []
    long = np.zeros(200 * SR, dtype=np.float32)
    spans = [(index * 11 * SR, (index * 11 + 10) * SR) for index in range(18)]
    pieces = label.plan_pieces(long, spans)
    assert len(pieces) > 1
    assert all(30.0 <= end - start <= 60.0 for start, end in pieces)


def test_records_read_as_a_run_writes_them(tmp_path):
    _write(tmp_path / "in/wug/word.wav", 8.0)
    _write(tmp_path / "in/long.flac", 200.0)
    summary = label.run(
        tmp_path / "in", tmp_path / "out", _stages(speech=_speech_in_the_middle)
    )
    assert summary["files"] == 2
    assert summary["errors"] == {}
    lines = [json.loads(x) for x in (tmp_path / "out/manifest.jsonl").read_text().splitlines()]
    ids = {line["id"] for line in lines}
    assert "wug/word" in ids
    short = next(line for line in lines if line["id"] == "wug/word")
    assert short["start_s"] == 1.5
    assert short["end_s"] == 6.5
    assert short["units_k8192"] == "y"
    record = records.read_record((tmp_path / "out/wug/word.npz").read_bytes())
    assert record["codes"].shape == (32, short["frames"])
    assert short["n_sil"] == record["sil_start"].shape[0]
    assert units.build_m(record, "B", 8192).shape == (short["frames"],)


def test_a_restart_skips_finished_files_and_a_bad_file_stops_nothing(tmp_path):
    _write(tmp_path / "in/a.wav", 2.0)
    _write(tmp_path / "in/b.wav", 3.0)
    (tmp_path / "in/broken.wav").write_bytes(b"not audio")
    stages = _stages()
    first = label.run(tmp_path / "in", tmp_path / "out", stages)
    assert first["files"] == 2
    assert list(first["errors"]) == ["broken"]
    second = label.run(tmp_path / "in", tmp_path / "out", stages)
    assert second["files"] == 0
    assert second["skipped"] == 2
    assert len((tmp_path / "out/manifest.jsonl").read_text().splitlines()) == 2


def test_a_batch_that_fails_is_retried_file_by_file(tmp_path):
    _write(tmp_path / "in/good.wav", 2.0)
    _write(tmp_path / "in/bad.wav", 3.0)
    stages = _stages()
    fine = stages.encode_mimi

    def picky(chunks):
        if any(chunk.size == 3 * SR for chunk in chunks):
            raise RuntimeError("too much")
        return fine(chunks)

    original = label.GROUP_PIECES
    label.GROUP_PIECES = 2
    try:
        summary = label.run(tmp_path / "in", tmp_path / "out", stages.with_overrides(
            encode_mimi=picky))
    finally:
        label.GROUP_PIECES = original
    assert summary["files"] == 1
    assert list(summary["errors"]) == ["bad"]


def test_clicks_land_on_the_events():
    wav = np.zeros(SR, dtype=np.float32)
    out = listen.with_clicks(wav, np.array([10]), np.array([30]))
    assert out.shape == wav.shape
    assert np.abs(out[10 * 320 : 10 * 320 + 100]).max() > 0.1
    assert np.abs(out[30 * 320 : 30 * 320 + 100]).max() > 0.1
    assert out[: 10 * 320].max() == 0.0
    assert listen.with_clicks(wav, np.array([49]), np.zeros(0)).shape == wav.shape  # at the end


def test_labels_are_in_time_order():
    text = listen.labels(np.array([0, 50]), np.array([25, 75]), np.array([30]))
    assert text.splitlines() == [
        "0.000\t0.500\tu0", "0.600\t0.600\tSIL", "1.000\t1.500\tu1",
    ]


def test_pick_files_spreads_over_the_tier():
    assert listen.pick_files(list(range(100)), 4) == [0, 25, 50, 75]
    assert listen.pick_files([1, 2], 20) == [1, 2]
    assert listen.pick_files([], 3) == []


def test_listen_writes_the_three_files_of_each_chunk(tmp_path):
    files = [source(f"small-1-b-f{i}") for i in range(3)]
    report = listen.run(files, FakeModels().stages(), tmp_path / "listen", count=2)
    assert len(report["chunks"]) == 2
    for stem in ("000", "001"):
        assert (tmp_path / f"listen/{stem}.wav").exists()
        assert (tmp_path / f"listen/{stem}-clicks.wav").exists()
        assert (tmp_path / f"listen/{stem}.txt").read_text()
    assert json.loads((tmp_path / "listen/index.json").read_text())["units_per_s"] > 0
