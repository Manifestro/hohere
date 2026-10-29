from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pytest

from hohere.librilight import encode
from hohere.librilight.activity import FRAME_SAMPLES, SAMPLE_RATE, SILENCE_CODES
from hohere.librilight.encode import (
    Writers,
    _Progress,
    channel,
    cut,
    manifest_line,
    numbering,
    pack_batches,
    tokens,
    write_file,
)
from hohere.librilight.ids import segment_id
from hohere.librilight.plan import PlanFile

KEYS = [
    "id",
    "source",
    "skill",
    "family",
    "user_voice",
    "frames",
    "tokens_path",
    "sha256",
    "dataset",
    "dataset_revision",
    "dataset_config",
    "dataset_split",
    "codec_sha256",
    "start_seconds",
    "end_seconds",
]


def test_cut_rounds_to_the_nearest_sample_and_clamps() -> None:
    audio = np.arange(SAMPLE_RATE * 2, dtype=np.float32)
    assert cut(audio, 0.5, 1.0).tolist() == audio[SAMPLE_RATE // 2 : SAMPLE_RATE].tolist()
    assert cut(audio, -1.0, 0.001).shape == (24,)
    assert cut(audio, 1.0, 1.0).shape == (0,)
    assert cut(audio, 10.0, 20.0).shape == (0,)


def test_numbering_skips_dropped_segments_and_numbers_in_time_order() -> None:
    kept = numbering([5, 0, 2])  # segments 1, 3 and 4 were dropped
    assert kept == [(0, 0), (1, 2), (2, 5)]
    ids = [segment_id("0007", "book_01_a", number) for number, _ in kept]
    assert ids == ["0007-book_01_a-0", "0007-book_01_a-1", "0007-book_01_a-2"]


def test_channel_is_stable_and_splits_segments_between_a_and_b() -> None:
    ids = [segment_id("0007", "book_01_a", i) for i in range(200)]
    sides = [channel(i) for i in ids]
    assert sides == [channel(i) for i in ids]
    assert set(sides) == {"A", "B"}
    assert 60 <= sides.count("A") <= 140


def test_tokens_put_the_speech_on_one_channel_only() -> None:
    codes = np.arange(len(SILENCE_CODES) * 5, dtype=np.int16).reshape(8, 5)
    silence = [[code] * 5 for code in SILENCE_CODES]
    act = np.ones(5, np.uint8)
    on_a = tokens(codes, act, "A")
    assert set(on_a) == {"user", "target", "user_activity", "activity"}
    assert on_a["user"] is codes and on_a["user_activity"] is act
    assert on_a["target"].tolist() == silence and on_a["activity"].tolist() == [0] * 5
    on_b = tokens(codes, act, "B")
    assert on_b["target"] is codes and on_b["activity"] is act
    assert on_b["user"].tolist() == silence


def test_manifest_line_has_the_exact_keys() -> None:
    line = manifest_line(
        id="0007-book_01_a-3",
        speaker="0007",
        frames=187,
        sha256="ab" * 32,
        codec_sha256="cd" * 32,
        start=1.23456,
        end=14.5,
    )
    assert list(line) == KEYS
    assert line["source"] == f"1a_on_{channel('0007-book_01_a-3')}"
    assert line["skill"] == "single_speech"
    assert line["family"] == "librilight/0007" and line["user_voice"] == "0007"
    assert line["dataset"] == "libri-light" and line["dataset_config"] == "medium"
    assert line["dataset_revision"] == "md5:c75e7ac62471bfbf2db77528d62a9b74"
    assert line["tokens_path"] == "tokens/0007-book_01_a-3.pt"
    assert (line["start_seconds"], line["end_seconds"]) == (1.235, 14.5)


def plan(chapter: str, segments: tuple[tuple[float, float], ...]) -> PlanFile:
    return PlanFile(
        speaker="0007",
        book_dir="book-01",
        stem=chapter.rsplit("_", 1)[1],
        chapter=chapter,
        flac_offset=1024,
        flac_size=1000,
        duration=12.0,
        snr=12.5,
        segments=segments,
    )


def test_manifest_source_matches_the_channel_that_holds_the_speech() -> None:
    codes = np.arange(len(SILENCE_CODES) * 3, dtype=np.int16).reshape(8, 3)
    act = np.ones(3, np.uint8)
    sides = set()
    for index in range(12):
        sid = segment_id("0007", "book_01_a", index)
        line = manifest_line(
            id=sid,
            speaker="0007",
            frames=3,
            sha256="ab" * 32,
            codec_sha256="cd" * 32,
            start=0.0,
            end=0.24,
        )
        side = line["source"].removeprefix("1a_on_")
        sides.add(side)
        assert side == channel(sid)
        payload = tokens(codes, act, side)
        speech, silent = ("user", "target") if side == "A" else ("target", "user")
        assert payload[speech] is codes and payload[silent].tolist() != codes.tolist()
    assert sides == {"A", "B"}  # both layouts are covered


def test_write_file_saves_every_token_then_the_marker(tmp_path: Path) -> None:
    file = PlanFile(
        speaker="0007",
        book_dir="book-01",
        stem="a",
        chapter="book_01_a",
        flac_offset=1024,
        flac_size=1000,
        duration=20.0,
        snr=12.5,
        segments=((0.5, 5.0), (6.0, 12.0), (13.0, 20.0)),
    )
    tokens_dir, done_dir = tmp_path / "tokens", tmp_path / "done"
    tokens_dir.mkdir()
    done_dir.mkdir()
    marker = done_dir / "book_01_a.json"
    saved: list[str] = []

    def save(path: Path, payload: dict) -> str:
        assert not marker.exists()  # the marker is the last write of a chapter
        saved.append(path.name)
        path.write_bytes(b"pt")
        return "ab" * 32

    codes = np.zeros((8, 4), np.int16) + np.arange(8, dtype=np.int16)[:, None]
    payloads = {0: (codes, np.ones(4, np.uint8)), 2: (codes, np.zeros(4, np.uint8))}
    lines = write_file(
        file,
        payloads,
        tokens_dir=tokens_dir,
        done_dir=done_dir,
        codec_sha256="cd" * 32,
        save=save,
    )
    assert saved == ["0007-book_01_a-0.pt", "0007-book_01_a-1.pt"]  # segment 1 dropped
    assert [line["id"] for line in lines] == ["0007-book_01_a-0", "0007-book_01_a-1"]
    assert list(lines[0]) == KEYS and lines[1]["end_seconds"] == 20.0
    assert json.loads(marker.read_text()) == lines
    assert not (done_dir / "book_01_a.tmp").exists()
    assert (
        write_file(
            file, {}, tokens_dir=tokens_dir, done_dir=done_dir, codec_sha256="cd" * 32, save=save
        )
        == []
    )  # a file without kept segments still gets its marker
    assert marker.exists()


def test_writers_run_every_job_and_bound_the_queue() -> None:
    seen: list[int] = []
    with Writers(workers=2, pending=2) as writers:
        for i in range(10):
            writers.submit(lambda i=i: seen.append(i))
            assert len(writers.pending) <= 2
        writers.wait()
        assert writers.done == 10
    assert sorted(seen) == list(range(10))


def test_a_failed_write_is_raised_instead_of_lost() -> None:
    def boom() -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"), Writers(workers=1) as writers:
        writers.submit(boom)
        writers.wait()


def test_a_full_queue_cannot_hide_a_failed_write() -> None:
    def boom() -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"), Writers(workers=1, pending=1) as writers:
        writers.submit(boom)
        for _ in range(50):  # `submit` harvests the finished failure
            writers.submit(lambda: None)
        writers.wait()


def test_run_encode_writes_every_file_and_waits_for_the_writers(tmp_path, monkeypatch) -> None:
    files = [
        plan("book_01_a", ((0.0, 8.0),)),  # a tone with no pause: the segment is dropped
        plan("book_01_b", ((0.0, 12.0),)),
    ]
    speech = np.zeros(12 * SAMPLE_RATE, np.float32)
    speech[: 4 * SAMPLE_RATE] = 0.5  # speech, 2 s of silence, speech again
    speech[6 * SAMPLE_RATE :] = 0.5
    stream = iter([(0, files[0], np.full(8 * SAMPLE_RATE, 0.5, np.float32)), (1, files[1], speech)])
    writes: list[str] = []

    def fake_write_file(file, payloads, **kwargs) -> list:
        time.sleep(0.05)  # slow enough that a run that forgot to wait would miss it
        writes.append(file.chapter)
        return []

    def fake_encoded(mimi, audios, device) -> list:
        return [np.zeros((8, max(1, len(a) // FRAME_SAMPLES)), np.int16) for a in audios]

    monkeypatch.setattr(encode, "read_plan", lambda path: files)
    monkeypatch.setattr(encode, "assign_plan_parts", lambda fs, n: [list(fs)])
    monkeypatch.setattr(encode, "resolve_checkpoint", lambda name: tmp_path / "mimi.safetensors")
    monkeypatch.setattr(encode, "sha256_file", lambda path: "cd" * 32)
    monkeypatch.setattr(encode, "load_mimi", lambda path, device: object())
    monkeypatch.setattr(encode, "check_silence", lambda mimi, device: None)
    monkeypatch.setattr(encode, "RangeReader", lambda url: object())
    monkeypatch.setattr(encode, "encode_batch", fake_encoded)
    monkeypatch.setattr(encode, "write_file", fake_write_file)
    monkeypatch.setattr(encode, "_decoded", lambda todo, reader, downloads, decodes: stream)

    result = encode.run_encode(
        plan=tmp_path / "plan.jsonl", part=0, parts=1, out=tmp_path / "part-0", checkpoint="m"
    )
    assert sorted(writes) == ["book_01_a", "book_01_b"]  # every file, kept or not
    assert (result["done"], result["segments"], result["dropped"]) == (2, 1, {"no_pause": 1})


def test_pack_batches_respects_items_and_padded_frames() -> None:
    fifteen = round(15 * SAMPLE_RATE)  # 188 padded frames
    assert [len(b) for b in pack_batches([fifteen] * 40, 16, 3000)] == [15, 15, 10]
    assert [len(b) for b in pack_batches([fifteen] * 40, 16, 3200)] == [16, 16, 8]
    thirty = round(30 * SAMPLE_RATE)
    assert [len(b) for b in pack_batches([thirty] * 9, 16, 3000)] == [8, 1]  # 375 frames each


def test_pack_batches_sorts_by_length_and_keeps_long_items_alone() -> None:
    lengths = [round(s * SAMPLE_RATE) for s in (9.0, 2.0, 5.0)]
    batches = pack_batches(lengths, batch_items=3, max_padded_frames=1000)
    assert [lengths[i] for i in batches[0]] == sorted(lengths)
    assert len(pack_batches([round(40 * SAMPLE_RATE)], 16, 3000)) == 1
    with pytest.raises(ValueError, match="batch_items"):
        pack_batches([], 0, 3000)


def test_progress_prints_one_json_line_every_30_seconds() -> None:
    clock = {"t": 0.0}
    out = io.StringIO()
    progress = _Progress(total=4, done=1, clock=lambda: clock["t"], stream=out)
    progress.tick(segments=10, seconds=600.0, dropped={"silent": 1})
    assert out.getvalue() == ""
    clock["t"] = 3600.0
    progress.tick(segments=20, seconds=1200.0, dropped={"silent": 2})
    line = json.loads(out.getvalue())
    assert line["files_done"] == 1 and line["segments"] == 20
    assert line["audio_hours_per_wall_hour"] == 0.33
    assert line["dropped"] == {"silent": 2}
