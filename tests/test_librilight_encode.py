from __future__ import annotations

import io
import json

import numpy as np
import pytest

from hohere.librilight.activity import SAMPLE_RATE, SILENCE_CODES
from hohere.librilight.encode import (
    _Progress,
    channel,
    cut,
    manifest_line,
    numbering,
    pack_batches,
    tokens,
)
from hohere.librilight.ids import segment_id

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
        source="medium/0007/book-01/a.flac",
        speaker="0007",
        frames=187,
        sha256="ab" * 32,
        codec_sha256="cd" * 32,
        start=1.23456,
        end=14.5,
    )
    assert list(line) == KEYS
    assert line["skill"] == "single_speech"
    assert line["family"] == "librilight/0007" and line["user_voice"] == "0007"
    assert line["dataset"] == "libri-light" and line["dataset_config"] == "medium"
    assert line["dataset_revision"] == "md5:c75e7ac62471bfbf2db77528d62a9b74"
    assert line["tokens_path"] == "tokens/0007-book_01_a-3.pt"
    assert (line["start_seconds"], line["end_seconds"]) == (1.235, 14.5)


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
