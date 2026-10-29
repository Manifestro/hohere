from __future__ import annotations

import json
from pathlib import Path

import pytest

from hohere.librilight.files import SourceFile
from hohere.librilight.plan import (
    PlanFile,
    assign_plan_parts,
    build,
    filter_files,
    parse_line,
    read_plan,
    write_plan,
)
from hohere.librilight.tarindex import TarHeader

VAD = [(1.0, 4.0), (4.5, 9.0), (12.0, 14.0)]  # one 14 s segment, 0.5 .. 14.5


def source(speaker: str, stem: str = "a", size: int = 1000, offset: int = 1024) -> SourceFile:
    directory = f"medium/{speaker}/book-01"
    return SourceFile(
        speaker=speaker,
        book_dir="book-01",
        stem=stem,
        flac=TarHeader(f"{directory}/{stem}.flac", size, "0", offset - 512),
        json=TarHeader(f"{directory}/{stem}.json", 100, "0", offset + size),
    )


def plan_file(speaker: str, stem: str = "a", size: int = 1000, offset: int = 1024) -> PlanFile:
    entry = build(source(speaker, stem, size, offset), 20.0, 12.5, VAD)
    assert entry is not None
    return entry


def test_build_turns_meta_into_segments() -> None:
    entry = plan_file("0007")
    assert (entry.chapter, entry.flac_offset, entry.flac_size) == ("book_01_a", 1024, 1000)
    assert entry.segments == ((0.5, 14.5),)
    assert entry.archive_path == "medium/0007/book-01/a.flac"
    assert entry.key == (7, "book_01_a")


def test_build_drops_a_file_without_segments() -> None:
    assert build(source("0007"), 20.0, 0.0, []) is None
    assert build(source("0007"), 20.0, 0.0, [(1.0, 20.0)]) is None  # one long run, no pause


def test_a_line_round_trips() -> None:
    entry = plan_file("0007")
    raw = json.loads(json.dumps(entry.line()))
    assert parse_line(raw) == entry
    assert list(raw) == [
        "speaker",
        "book_dir",
        "stem",
        "chapter",
        "flac_offset",
        "flac_size",
        "duration",
        "snr",
        "segments",
    ]


def test_write_plan_sorts_speakers_numerically(tmp_path: Path) -> None:
    files = [plan_file("100"), plan_file("0007"), plan_file("99")]
    path = tmp_path / "plan.jsonl"
    assert write_plan(path, files) == 3
    assert [f.speaker for f in read_plan(path)] == ["0007", "99", "100"]


def test_filter_drops_speakers_and_recordings() -> None:
    files = [source("0007", "a"), source("0008", "a"), source("0009", "a")]
    kept, dropped = filter_files(
        files, exclude_speakers=["0008"], exclude_recordings=[["0009", "a"]]
    )
    assert [f.speaker for f in kept] == ["0007"]
    assert dropped == {"speaker": 1, "recording": 1}


def test_assign_plan_parts_keeps_a_speaker_in_one_part() -> None:
    files = [
        plan_file("0001", "a", 1000, 0),
        plan_file("0001", "b", 1000, 4096),
        plan_file("0002", "a", 600, 8192),
        plan_file("0003", "a", 400, 12288),
    ]
    parts = assign_plan_parts(files, 2)
    owners = [{f.speaker for f in part} for part in parts]
    assert owners[0].isdisjoint(owners[1])
    assert [sum(f.flac_size for f in part) for part in parts] == [2000, 1000]
    for part in parts:
        assert [f.flac_offset for f in part] == sorted(f.flac_offset for f in part)


def test_assign_plan_parts_is_deterministic_and_accepts_empty_parts() -> None:
    files = [plan_file("0002", "a", 100, 0), plan_file("0001", "a", 300, 512)]
    assert assign_plan_parts(files, 3) == assign_plan_parts(list(reversed(files)), 3)
    assert [len(part) for part in assign_plan_parts(files, 3)] == [1, 1, 0]
    with pytest.raises(ValueError, match="parts"):
        assign_plan_parts(files, 0)
