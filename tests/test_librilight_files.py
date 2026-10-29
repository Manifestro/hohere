from __future__ import annotations

import pytest

from hohere.librilight.files import SourceFile, assign_parts, pair_entries
from hohere.librilight.tarindex import TarHeader

DIR = "medium/0007/book-01"


def header(path: str, size: int = 1000, offset: int = 0) -> TarHeader:
    return TarHeader(path=path, size=size, typeflag="0", header_offset=offset)


def flac(stem: str, offset: int = 0, directory: str = DIR) -> TarHeader:
    return header(f"{directory}/{stem}.flac", offset=offset)


def meta(stem: str, offset: int = 4096, directory: str = DIR) -> TarHeader:
    return header(f"{directory}/{stem}.json", size=300, offset=offset)


def source(speaker: str, stem: str, size: int, offset: int) -> SourceFile:
    directory = f"medium/{speaker}/book-01"
    return SourceFile(
        speaker=speaker,
        book_dir="book-01",
        stem=stem,
        flac=header(f"{directory}/{stem}.flac", size=size, offset=offset),
        json=header(f"{directory}/{stem}.json", size=300, offset=offset + 512 * 3),
    )


def test_pairs_json_before_and_after_flac() -> None:
    entries = [
        flac("a"),
        flac("b"),
        meta("a"),
        meta("b"),
        meta("c"),  # JSON first
        flac("c"),
    ]
    files, unpaired = pair_entries(entries)
    assert [(f.speaker, f.book_dir, f.stem) for f in files] == [
        ("0007", "book-01", "a"),
        ("0007", "book-01", "b"),
        ("0007", "book-01", "c"),
    ]
    assert files[0].flac.header_offset == 0 and files[0].json.header_offset == 4096
    assert files[2].json.header_offset == 4096 and files[2].flac.header_offset == 0
    assert unpaired == []


def test_pairs_files_of_different_speakers_separately() -> None:
    entries = [
        flac("a", directory="medium/0007/book-01"),
        flac("a", directory="medium/0008/book-01"),
        meta("a", directory="medium/0008/book-01"),
        meta("a", directory="medium/0007/book-01"),
    ]
    files, _ = pair_entries(entries)
    assert [(f.speaker, f.stem) for f in files] == [("0007", "a"), ("0008", "a")]


def test_reports_unpaired_entries() -> None:
    lonely_flac, lonely_json = flac("x"), meta("y")
    files, unpaired = pair_entries([lonely_flac, meta("a"), flac("a"), lonely_json])
    assert [f.stem for f in files] == ["a"]
    assert unpaired == [lonely_flac, lonely_json]


def test_ignores_entries_outside_the_expected_layout() -> None:
    files, unpaired = pair_entries(
        [
            header("medium/0007/", size=0),
            header("medium/0007/book-01/notes.txt"),
            header("other/0007/book-01/a.flac"),
            header("medium/0007/a.flac"),
            flac("a"),
            meta("a"),
        ]
    )
    assert [f.stem for f in files] == ["a"]
    assert unpaired == []


def test_rejects_a_non_numeric_speaker() -> None:
    with pytest.raises(ValueError, match="speaker"):
        pair_entries([flac("a", directory="medium/abc/book-01")])


def test_assign_parts_keeps_a_speaker_in_one_part() -> None:
    files = [
        source("0002", "s1", 100, 1024),
        source("0002", "s2", 100, 4096),
        source("0003", "s1", 100, 2048),
        source("0004", "s1", 100, 512),
    ]
    parts = assign_parts(files, 2)
    owners = [{f.speaker for f in part} for part in parts]
    assert owners[0].isdisjoint(owners[1])
    assert len(owners[0] | owners[1]) == 3
    for part in parts:
        assert [f.flac.data_offset for f in part] == sorted(f.flac.data_offset for f in part)


def test_assign_parts_balances_flac_bytes() -> None:
    files = [
        source("0001", "a", 1000, 0),
        source("0002", "a", 500, 0),
        source("0003", "a", 400, 0),
        source("0004", "a", 100, 0),
    ]
    parts = assign_parts(files, 2)
    loads = [sum(f.flac.size for f in part) for part in parts]
    assert loads == [1000, 1000]
    assert {f.speaker for f in parts[0]} == {"0001"}
    assert {f.speaker for f in parts[1]} == {"0002", "0003", "0004"}


def test_assign_parts_is_deterministic() -> None:
    files = [source(s, "a", size, 0) for s, size in [("0003", 300), ("0001", 100), ("0002", 100)]]
    assert assign_parts(files, 2) == assign_parts(list(reversed(files)), 2)
    spread = assign_parts(files, 5)
    assert len(spread) == 5 and [len(part) for part in spread] == [1, 1, 1, 0, 0]


def test_assign_parts_rejects_no_parts() -> None:
    with pytest.raises(ValueError, match="parts"):
        assign_parts([], 0)
