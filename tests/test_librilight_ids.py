from __future__ import annotations

import pytest

from hohere.librilight.files import SourceFile
from hohere.librilight.ids import (
    chapter_id,
    check_unique_chapters,
    parse_segment_id,
    segment_id,
)
from hohere.librilight.tarindex import TarHeader


def source(speaker: str, book_dir: str, stem: str) -> SourceFile:
    return SourceFile(
        speaker=speaker,
        book_dir=book_dir,
        stem=stem,
        flac=TarHeader(f"medium/{speaker}/{book_dir}/{stem}.flac", 1000, "0", 0),
        json=TarHeader(f"medium/{speaker}/{book_dir}/{stem}.json", 300, "0", 512 * 3),
    )


def test_chapter_id_sanitises_unsafe_characters() -> None:
    assert chapter_id("book-01", "1234-5678") == "book_01_1234_5678"
    assert chapter_id("a b/c.d", "e:f") == "a_b_c_d_e_f"
    # a chapter id is the middle field of a segment id, so a dash would break parsing
    assert "-" not in chapter_id("book-01", "0000")
    assert segment_id("0007", chapter_id("book-01", "0000"), 0) == "0007-book_01_0000-0"


def test_segment_id_round_trips() -> None:
    for speaker, chapter, index in [("0007", "book_01_0000", 0), ("1234", "a", 42)]:
        ident = segment_id(speaker, chapter, index)
        assert parse_segment_id(ident) == (speaker, chapter, index)


def test_segment_id_validates_its_parts() -> None:
    with pytest.raises(ValueError, match="speaker"):
        segment_id("00a7", "book", 0)
    with pytest.raises(ValueError, match="chapter"):
        segment_id("0007", "book-01", 0)
    with pytest.raises(ValueError, match="index"):
        segment_id("0007", "book", -1)


def test_parse_segment_id_rejects_foreign_strings() -> None:
    for bad in ["", "0007", "0007-book", "0007-book-x", "00a7-book-1", "0007-b-01-1"]:
        with pytest.raises(ValueError, match="segment id"):
            parse_segment_id(bad)
    with pytest.raises(ValueError, match="segment id"):
        parse_segment_id("0007-book-007")  # does not round-trip through segment_id


def test_check_unique_chapters_lists_collisions() -> None:
    assert check_unique_chapters([source("0007", "book-01", "a"), source("0007", "b", "a")]) is None
    files = [
        source("0007", "book.01", "a"),
        source("0007", "book_01", "a"),
        source("0008", "book_01", "a"),
    ]
    with pytest.raises(ValueError) as excinfo:
        check_unique_chapters(files)
    message = str(excinfo.value)
    assert "book_01_a" in message
    assert "book.01" in message and "book_01" in message
    assert "0008" not in message  # another speaker's chapter of the same name is fine
