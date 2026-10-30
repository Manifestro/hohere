"""Libri-Light adapter: paths, pairing, FLAC duration and VAD."""

from __future__ import annotations

import json

import pytest
from helpers import FakeReader, flac_head, librilight_tar, make_tar, vad_json

from hohere import librilight
from hohere.librilight import (
    flac_duration,
    index,
    index_split,
    make_file_id,
    pair_entries,
    parse_path,
    read_vad,
    sanitize,
)
from hohere.remote import RangeReader
from hohere.tarindex import index_archive


def entries_of(data: bytes):
    return index_archive(FakeReader(data), workers=4)


def test_parse_path_accepts_only_librilight_members():
    parsed = parse_path("small/103/some book/1234-5678.flac")
    assert parsed is not None
    assert (parsed.split, parsed.speaker, parsed.book, parsed.stem, parsed.kind) == (
        "small",
        "103",
        "some book",
        "1234-5678",
        "flac",
    )
    assert parse_path("other/1/2/3.flac") is None  # unknown split
    assert parse_path("small/1/2/3.wav") is None
    assert parse_path("small/1/3.flac") is None
    assert parse_path("small/1/2/3") is None


def test_file_ids_are_sanitized_and_unique():
    fid = make_file_id("small", "103", "some book/1", "a b.flac")
    assert fid == "small-103-some_book_1-a_b_flac"
    assert sanitize("a.b c/¿") == "a_b_c__"
    assert make_file_id("small", "1", "book", "stem") != make_file_id("small", "2", "book", "stem")


def test_flac_duration_reads_streaminfo():
    head = flac_head(sample_rate=16000, samples=160_000)
    assert flac_duration(head) == pytest.approx(10.0)
    assert flac_duration(flac_head(sample_rate=8000, samples=20_000)) == pytest.approx(2.5)


def test_flac_duration_rejects_non_flac_bytes():
    with pytest.raises(ValueError):
        flac_duration(b"not a flac stream at all, really not")
    with pytest.raises(ValueError):
        flac_duration(bytes(42))


def test_read_vad_parses_and_sorts_spans():
    spans = read_vad(vad_json((1.0, 2.0), (0.2, 0.5)))
    assert spans == [(0.2, 0.5), (1.0, 2.0)]
    assert read_vad(json.dumps({"voice_activity": None}).encode()) == []


def test_pairing_reports_flacs_without_json():
    data = make_tar(
        [
            ("small/a/b/one.flac", b"x"),
            ("small/a/b/one.json", vad_json((0, 1))),
            ("small/a/b/two.flac", b"y"),
            ("small/a/b/notes.txt", b"?"),
        ]
    )
    pairs = pair_entries(entries_of(data))
    assert [flac.path for flac, _js in pairs.pairs] == ["small/a/b/one.flac"]
    assert pairs.orphan_flacs == ["small/a/b/two.flac"]


def test_index_split_builds_source_files_with_duration_and_vad():
    data = librilight_tar("small", count=3)
    reader = FakeReader(data)
    files = index_split("small", workers=3, reader=reader, entries=entries_of(data))
    assert len(files) == 3
    first = files[0]
    assert first.file_id.startswith("small-speaker-some_book-")
    assert first.archive_url.endswith("/small.tar")
    assert first.duration_s == pytest.approx(1.0)
    assert first.vad == [(0.5, 3.5), (4.0, 6.0)]
    assert data[first.flac_offset : first.flac_offset + 4] == b"fLaC"
    assert json.loads(data[first.json_offset : first.json_offset + first.json_size])


def test_index_reuses_the_state_cache(tmp_path, monkeypatch):
    from hohere.state import State

    data = librilight_tar("small", count=2)
    entries = entries_of(data)
    with State(tmp_path / "state.db") as state:
        files = index_split("small", reader=FakeReader(data), entries=entries)
        state.add_files(files)
        assert state.has_files("small")

        def explode(*args, **kwargs):  # a second scan would be a cache miss
            raise AssertionError("index_split called again")

        monkeypatch.setattr(librilight, "index_split", explode)
        again = index(["small"], state=state)
    assert [f.file_id for f in again] == [f.file_id for f in files]


def test_index_rejects_unknown_splits():
    with pytest.raises(ValueError, match="unknown split"):
        index(["tiny"])


def test_range_reader_is_used_for_the_flac_head_and_json():
    data = librilight_tar("small", count=1)
    requests: list[tuple[int, int]] = []

    def opener(offset: int, length: int):
        requests.append((offset, length))
        yield data[offset : offset + length]

    reader = RangeReader("http://example/small.tar", size=len(data), opener=opener)
    files = index_split("small", workers=1, reader=reader, entries=entries_of(data))
    assert len(files) == 1
    assert requests[0] == (files[0].flac_offset, librilight.FLAC_HEAD)
    assert requests[1] == (files[0].json_offset, files[0].json_size)
