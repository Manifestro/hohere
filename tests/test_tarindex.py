"""tar indexing: headers, GNU long names, PAX, slice chaining."""

from __future__ import annotations

import tarfile

import pytest
from helpers import FakeReader, make_tar

from hohere import tarindex
from hohere.tarindex import BLOCK, Entry, find_first_header, index_archive, scan


def test_scan_yields_entries_with_offsets_and_sizes():
    data = make_tar([("small/a/b/one.flac", b"x" * 700), ("small/a/b/two.json", b"{}")])
    reader = FakeReader(data)
    entries = list(scan(reader, 0, len(data)))
    assert [e.path for e in entries] == ["small/a/b/one.flac", "small/a/b/two.json"]
    assert entries[0].data_offset == BLOCK
    assert entries[0].size == 700
    assert entries[1].data_offset == 4 * BLOCK  # header + 1024 padded bytes
    assert all(data[e.data_offset : e.data_offset + e.size] for e in entries)


def test_gnu_long_names_are_applied_to_the_next_entry():
    long_name = "medium/speaker/" + "b" * 90 + "/" + "c" * 90 + ".flac"
    data = make_tar([(long_name, b"abc")], fmt=tarfile.GNU_FORMAT)
    entries = index_archive(FakeReader(data), workers=2)
    assert [e.path for e in entries] == [long_name]


def test_pax_path_and_size_override_are_applied():
    data = make_tar([("small/a/b/" + "d" * 120 + ".flac", b"0123456789")], fmt=tarfile.PAX_FORMAT)
    entries = index_archive(FakeReader(data), workers=3)
    assert len(entries) == 1
    assert entries[0].path.endswith("d" * 120 + ".flac")
    assert entries[0].size == 10


def pax_record(key: str, value: str) -> bytes:
    body = f"{key}={value}".encode()
    length = len(body) + 2  # space + newline
    while True:
        record = f"{length} ".encode() + body + b"\n"
        if len(record) == length:
            return record
        length = len(record)


def test_pax_size_record_overrides_the_header_size():
    """A PAX `size` record wins over the (possibly truncated) header field."""
    import io

    payload = b"z" * 300
    pax = pax_record("path", "small/a/b/long.flac") + pax_record("size", str(len(payload)))
    preamble = tarfile.TarInfo("small/a/b/truncated.flac")
    preamble.type = tarfile.XHDTYPE
    preamble.size = len(pax)
    entry = tarfile.TarInfo("small/a/b/truncated.flac")
    entry.size = len(payload)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        tar.addfile(preamble, io.BytesIO(pax))
        tar.addfile(entry, io.BytesIO(payload))
    data = buffer.getvalue()
    entries = list(scan(FakeReader(data), 0, len(data)))
    assert entries[0].path == "small/a/b/long.flac"
    assert entries[0].size == len(payload)


def test_find_first_header_skips_payload_bytes():
    data = make_tar([("small/a/b/one.flac", b"q" * 4096), ("small/a/b/two.flac", b"r" * 10)])
    reader = FakeReader(data)
    start = find_first_header(reader, BLOCK + 1000)
    entries = list(scan(reader, start, len(data)))
    assert entries[0].path == "small/a/b/two.flac"


def test_find_first_header_returns_none_at_the_end():
    data = make_tar([("small/a/b/one.flac", b"q" * 100)])
    assert find_first_header(FakeReader(data), len(data)) is None


def test_find_first_header_rejects_blocks_without_the_ustar_magic():
    data = make_tar([("small/a/b/one.flac", b"q" * 2048), ("small/a/b/two.flac", b"z")])
    _first, following = list(scan(FakeReader(data), 0, len(data)))
    corrupted = bytearray(data)
    corrupted[257:262] = b"xxxxx"  # break the magic of the first header
    assert find_first_header(FakeReader(bytes(corrupted)), 0) == following.data_offset - BLOCK


def test_index_archive_covers_every_entry_once_across_slices():
    members = [(f"medium/a/b/{i}.flac", bytes([i]) * (700 + 37 * i)) for i in range(40)]
    data = make_tar(members)
    entries = index_archive(FakeReader(data), workers=8)
    assert [e.path for e in entries] == [name for name, _ in members]
    for (_name, payload), entry in zip(members, entries, strict=True):
        start = entry.data_offset
        assert data[start : start + entry.size] == payload
    assert len({e.data_offset for e in entries}) == len(members)


def test_index_archive_merges_slices_whose_header_cannot_be_found():
    data = make_tar([("small/a/b/one.flac", b"x" * 200_000)])
    entries = index_archive(FakeReader(data), workers=16, points=None)
    assert [e.path for e in entries] == ["small/a/b/one.flac"]


def test_a_gap_in_the_chain_is_an_error_not_a_silent_drop():
    members = [(f"small/a/b/{i}.flac", b"p" * 600) for i in range(12)]
    data = bytearray(make_tar(members))
    # zero one header in the middle: the walk from slice 0 stops there
    victim = 6 * 2 * BLOCK
    data[victim : victim + BLOCK] = b"\0" * BLOCK
    with pytest.raises(tarindex.TarIndexError):
        index_archive(FakeReader(bytes(data)), workers=1)


def test_split_points_cover_the_archive():
    assert tarindex.split_points(1000, 4) == [0, 250, 500, 750]
    assert tarindex.split_points(1000, 1) == [0]
    with pytest.raises(ValueError):
        tarindex.split_points(1000, 0)


def test_parse_header_rejects_a_bad_checksum():
    data = bytearray(make_tar([("small/a/b/one.flac", b"x")]))
    data[150] = ord("7")  # inside the checksum field
    with pytest.raises(ValueError, match="checksum"):
        tarindex.parse_header(bytes(data[:BLOCK]))


def test_next_offset_pads_payloads_to_whole_blocks():
    header = tarindex.parse_header(make_tar([("a", b"x" * 700)])[:BLOCK])
    assert header is not None
    assert tarindex.next_offset(header) == BLOCK + 2 * BLOCK


def test_entries_are_reported_in_archive_order():
    data = make_tar([(f"small/a/b/{i}.json", b"{}") for i in range(5)])
    entries = list(scan(FakeReader(data), 0, len(data)))
    assert all(isinstance(e, Entry) for e in entries)
    assert [e.data_offset for e in entries] == sorted(e.data_offset for e in entries)
