from __future__ import annotations

import io
import tarfile

import pytest

from hohere.librilight.tarindex import (
    TarHeader,
    find_first_header,
    next_offset,
    parse_header,
    split_points,
    walk,
)

SPEAKER = "0007"


def build(entries: list[tuple[str, bytes]], fmt: int = tarfile.USTAR_FORMAT) -> bytes:
    """In-memory archive; `entries` are (path, payload) pairs."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for path, payload in entries:
            info = tarfile.TarInfo(path)
            info.mtime = 0
            if path.endswith("/"):
                info.type = tarfile.DIRTYPE
            else:
                info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def build_tree() -> list[tuple[str, bytes]]:
    stem = f"medium/{SPEAKER}/book-01/{SPEAKER}-book-01-0000"
    return [
        (f"{stem}.flac", b"fLaC" + b"\x01" * 600),
        (f"{stem}.json", b'{"snr": 12.5}'),
        (f"medium/{SPEAKER}/book-01/{SPEAKER}-book-01-0001.flac", b"fLaC"),
    ]


def reader(data: bytes):
    def read_range(offset: int, n: int) -> bytes:
        return data[offset : offset + n]

    return read_range


def block(
    path: bytes,
    size: int = 0,
    typeflag: bytes = b"0",
    prefix: bytes = b"",
    signed: bool = False,
    size_field: bytes | None = None,
) -> bytes:
    """A hand-built header block, for shapes `tarfile` will not write."""
    out = bytearray(512)
    out[0 : len(path)] = path
    out[100:108] = b"0000644\0"
    out[108:116] = b"0000000\0"
    out[116:124] = b"0000000\0"
    out[124:136] = size_field if size_field is not None else f"{size:011o}\0".encode()
    out[136:148] = b"00000000000\0"
    out[156:157] = typeflag
    out[257:263] = b"ustar\0"
    out[263:265] = b"00"
    out[345 : 345 + len(prefix)] = prefix
    out[148:156] = b" " * 8
    total = sum(b if b < 128 else b - 256 for b in out) if signed else sum(out)
    out[148:156] = f"{total:06o}\0 ".encode()
    return bytes(out)


def test_parse_header_fields() -> None:
    data = build(build_tree())
    h = parse_header(data[:512])
    assert h == TarHeader(
        path=f"medium/{SPEAKER}/book-01/{SPEAKER}-book-01-0000.flac",
        size=604,
        typeflag="0",
        header_offset=0,
    )
    assert h is not None and h.data_offset == 512


def test_parse_header_zero_block_is_end_of_archive() -> None:
    assert parse_header(bytes(512)) is None
    with pytest.raises(ValueError, match="512 bytes"):
        parse_header(b"\0" * 100)


def test_parse_header_rejects_bad_checksum() -> None:
    broken = bytearray(block(b"medium/0007/book/f.flac"))
    broken[20] = ord("Z")
    with pytest.raises(ValueError, match="checksum"):
        parse_header(bytes(broken))


def test_parse_header_joins_ustar_prefix() -> None:
    data = build([(f"medium/{SPEAKER}/{'b' * 60}/{'f' * 40}.flac", b"x")])
    h = parse_header(data[:512])
    assert h is not None
    assert h.path == f"medium/{SPEAKER}/{'b' * 60}/{'f' * 40}.flac"


def test_parse_header_accepts_signed_checksum() -> None:
    path = "medium/0007/book/f\xff\xfe.flac".encode("latin-1")
    for raw in (block(path), block(path, signed=True)):
        h = parse_header(raw)
        assert h is not None and h.path.encode("utf-8", "surrogateescape") == path


def test_parse_header_gnu_base256_size() -> None:
    field = b"\x80" + (3 * 512 + 7).to_bytes(11, "big")
    h = parse_header(block(b"medium/0007/book/f.flac", size_field=field))
    assert h is not None and h.size == 3 * 512 + 7


def test_next_offset_lands_on_next_header() -> None:
    data = build(build_tree())
    headers = list(walk(reader(data), 0, len(data)))
    assert [next_offset(h) for h in headers[:-1]] == [h.header_offset for h in headers[1:]]


def test_walk_yields_regular_files_only() -> None:
    data = build([("medium/0007/", b""), *build_tree()])
    headers = list(walk(reader(data), 0, len(data)))
    assert [h.path.split("/")[-1] for h in headers] == [
        f"{SPEAKER}-book-01-0000.flac",
        f"{SPEAKER}-book-01-0000.json",
        f"{SPEAKER}-book-01-0001.flac",
    ]
    assert [h.size for h in headers] == [604, len(b'{"snr": 12.5}'), 4]


def test_walk_stops_at_end_and_zero_block() -> None:
    data = build(build_tree())
    headers = list(walk(reader(data), 0, len(data)))
    assert len(headers) == 3
    # end is exclusive: stopping at the second header yields only the first entry
    assert [h.path for h in walk(reader(data), 0, headers[1].header_offset)] == [headers[0].path]


def test_walk_gnu_long_name() -> None:
    long_path = f"medium/{SPEAKER}/{'b' * 120}/{SPEAKER}-book-01-0000.flac"
    data = build([(long_path, b"fLaC")], fmt=tarfile.GNU_FORMAT)
    headers = list(walk(reader(data), 0, len(data)))
    assert [h.path for h in headers] == [long_path]


def test_walk_pax_path() -> None:
    long_path = f"medium/{SPEAKER}/{'b' * 120}/{SPEAKER}-book-01-0000.json"
    data = build([(long_path, b"{}")], fmt=tarfile.PAX_FORMAT)
    headers = list(walk(reader(data), 0, len(data)))
    assert [h.path for h in headers] == [long_path]


def test_find_first_header_on_a_mid_archive_slice() -> None:
    data = build(build_tree())
    headers = list(walk(reader(data), 0, len(data)))
    second = headers[1].header_offset
    base = headers[0].data_offset + 100  # inside the first FLAC payload
    chunk = data[base : base + 2048]
    assert find_first_header(chunk, base) == second


def test_find_first_header_skips_flac_payload_with_ustar_bytes() -> None:
    fake = bytearray(block(b"medium/0007/book/fake.flac"))
    fake[0:6] = b"fLaC\x00\x00"
    fake[148:156] = b"7777777\0"  # a checksum that does not match
    assert b"ustar" in fake[257:262] and fake[:6] == b"fLaC\0\0"
    data = build([(f"medium/{SPEAKER}/book-01/a.flac", bytes(fake)), *build_tree()])
    headers = list(walk(reader(data), 0, len(data)))
    suffixes = [h.path.rsplit(".", 1)[-1] for h in headers]
    assert suffixes == ["flac", "flac", "json", "flac"]
    real_second = headers[1].header_offset
    base = 512  # the fake block sits inside the first payload
    assert find_first_header(data[base : base + 2048], base) == real_second
    assert find_first_header(build([("other/0007/book/f.flac", b"x")]), 0) is None


def test_split_points() -> None:
    assert split_points(1000, 4) == [0, 250, 500, 750]
    assert split_points(10, 3) == [0, 3, 6]
    assert split_points(0, 2) == [0, 0]
    with pytest.raises(ValueError, match="parts"):
        split_points(1000, 0)
