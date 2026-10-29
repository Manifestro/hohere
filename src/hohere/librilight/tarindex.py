"""ustar header parsing and sequential walking for a range-served tarball.

`medium.tar` is 321 GB and only ever read over HTTP Range requests, so nothing here
seeks: a caller splits the archive with `split_points`, finds a safe starting header
with `find_first_header` and then follows `next_offset` to the end of its slice.

Regular files of one source utterance are laid out as
`medium/<speaker_id>/<book_dir>/<stem>.{flac,json}`; the two entries are close but
their order is not guaranteed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

BLOCK = 512
ROOT = "medium/"
# typeflags that carry the path of the NEXT entry instead of a file.
LONG_NAME = "L"
PAX_EXTENDED = "x"
PAX_GLOBAL = "g"
REGULAR = ("0", "")


@dataclass(frozen=True)
class TarHeader:
    """One archive entry; `data_offset` is where its payload starts."""

    path: str
    size: int
    typeflag: str
    header_offset: int

    @property
    def data_offset(self) -> int:
        return self.header_offset + BLOCK


def next_offset(h: TarHeader) -> int:
    """Offset of the header after `h` (payloads are padded to whole blocks)."""
    return h.data_offset + -(-h.size // BLOCK) * BLOCK


def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


def _signed_sum(data: bytes) -> int:
    return sum(b if b < 128 else b - 256 for b in data)


def _checksum_ok(block: bytes) -> bool:
    field = block[148:156].replace(b"\0", b" ").strip()
    try:
        expected = int(field or b"0", 8)
    except ValueError:
        return False
    blanked = block[:148] + b" " * 8 + block[156:]
    # Some writers summed the header bytes as signed chars, so both sums count as valid.
    return expected in (sum(blanked), _signed_sum(blanked))


def _number(field: bytes) -> int:
    """Octal field, or GNU base-256 (first byte has the 0x80 marker bit)."""
    if field and field[0] & 0x80:
        return int.from_bytes(bytes([field[0] & 0x7F]) + field[1:], "big")
    text = field.replace(b"\0", b" ").strip()
    return int(text or b"0", 8)


def _magic(block: bytes) -> bool:
    return block[257:262] == b"ustar"


def parse_header(block: bytes, offset: int = 0) -> TarHeader | None:
    """Parse one block; None for the all-zero end-of-archive block.

    `offset` is the absolute position of the block in the archive, 0 for a buffer
    that starts at a header.
    """
    if len(block) != BLOCK:
        raise ValueError(f"tar block must be 512 bytes, got {len(block)}")
    if not any(block):
        return None
    if not _checksum_ok(block):
        raise ValueError(f"bad tar checksum at offset {offset}")
    name = _text(block[0:100])
    # Old GNU headers reuse the prefix area, only POSIX ustar actually has a prefix.
    prefix = _text(block[345:500]) if block[257:265] == b"ustar\x0000" else ""
    return TarHeader(
        path=f"{prefix}/{name}" if prefix else name,
        size=_number(block[124:136]),
        typeflag=block[156:157].decode("ascii", "replace").replace("\0", ""),
        header_offset=offset,
    )


def _pax_path(data: bytes) -> str | None:
    for line in data.split(b"\n"):
        _, _, record = line.partition(b" ")
        key, _, value = record.partition(b"=")
        if key == b"path":
            return value.decode("utf-8", "surrogateescape")
    return None


def _long_path(h: TarHeader, read_range: Callable[[int, int], bytes]) -> str | None:
    data = read_range(h.data_offset, h.size) if h.size else b""
    if h.typeflag == LONG_NAME:
        return _text(data)
    if h.typeflag == PAX_EXTENDED:
        return _pax_path(data)
    return None


def find_first_header(buf: bytes, base_offset: int) -> int | None:
    """Absolute offset of the first `medium/...` header in `buf` at or after `base_offset`.

    Positions are 512-aligned in absolute archive coordinates, so `base_offset` need
    not be aligned itself. The previous entry of a slice is not looked at, so a GNU
    `L` or pax `x` preamble is lost if the found header is the one it belongs to; the
    caller should treat the first entry of a slice as possibly truncated.
    """
    first = (-base_offset) % BLOCK
    for pos in range(first, len(buf) - BLOCK + 1, BLOCK):
        block = buf[pos : pos + BLOCK]
        if not _magic(block) or not _checksum_ok(block):
            continue
        h = parse_header(block, base_offset + pos)
        if h is not None and h.path.startswith(ROOT):
            return base_offset + pos
    return None


def walk(read_range: Callable[[int, int], bytes], start: int, end: int) -> Iterator[TarHeader]:
    """Yield regular files from header offset `start` up to `end` (exclusive).

    GNU long names (`L`) and pax `path=` records are applied to the entry that follows
    them. `read_range(offset, n)` must return exactly n bytes.
    """
    pos = start
    long_path: str | None = None
    while pos + BLOCK <= end:
        block = read_range(pos, BLOCK)
        if len(block) != BLOCK:
            return
        h = parse_header(block, pos)
        if h is None:
            return
        if h.typeflag in (LONG_NAME, PAX_EXTENDED, PAX_GLOBAL):
            if h.typeflag != PAX_GLOBAL:
                long_path = _long_path(h, read_range)
            pos = next_offset(h)
            continue
        if long_path is not None:
            h = replace(h, path=long_path)
            long_path = None
        if h.typeflag in REGULAR:
            yield h
        pos = next_offset(h)


def split_points(total_size: int, parts: int) -> list[int]:
    """Byte offsets that cut the archive into `parts` roughly equal slices."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    return [k * total_size // parts for k in range(parts)]
