"""Index a remote tar archive without downloading it.

An archive is a chain of 512-byte headers: a header names the next entry, its size,
and the payload follows padded to whole blocks. Nothing here seeks randomly: the
archive is cut into `workers` slices, the first valid header of every slice is found
by reading forward, and every slice is then walked from its own first header up to
the next slice's first header.

Two things make remote walking harder than `tarfile`:

* GNU long names (`L` pax-less) and PAX extended headers (`x`) carry the path and
  sometimes the size of the entry that follows them, so walking must carry state
  across two headers.
* A false positive is cheap: any 512-aligned block inside a FLAC payload could pass
  the checksum. `find_first_header` demands the ustar magic and additionally that
  the header found is followed by another valid header (or the end of the archive).

The chained slices are checked to cover the archive exactly once: a slice whose walk
does not end exactly at the next slice's first header is an error, not a silent gap.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

log = logging.getLogger(__name__)

BLOCK = 512
# typeflags that carry the path (and maybe the size) of the NEXT entry
LONG_NAME = "L"
PAX_EXTENDED = "x"
PAX_GLOBAL = "g"
REGULAR = ("0", "")
PREAMBLE = (LONG_NAME, PAX_EXTENDED, PAX_GLOBAL)
WINDOW = 4096  # walk read size: a few small headers or one header plus a payload sliver
SCAN_WINDOW = 1 << 18  # forward search for a slice start is read in bigger strides


class TarIndexError(RuntimeError):
    """The archive could not be indexed consistently."""


@dataclass(frozen=True)
class Entry:
    """One regular file: `data_offset` is where its payload starts."""

    path: str
    data_offset: int
    size: int


@dataclass(frozen=True)
class _Header:
    path: str
    size: int
    typeflag: str
    offset: int

    @property
    def data_offset(self) -> int:
        return self.offset + BLOCK


def next_offset(header: _Header) -> int:
    """Offset of the header after `header` (payloads are padded to whole blocks)."""
    return header.data_offset + -(-header.size // BLOCK) * BLOCK


# -- low level block parsing -----------------------------------------------------
def _text(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", "surrogateescape")


def _signed_sum(data: bytes) -> int:
    return sum(b if b < 128 else b - 256 for b in data)


def checksum_ok(block: bytes) -> bool:
    """True when the header checksum field matches the block.

    Both the unsigned and the signed sum are accepted: some old writers summed the
    header bytes as signed chars.
    """
    field = block[148:156].replace(b"\0", b" ").strip()
    try:
        expected = int(field or b"0", 8)
    except ValueError:
        return False
    blanked = block[:148] + b" " * 8 + block[156:]
    return expected in (sum(blanked), _signed_sum(blanked))


def _number(field: bytes) -> int:
    """Octal field, or GNU base-256 (the first byte has the 0x80 marker bit)."""
    if field and field[0] & 0x80:
        return int.from_bytes(bytes([field[0] & 0x7F]) + field[1:], "big")
    text = field.replace(b"\0", b" ").strip()
    return int(text or b"0", 8)


def parse_header(block: bytes, offset: int = 0) -> _Header | None:
    """Parse one block; None for the all-zero end-of-archive block."""
    if len(block) != BLOCK:
        raise ValueError(f"tar block must be {BLOCK} bytes, got {len(block)}")
    if not any(block):
        return None
    if not checksum_ok(block):
        raise ValueError(f"bad tar checksum at offset {offset}")
    name = _text(block[0:100])
    # Old GNU headers reuse the prefix area; only POSIX ustar really has a prefix.
    prefix = _text(block[345:500]) if block[257:265] == b"ustar\x0000" else ""
    return _Header(
        path=f"{prefix}/{name}" if prefix else name,
        size=_number(block[124:136]),
        typeflag=block[156:157].decode("ascii", "replace").replace("\0", ""),
        offset=offset,
    )


def looks_like_header(block: bytes) -> bool:
    """Cheap validity test used while scanning forward for a slice start."""
    if len(block) != BLOCK or not any(block):
        return False
    if block[257:262] != b"ustar":
        return False
    return checksum_ok(block)


def parse_pax(data: bytes) -> dict[str, str]:
    """Parse `x`/`g` header records (`<length> <key>=<value>\\n`)."""
    out: dict[str, str] = {}
    pos = 0
    while pos < len(data):
        space = data.find(b" ", pos)
        if space < 0:
            break
        try:
            length = int(data[pos:space])
        except ValueError:
            break
        if length <= 0 or pos + length > len(data):
            break
        record = data[space + 1 : pos + length].rstrip(b"\n")
        key, _, value = record.partition(b"=")
        out[key.decode("utf-8", "replace")] = value.decode("utf-8", "surrogateescape")
        pos += length
    return out


class _Window:
    """A read-through cache: sequential header walking costs one request per window."""

    def __init__(self, reader, window: int = WINDOW) -> None:
        self.reader = reader
        self.window = window
        self._base = 0
        self._buf = b""

    def read(self, offset: int, length: int) -> bytes:
        base, buf = self._base, self._buf
        if base <= offset and offset + length <= base + len(buf):
            start = offset - base
            return buf[start : start + length]
        self._buf = self.reader.read(offset, max(length, self.window))
        self._base = offset
        return self._buf[:length]


def _read_preamble(window: _Window, header: _Header) -> tuple[str | None, int | None]:
    """Path and size override carried by a GNU `L` / PAX `x` header."""
    data = window.read(header.data_offset, header.size) if header.size else b""
    if len(data) != header.size:
        raise TarIndexError(f"truncated {header.typeflag} payload at {header.offset}")
    if header.typeflag == LONG_NAME:
        return _text(data), None
    records = parse_pax(data)
    size = records.get("size")
    try:
        size_override = int(size) if size is not None else None
    except ValueError:
        size_override = None
    return records.get("path"), size_override


def _walk(reader, start: int, end: int) -> tuple[list[Entry], int]:
    """Walk headers from `start` up to `end`; returns the entries and the stop offset."""
    window = _Window(reader) if not isinstance(reader, _Window) else reader
    entries: list[Entry] = []
    pos = start
    path: str | None = None
    size: int | None = None
    while pos + BLOCK <= end:
        block = window.read(pos, BLOCK)
        if len(block) < BLOCK:
            break
        try:
            header = parse_header(block, pos)
        except ValueError as exc:
            raise TarIndexError(f"{exc} (slice {start}..{end})") from exc
        if header is None:  # end-of-archive zero block
            break
        if header.typeflag in PREAMBLE:
            if header.typeflag == PAX_GLOBAL:
                pos = next_offset(header)
                continue
            carried_path, carried_size = _read_preamble(window, header)
            if carried_path is not None:
                path = carried_path
            if carried_size is not None:
                size = carried_size
            pos = next_offset(header)
            continue
        entry_path = path if path is not None else header.path
        entry_size = size if size is not None else header.size
        path = size = None
        if header.typeflag in REGULAR:
            entries.append(Entry(entry_path, header.data_offset, entry_size))
        pos = header.data_offset + -(-entry_size // BLOCK) * BLOCK
    return entries, pos


def scan(reader, start: int, end: int) -> Iterator[Entry]:
    """Yield the regular files of a slice that starts at the header offset `start`."""
    entries, _ = _walk(reader, start, end)
    yield from entries


def scan_to(reader, start: int, end: int) -> tuple[list[Entry], int]:
    """`scan`, plus the offset at which the walk stopped (used to chain slices)."""
    return _walk(reader, start, end)


# -- finding a safe slice start --------------------------------------------------
def _previous_preamble(reader, candidate: int) -> int | None:
    """Offset of a `L`/`x`/`g` header whose payload ends exactly at `candidate`."""
    for back in (BLOCK, 2 * BLOCK, 3 * BLOCK, 4 * BLOCK):
        if candidate - back < 0:
            break
        block = reader.read(candidate - back, BLOCK)
        if not looks_like_header(block):
            continue
        header = parse_header(block, candidate - back)
        if (
            header is not None
            and header.typeflag in PREAMBLE
            and next_offset(header) == candidate
        ):
            return candidate - back
    return None


def find_first_header(reader, offset: int, *, size: int | None = None) -> int | None:
    """Absolute offset of the first valid header at or after `offset`, or None.

    A candidate must have the ustar magic, a valid checksum and be followed by
    another valid header (or by the end-of-archive padding). If the candidate is the
    entry of a GNU/PAX preamble that sits right before it, the preamble is returned
    instead so the caller can apply the long path.
    """
    if size is None:
        size = reader.size()
    pos = offset - offset % BLOCK
    while pos + BLOCK <= size:
        block = reader.read(pos, max(SCAN_WINDOW, BLOCK))
        if not block:
            return None
        stop = min(len(block), size - pos)
        for rel in range(0, stop - BLOCK + 1, BLOCK):
            candidate = pos + rel
            chunk = block[rel : rel + BLOCK]
            if not looks_like_header(chunk):
                continue
            header = parse_header(chunk, candidate)
            if header is None or header.typeflag not in (*REGULAR, *PREAMBLE, "1", "2", "5", "6"):
                continue
            after = next_offset(header)
            if after + BLOCK <= size:
                following = reader.read(after, BLOCK)
                if following and any(following) and not looks_like_header(following):
                    continue
            start = candidate
            for _ in range(8):
                earlier = _previous_preamble(reader, start)
                if earlier is None:
                    break
                start = earlier
            return start
        if stop < len(block):  # reached the end of the archive
            return None
        pos += (stop // BLOCK) * BLOCK
    return None


def split_points(size: int, parts: int) -> list[int]:
    """Byte offsets that cut the archive into `parts` roughly equal slices."""
    if parts < 1:
        raise ValueError("parts must be >= 1")
    points = sorted({k * size // parts for k in range(parts)})
    return points


def _tail_is_empty(reader, pos: int, end: int) -> bool:
    if pos >= end:
        return True
    rest = reader.read(pos, end - pos)
    return len(rest) == end - pos and not any(rest)


def slice_starts(reader, points: Sequence[int], size: int, workers: int) -> list[int]:
    """Header offsets that chain the archive, one per usable slice.

    A split point whose first header cannot be found (or that lands inside the
    previous slice) is merged into the previous slice, never dropped silently.
    """

    def locate(point: int) -> int | None:
        try:
            return find_first_header(reader, point, size=size)
        except Exception as exc:  # a broken slice must not kill the whole scan
            log.warning("cannot find a header at %d: %s", point, exc)
            return None

    candidates = points[1:]
    if candidates:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(candidates)))) as pool:
            found = list(pool.map(locate, candidates))
    else:
        found = []
    starts = [0]
    for start in found:
        if start is None or start <= starts[-1] or start >= size:
            continue
        starts.append(start)
    return starts


def index_archive(reader, workers: int = 8, points: Sequence[int] | None = None) -> list[Entry]:
    """Index the whole archive, all slices in parallel, and check the chaining."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    size = reader.size()
    if size == 0:
        return []
    cuts = list(points) if points is not None else split_points(size, workers)
    cuts = [c for c in cuts if 0 < c < size]
    starts = slice_starts(reader, [0, *cuts], size, workers) if cuts else [0]
    bounds = [*starts, size]
    pairs = list(itertools.pairwise(bounds))
    if len(pairs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(pairs))) as pool:
            slices = list(pool.map(lambda ab: scan_to(reader, *ab), pairs))
    else:
        slices = [scan_to(reader, *pairs[0])] if pairs else []
    entries: list[Entry] = []
    for idx, ((start, end), (slice_entries, final)) in enumerate(
        zip(pairs, slices, strict=True)
    ):
        last = idx == len(pairs) - 1
        if final != end and not (last and _tail_is_empty(reader, final, end)):
            raise TarIndexError(
                f"slice {idx} ({start}..{end}) walked to {final}: "
                f"{'gap' if final < end else 'overlap'} in the archive chain"
            )
        entries.extend(slice_entries)
    return entries

