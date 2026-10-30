"""Libri-Light source adapter: archive entries -> `SourceFile` records.

Layout of every archive (`small.tar`, `medium.tar`, `large.tar`):

    <split>/<speaker>/<book_dir>/<stem>.flac
    <split>/<speaker>/<book_dir>/<stem>.json      # {"voice_activity": [[s, e], ...]}

The JSON is tiny and the FLAC head carries the duration (STREAMINFO), so indexing
costs one 42-byte read and one small read per utterance. Results are cached in the
state DB: a restart never rescans an archive.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import tarindex
from .remote import RangeReader

log = logging.getLogger(__name__)

ARCHIVE_URL = "https://dl.fbaipublicfiles.com/librilight/data/{split}.tar"
SPLITS = ("small", "medium", "large")
FLAC_HEAD = 42  # magic + STREAMINFO block header + the 34 STREAMINFO bytes
SAFE = re.compile(r"[^A-Za-z0-9_-]")


@dataclass(frozen=True)
class SourceFile:
    """One utterance: where its two members live, how long it is and where it speaks."""

    file_id: str
    split: str
    speaker: str
    book: str
    archive_url: str
    flac_offset: int
    flac_size: int
    json_offset: int
    json_size: int
    duration_s: float
    vad: list[tuple[float, float]] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return self.duration_s / 3600.0

    @property
    def speech_seconds(self) -> float:
        return sum(end - start for start, end in self.vad)


@dataclass(frozen=True)
class ParsedPath:
    split: str
    speaker: str
    book: str
    stem: str
    kind: str  # "flac" or "json"


def archive_url(split: str) -> str:
    return ARCHIVE_URL.format(split=split)


def sanitize(text: str) -> str:
    """Everything outside `[A-Za-z0-9_-]` becomes `_` (file ids go into paths)."""
    return SAFE.sub("_", text)


def duplicate_ids(ids: Iterable[str]) -> list[str]:
    """Ids that appear more than once, sorted — one pass, not `ids.count` per id."""
    counts = Counter(ids)
    return sorted(fid for fid, count in counts.items() if count > 1)


def make_file_id(split: str, speaker: str, book: str, stem: str) -> str:
    return f"{split}-{speaker}-{sanitize(book)}-{sanitize(stem)}"


def parse_path(path: str) -> ParsedPath | None:
    """`<split>/<speaker>/<book>/<stem>.<flac|json>`, or None for anything else."""
    parts = path.split("/")
    if len(parts) != 4:
        return None
    split, speaker, book, name = parts
    if split not in SPLITS or not speaker or not book:
        return None
    stem, dot, kind = name.rpartition(".")
    if not dot or not stem or kind not in ("flac", "json"):
        return None
    return ParsedPath(split=split, speaker=speaker, book=book, stem=stem, kind=kind)


@dataclass
class Pairs:
    """FLAC/JSON pairs of one archive plus the FLACs that have no JSON."""

    pairs: list[tuple[tarindex.Entry, tarindex.Entry]] = field(default_factory=list)
    orphan_flacs: list[str] = field(default_factory=list)


def pair_entries(entries: Iterable[tarindex.Entry]) -> Pairs:
    """Pair every `<stem>.flac` with its sibling `<stem>.json`."""
    flacs: dict[tuple[str, str, str, str], tarindex.Entry] = {}
    jsons: dict[tuple[str, str, str, str], tarindex.Entry] = {}
    for entry in entries:
        parsed = parse_path(entry.path)
        if parsed is None:
            continue
        key = (parsed.split, parsed.speaker, parsed.book, parsed.stem)
        (flacs if parsed.kind == "flac" else jsons)[key] = entry
    result = Pairs()
    for key in sorted(flacs):
        flac = flacs[key]
        sibling = jsons.get(key)
        if sibling is None:
            result.orphan_flacs.append(flac.path)
        else:
            result.pairs.append((flac, sibling))
    return result


def flac_duration(head: bytes) -> float:
    """Duration in seconds from the first 42 bytes of a FLAC stream.

    `fLaC`, one metadata block header, then STREAMINFO: 20 bits of sample rate and
    36 bits of total samples packed into 8 bytes at offset 18.
    """
    if len(head) < FLAC_HEAD or head[:4] != b"fLaC":
        raise ValueError(f"not a FLAC stream: {head[:4]!r}")
    if head[4] & 0x7F != 0:  # type 0 is STREAMINFO, which must come first
        raise ValueError("first FLAC metadata block is not STREAMINFO")
    packed = int.from_bytes(head[18:26], "big")
    sample_rate = (packed >> 44) & 0xFFFFF
    total_samples = packed & ((1 << 36) - 1)
    if sample_rate == 0:
        raise ValueError("FLAC STREAMINFO has a zero sample rate")
    return total_samples / sample_rate


def read_vad(data: bytes) -> list[tuple[float, float]]:
    """The `voice_activity` spans of a Libri-Light JSON member."""
    payload = json.loads(data)
    spans: list[tuple[float, float]] = []
    for span in payload.get("voice_activity") or []:
        if len(span) < 2:
            continue
        start, end = float(span[0]), float(span[1])
        if end > start:
            spans.append((start, end))
    spans.sort()
    return spans


def _source_file(
    split: str, url: str, flac: tarindex.Entry, js: tarindex.Entry, reader
) -> SourceFile:
    parsed = parse_path(flac.path)
    assert parsed is not None  # pair_entries only emits parseable paths
    head = reader.read(flac.data_offset, FLAC_HEAD)
    raw = reader.read(js.data_offset, js.size)
    if len(raw) != js.size:
        raise ValueError(f"short JSON read for {js.path}")
    return SourceFile(
        file_id=make_file_id(split, parsed.speaker, parsed.book, parsed.stem),
        split=split,
        speaker=parsed.speaker,
        book=parsed.book,
        archive_url=url,
        flac_offset=flac.data_offset,
        flac_size=flac.size,
        json_offset=js.data_offset,
        json_size=js.size,
        duration_s=flac_duration(head),
        vad=read_vad(raw),
    )


def index_split(
    split: str,
    *,
    workers: int = 64,
    reader: RangeReader | None = None,
    entries: Sequence[tarindex.Entry] | None = None,
) -> list[SourceFile]:
    """Index one archive: scan its headers, pair the members, read heads and VADs."""
    url = archive_url(split)
    reader = reader or RangeReader(url)
    if entries is None:
        log.info("%s: scanning %s with %d slices", split, url, workers)
        entries = tarindex.index_archive(reader, workers=workers)
    pairs = pair_entries(entries)
    for orphan in pairs.orphan_flacs:
        log.warning("%s: FLAC without a sibling JSON, skipped", orphan)

    def build(pair: tuple[tarindex.Entry, tarindex.Entry]) -> SourceFile:
        return _source_file(split, url, pair[0], pair[1], reader)

    if workers > 1 and pairs.pairs:
        with ThreadPoolExecutor(max_workers=min(workers, len(pairs.pairs))) as pool:
            files = list(pool.map(build, pairs.pairs))
    else:
        files = [build(pair) for pair in pairs.pairs]
    duplicates = duplicate_ids(f.file_id for f in files)
    assert not duplicates, f"non-unique file ids in {split}: {duplicates[:5]}"
    log.info("%s: %d files, %d orphans", split, len(files), len(pairs.orphan_flacs))
    return files


def index(
    splits: Iterable[str],
    workers: int = 64,
    *,
    state: Any = None,
    use_cache: bool = True,
) -> list[SourceFile]:
    """Index the requested splits, reusing the state DB index when it already has one."""
    wanted = list(splits)
    unknown = [s for s in wanted if s not in SPLITS]
    if unknown:
        raise ValueError(f"unknown split(s) {unknown}, expected a subset of {SPLITS}")
    files: list[SourceFile] = []
    for split in wanted:
        cached = state.files(split=split) if state is not None and use_cache else []
        if cached:
            log.info("%s: %d files from the index cache", split, len(cached))
            files.extend(cached)
            continue
        fresh = index_split(split, workers=workers)
        if state is not None:
            state.add_files(fresh)
        files.extend(fresh)
    duplicates = duplicate_ids(f.file_id for f in files)
    assert not duplicates, f"non-unique file ids: {duplicates[:5]}"
    return files
