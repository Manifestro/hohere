"""`hohere librilight index`: walk the tar once and write `plan.jsonl`.

Nothing here decodes audio: the length of a FLAC comes from its STREAMINFO block and
the speech intervals from the JSON next to it, both fetched by range. The archive is
cut into `parts` slices that are walked in parallel and then checked to be one
contiguous chain of regular entries, so a missing or duplicated slice cannot hide.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

from hohere.librilight import plan as plan_mod
from hohere.librilight.files import SourceFile, pair_entries
from hohere.librilight.flac import parse_streaminfo
from hohere.librilight.ids import chapter_id, check_unique_chapters
from hohere.librilight.remote import RangeReader
from hohere.librilight.tarindex import (
    BLOCK,
    REGULAR,
    TarHeader,
    find_first_header,
    next_offset,
    parse_header,
    split_points,
    walk,
)

log = logging.getLogger(__name__)

DEFAULT_URL = "https://dl.fbaipublicfiles.com/librilight/data/medium.tar"
TAR_MD5 = "c75e7ac62471bfbf2db77528d62a9b74"
PARTS = 512
WORKERS = 64
SCAN_CHUNK = 1 << 20
SCAN_LIMIT = 64 << 20
STREAMINFO_BYTES = 4096

ReadRange = Callable[[int, int], bytes]


def _scan(read_range: ReadRange, start: int, end: int) -> int | None:
    """First header offset at or after `start`, scanning forward in 1 MiB chunks.

    Each window peeks one block past its end, so a header that starts inside the slice
    but ends past it is still found when the slice is narrower than a header.
    """
    pos = start
    limit = min(end, start + SCAN_LIMIT)
    while pos < limit:
        n = min(SCAN_CHUNK, limit - pos) + BLOCK
        buf = read_range(pos, n)
        found = find_first_header(buf, pos)
        if found is not None and found < end:
            return found
        if len(buf) < n:
            return None
        pos += SCAN_CHUNK
    return None


def part_headers(read_range: ReadRange, start: int, end: int) -> list[TarHeader]:
    """Regular entries of one slice `[start, end)` of the archive.

    `walk` needs one whole block, so the end is widened by a block when the slice is
    narrower than the header it starts on; duplicates are dropped by `index_headers`.
    """
    found = _scan(read_range, start, end)
    if found is None:
        return []
    return list(walk(read_range, found, max(end, found + BLOCK)))


def index_headers(
    read_range: ReadRange, total: int, parts: int, workers: int = WORKERS
) -> list[TarHeader]:
    """Walk every slice in `workers` threads; entries come back in archive order."""
    points = split_points(total, parts)
    slices = [(points[k], points[k + 1] if k + 1 < parts else total) for k in range(parts)]
    if workers <= 1:
        chunks = [part_headers(read_range, start, end) for start, end in slices]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            chunks = list(pool.map(lambda se: part_headers(read_range, se[0], se[1]), slices))
    by_offset: dict[int, TarHeader] = {}
    for headers in chunks:
        for h in headers:
            by_offset.setdefault(h.header_offset, h)
    return [by_offset[offset] for offset in sorted(by_offset)]


def check_chain(read_range: ReadRange, headers: Sequence[TarHeader]) -> list[str]:
    """Problems with the chain of regular entries; an empty list means it is contiguous.

    Directory, long-name and pax entries are allowed to sit between two regular
    entries; anything else that does not line up with `next_offset` is a hole.
    """
    problems: list[str] = []
    ordered = sorted(headers, key=lambda h: h.header_offset)
    for prev, nxt in pairwise(ordered):
        pos = next_offset(prev)
        while pos < nxt.header_offset:
            block = read_range(pos, BLOCK)
            h = parse_header(block, pos) if len(block) == BLOCK else None
            if h is None:
                problems.append(f"zero block at {pos}, before {nxt.path!r} at {nxt.header_offset}")
                break
            if h.typeflag in REGULAR:
                problems.append(
                    f"hole: {prev.path!r} ends at {pos}, but {h.path!r} starts there"
                    f" ({nxt.path!r} is at {nxt.header_offset})"
                )
                break
            pos = next_offset(h)
        else:
            if pos != nxt.header_offset:
                problems.append(
                    f"overrun: {prev.path!r} runs past {nxt.header_offset} to {pos},"
                    f" losing {nxt.path!r}"
                )
    return problems


def _header_line(h: TarHeader) -> dict:
    return {
        "path": h.path,
        "size": h.size,
        "typeflag": h.typeflag,
        "header_offset": h.header_offset,
    }


def write_index(path: Path, headers: Iterable[TarHeader]) -> int:
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with open(tmp, "w") as f:
        for h in headers:
            f.write(json.dumps(_header_line(h)) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_index(path: Path) -> list[TarHeader]:
    headers = []
    with open(path) as f:
        for line in f:
            if line.strip():
                headers.append(TarHeader(**json.loads(line)))
    return headers


def _meta_path(cache: Path | None, file: SourceFile) -> Path | None:
    if cache is None:
        return None
    return cache / file.speaker / f"{chapter_id(file.book_dir, file.stem)}.json"


def read_meta(read_range: ReadRange, file: SourceFile, cache: Path | None = None) -> dict:
    """`duration`, `snr` and `voice_activity` of one file, cached in `WORK/meta/`."""
    path = _meta_path(cache, file)
    if path is not None and path.exists():
        return json.loads(path.read_text())
    raw = json.loads(read_range(file.json.data_offset, file.json.size))
    info = parse_streaminfo(read_range(file.flac.data_offset, STREAMINFO_BYTES))
    meta = {
        "duration": info.duration,
        "snr": float(raw.get("snr") or 0.0),
        "voice_activity": [[float(a), float(b)] for a, b in raw.get("voice_activity") or []],
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(path)
    return meta


def _plan_for(
    read_range: ReadRange, file: SourceFile, cache: Path | None
) -> plan_mod.PlanFile | None:
    meta = read_meta(read_range, file, cache)
    return plan_mod.build(file, meta["duration"], meta["snr"], meta["voice_activity"])


def run_index(
    out: Path,
    url: str = DEFAULT_URL,
    parts: int = PARTS,
    workers: int = WORKERS,
    exclude_speakers: Iterable[str] = (),
    exclude_recordings: Iterable[Iterable[str]] = (),
    reader: RangeReader | None = None,
) -> dict:
    """Build `tar_index.jsonl`, `plan.jsonl` and `index_summary.json` in `out`."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = reader if reader is not None else RangeReader(url)
    index_path = out / "tar_index.jsonl"

    if index_path.exists():
        headers = read_index(index_path)
        total = max((next_offset(h) for h in headers), default=0)
        log.info("reusing %s (%d entries)", index_path, len(headers))
    else:
        total = source.size()
        log.info("%s: %d bytes in %d parts", url, total, parts)
        headers = index_headers(source.read, total, parts, workers)
        problems = check_chain(source.read, headers)
        if problems:
            raise RuntimeError(
                f"the archive does not form one contiguous chain ({len(problems)} problems):\n"
                + "\n".join(problems[:10])
            )
        write_index(index_path, headers)
        log.info("wrote %s: %d entries", index_path, len(headers))

    files, unpaired = pair_entries(headers)
    kept, dropped = plan_mod.filter_files(files, exclude_speakers, exclude_recordings)
    check_unique_chapters(kept)
    if unpaired:
        with open(out / "unpaired.jsonl", "w") as f:
            for h in unpaired:
                f.write(json.dumps(_header_line(h)) + "\n")

    cache = out / "meta"
    if workers <= 1:
        plans = [_plan_for(source.read, f, cache) for f in kept]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            plans = list(pool.map(lambda f: _plan_for(source.read, f, cache), kept))

    plan_files = [p for p in plans if p is not None]
    segments = sum(len(p.segments) for p in plan_files)
    seconds = sum(e - s for p in plan_files for s, e in p.segments)
    summary = {
        "url": url,
        "tar_md5": TAR_MD5,
        "bytes": total,
        "parts": parts,
        "entries": len(headers),
        "files_seen": len(files) + len(unpaired),
        "paired": len(files),
        "unpaired": len(unpaired),
        "excluded": {
            "speaker": dropped.get("speaker", 0),
            "recording": dropped.get("recording", 0),
        },
        "exclude_speakers": sorted({str(s) for s in exclude_speakers}),
        "exclude_recordings": sorted([str(s), str(t)] for s, t in exclude_recordings),
        "no_segments": len(plans) - len(plan_files),
        "kept": len(plan_files),
        "segments": segments,
        "hours": round(seconds / 3600, 4),
        "speakers": len({p.speaker for p in plan_files}),
    }
    plan_mod.write_plan(out / "plan.jsonl", plan_files)
    tmp = out / "index_summary.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    tmp.replace(out / "index_summary.json")
    log.info("plan: %d files, %d segments, %.1f h", len(plan_files), segments, summary["hours"])
    return summary
