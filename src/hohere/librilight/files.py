"""Pair the FLAC and JSON entries of one Libri-Light utterance, then shard speakers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

from hohere.librilight.tarindex import TarHeader

ROOT = "medium"
AUDIO = "flac"
META = "json"
T = TypeVar("T")


@dataclass(frozen=True)
class SourceFile:
    """One utterance in the archive: its FLAC and the JSON with `snr` and VAD."""

    speaker: str
    book_dir: str
    stem: str
    flac: TarHeader
    json: TarHeader


def _split_path(path: str) -> tuple[str, str, str, str] | None:
    """`medium/<speaker>/<book>/<stem>.<suffix>` -> its four parts, else None."""
    parts = path.split("/")
    if len(parts) != 4 or parts[0] != ROOT or not parts[1] or not parts[2]:
        return None
    stem, dot, suffix = parts[3].rpartition(".")
    if not dot or not stem:
        return None
    return parts[1], parts[2], stem, suffix.lower()


def pair_entries(headers: Iterable[TarHeader]) -> tuple[list[SourceFile], list[TarHeader]]:
    """Group entries by directory and stem: `(paired, unpaired)`.

    The pairs come back in archive order; the second list holds what could not be
    paired (a FLAC without a JSON, a JSON without a FLAC, a duplicate of either), so
    the caller can report it instead of silently dropping audio.
    """
    groups: dict[tuple[str, str, str], dict[str, TarHeader]] = defaultdict(dict)
    order: list[tuple[str, str, str]] = []
    unpaired: list[TarHeader] = []
    for h in headers:
        parts = _split_path(h.path)
        if parts is None:
            continue
        speaker, book_dir, stem, suffix = parts
        if suffix not in (AUDIO, META):
            continue
        if not speaker.isdigit():
            raise ValueError(f"speaker id {speaker!r} is not numeric: {h.path!r}")
        key = (speaker, book_dir, stem)
        if key not in groups:
            order.append(key)
        if suffix in groups[key]:
            unpaired.append(h)
            continue
        groups[key][suffix] = h

    files: list[SourceFile] = []
    for key in order:
        group = groups[key]
        if len(group) < 2:
            unpaired.extend(group.values())
            continue
        speaker, book_dir, stem = key
        files.append(
            SourceFile(
                speaker=speaker,
                book_dir=book_dir,
                stem=stem,
                flac=group[AUDIO],
                json=group[META],
            )
        )
    return files, unpaired


def balance_by_speaker(
    groups: dict[str, list[T]], sizes: dict[str, int], parts: int, order: Callable[[T], object]
) -> list[list[T]]:
    """Spread whole speakers over `parts` shards, balanced by total bytes.

    Greedy largest-first: equal sizes are ordered by speaker id and a tie between
    shards goes to the lowest index, so the assignment is deterministic. Each part is
    sorted with `order` (the archive offset, so a part reads the tar forward).
    """
    if parts < 1:
        raise ValueError("parts must be >= 1")
    out: list[list[T]] = [[] for _ in range(parts)]
    load = [0] * parts
    for speaker in sorted(groups, key=lambda s: (-sizes[s], int(s))):
        k = load.index(min(load))
        out[k].extend(groups[speaker])
        load[k] += sizes[speaker]
    for group in out:
        group.sort(key=order)
    return out


def assign_parts(files: Iterable[SourceFile], parts: int) -> list[list[SourceFile]]:
    """Shards of whole speakers, balanced by total FLAC bytes."""
    by_speaker: dict[str, list[SourceFile]] = defaultdict(list)
    for f in files:
        by_speaker[f.speaker].append(f)
    sizes = {s: sum(f.flac.size for f in group) for s, group in by_speaker.items()}
    return balance_by_speaker(by_speaker, sizes, parts, order=lambda f: f.flac.data_offset)
