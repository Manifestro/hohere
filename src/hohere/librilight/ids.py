"""Stable ids for chapters and segments.

A segment id is `<speaker>-<chapter>-<index>`: the speaker id and the index are what a
trainer needs to find a FLAC again, and the dashes are the only separators, so a
chapter id must never contain one — which is why `book_dir` and `stem` are sanitised.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from hohere.librilight.files import SourceFile

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_CHAPTER = re.compile(r"[A-Za-z0-9_]+")


def chapter_id(book_dir: str, stem: str) -> str:
    """`book_dir` + `stem` with every character outside `[A-Za-z0-9_]` replaced by `_`."""
    return _UNSAFE.sub("_", f"{book_dir}_{stem}")


def segment_id(speaker: str, chapter: str, index: int) -> str:
    """`<speaker>-<chapter>-<index>`; validates every part."""
    if not speaker.isdigit():
        raise ValueError(f"speaker id {speaker!r} is not numeric")
    if not _CHAPTER.fullmatch(chapter):
        raise ValueError(f"chapter id {chapter!r} must match {_CHAPTER.pattern}")
    if index < 0:
        raise ValueError(f"segment index must be >= 0, got {index}")
    return f"{speaker}-{chapter}-{index}"


def parse_segment_id(s: str) -> tuple[str, str, int]:
    """Inverse of `segment_id`; raises ValueError if `s` is not exactly one."""
    head, _, index = s.rpartition("-")
    speaker, _, chapter = head.partition("-")
    if not speaker.isdigit() or not _CHAPTER.fullmatch(chapter) or not index.isdigit():
        raise ValueError(f"not a segment id: {s!r}")
    value = int(index)
    if segment_id(speaker, chapter, value) != s:
        raise ValueError(f"not a segment id: {s!r}")
    return speaker, chapter, value


def check_unique_chapters(files: Iterable[SourceFile]) -> None:
    """Raise if two different files map to the same `(speaker, chapter_id)`.

    Two archive paths can sanitise to one id (`a.b/c` and `a_b/c`), and then segment
    ids of different chapters would collide.
    """
    owners: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    for f in files:
        owners[(f.speaker, chapter_id(f.book_dir, f.stem))].add((f.speaker, f.book_dir, f.stem))
    collisions = {key: sorted(v) for key, v in owners.items() if len(v) > 1}
    if not collisions:
        return
    lines = [
        f"speaker {speaker}: chapter {chapter!r} comes from {', '.join(map(str, sources))}"
        for (speaker, chapter), sources in sorted(collisions.items())
    ]
    raise ValueError("duplicate chapter ids:\n" + "\n".join(lines))
