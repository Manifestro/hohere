"""The deterministic order of all source files and the tiers built on it.

Every file is ranked by `sha256(file_id)`. The order never changes, so "the first N
hours of the order" is a well defined tier: tier 100 is the first 100 hours of kept
chunk audio, and a chunk recorded later still remembers its `order_rank` and
`cum_hours` in the manifest, which is what makes any tier selectable after the fact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

TIERS: tuple[Any, ...] = (100, 200, 500, 1000, 5000, "full")
HOUR = 3600.0


class HasFileId(Protocol):
    file_id: str
    duration_s: float


def order_key(file_id: str) -> str:
    """The sort key of a file: the hex digest of its id."""
    return hashlib.sha256(file_id.encode("utf-8")).hexdigest()


def rank(file_ids: Iterable[str]) -> dict[str, int]:
    """`file_id -> order_rank`, starting at 0."""
    ordered = sorted(file_ids, key=lambda fid: (order_key(fid), fid))
    return {fid: idx for idx, fid in enumerate(ordered)}


def sort_ids(file_ids: Iterable[str]) -> list[str]:
    return sorted(file_ids, key=lambda fid: (order_key(fid), fid))


def sort_files(files: Iterable[HasFileId]) -> list[Any]:
    return sorted(files, key=lambda f: (order_key(f.file_id), f.file_id))


def tier_index(tier: int | str) -> int:
    """Position of a tier in `TIERS` ("full" last)."""
    try:
        return TIERS.index(tier)
    except ValueError:
        raise ValueError(f"unknown tier {tier!r}, expected one of {TIERS}") from None


def tier_of(cumulative_hours: float) -> int | str:
    """Smallest tier that `cumulative_hours` has already reached, else "full"."""
    for tier in TIERS:
        if tier == "full" or cumulative_hours < float(tier):  # type: ignore[arg-type]
            return tier
    return "full"  # pragma: no cover - the loop always returns


def tier_hours(tier: int | str) -> float | None:
    """Target hours of a tier, None for "full"."""
    tier_index(tier)
    return None if tier == "full" else float(tier)  # type: ignore[arg-type]


def files_until(files: Sequence[HasFileId], hours: float) -> int:
    """How many leading files of `files` cover `hours` of audio (0 for `hours <= 0`)."""
    if hours <= 0:
        return 0
    budget = hours * HOUR
    total = 0.0
    for count, file in enumerate(files, start=1):
        total += float(file.duration_s)
        if total >= budget:
            return count
    return len(files)


def files_for_tier(files: Sequence[HasFileId], tier: int | str) -> int:
    """How many leading files belong to `tier`."""
    hours = tier_hours(tier)
    return len(files) if hours is None else files_until(files, hours)
