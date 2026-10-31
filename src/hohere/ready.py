"""`hohere ready`: is a tier whole on the hub, and under one set of centroids?

A run never says "the tier is finished": a file that failed is skipped, and a shard can
fail to upload. So the plan is compared with the manifests on the hub, and only a tier
that passes gets a `READY-<tier>.json` next to its shards. That file is what a reader
waits for; its `last_order_rank` selects the tier: every record with `order_rank` up to
it, whatever else the repo holds by then.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from . import records

UNIT_KEYS = ("units_k4096", "units_k8192")


def tier_name(until_hours: float | str) -> str:
    """`500.0` -> `500`, `full` stays `full`."""
    return until_hours if isinstance(until_hours, str) else f"{until_hours:g}"


def check(
    tier: str,
    planned: list[tuple[str, int, int]],
    manifests: Iterable[tuple[str, list[dict]]],
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Compare `(file_id, n_chunks, order_rank)` of a tier with `(shard, lines)` of the hub."""
    wanted = {
        records.chunk_id(file_id, index): file_id
        for file_id, n_chunks, _rank in planned
        for index in range(n_chunks)
    }
    seen: set[str] = set()
    duplicates: set[str] = set()
    versions: dict[str, set] = {key: set() for key in UNIT_KEYS}
    seconds = 0.0
    for _shard, lines in manifests:
        for line in lines:
            if line["id"] not in wanted:
                continue  # a later tier: the repo may hold more than this one
            if line["id"] in seen:
                duplicates.add(line["id"])
            seen.add(line["id"])
            seconds += float(line.get("duration_s") or 0.0)
            for key in UNIT_KEYS:
                versions[key].add(line.get(key))
    missing_files = sorted({wanted[chunk] for chunk in wanted.keys() - seen})
    one_version = all(len(values) == 1 and None not in values for values in versions.values())
    return {
        "tier": tier,
        "ready": one_version and not duplicates and (allow_missing or not missing_files),
        "files": len(planned),
        "chunks": len(wanted),
        "chunks_found": len(seen),
        "hours": round(seconds / 3600.0, 3),
        "last_order_rank": max((rank for _id, _n, rank in planned), default=None),
        "missing_files": missing_files,
        "duplicates": sorted(duplicates),
        **{key: sorted(map(str, values)) for key, values in versions.items()},
    }


def publish(uploader: Any, report: dict[str, Any]) -> str:
    """Put a passed report on the hub; returns its path in the repo."""
    if not report["ready"]:
        raise ValueError(f"tier {report['tier']} is not ready")
    path = uploader.path_in_repo(f"READY-{report['tier']}.json")
    uploader.api().upload_file(
        path_or_fileobj=(json.dumps(report, indent=2) + "\n").encode(),
        path_in_repo=path,
        repo_id=uploader.repo_id,
        repo_type="dataset",
    )
    return path
