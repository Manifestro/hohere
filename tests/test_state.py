"""SQLite resume state: index cache, order, shard lifecycle and recovery."""

from __future__ import annotations

import pytest

from hohere import order
from hohere.librilight import SourceFile
from hohere.shards import ClosedShard
from hohere.state import State


def source_file(file_id: str, duration_s: float = 1800.0) -> SourceFile:
    return SourceFile(
        file_id=file_id,
        split="small",
        speaker="1",
        book="b",
        archive_url="https://example/small.tar",
        flac_offset=512,
        flac_size=1000,
        json_offset=4096,
        json_size=64,
        duration_s=duration_s,
        vad=[(0.0, 10.0)],
    )


def closed_shard(name: str = "shard-000000", n_records: int = 2, hours: float = 0.5) -> ClosedShard:
    from pathlib import Path

    return ClosedShard(
        name=name,
        tar_path=Path(f"/nonexistent/{name}.tar"),
        jsonl_path=Path(f"/nonexistent/{name}.jsonl"),
        bytes=1024,
        sha256="aa",
        jsonl_sha256="bb",
        n_records=n_records,
        hours=hours,
    )


def test_files_are_cached_with_their_vad(tmp_path):
    with State(tmp_path / "state.db") as state:
        assert not state.has_files("small")
        added = state.add_files([source_file("small-1-b-x"), source_file("small-1-b-y", 3600)])
        assert added == 2
        assert state.has_files("small")
        assert state.add_files([source_file("small-1-b-x")]) == 0  # idempotent
        files = state.files(split="small")
        assert [f.file_id for f in files] == ["small-1-b-x", "small-1-b-y"]
        assert files[0].vad == [(0.0, 10.0)]
        assert state.count_files(status="todo") == 2
        assert state.source_hours() == pytest.approx(1.5)


def test_assign_order_ranks_by_sha256(tmp_path):
    ids = ["small-3-x-y", "small-1-a-b", "small-2-c-d"]
    with State(tmp_path / "state.db") as state:
        state.add_files([source_file(fid) for fid in ids])
        assert state.assign_order() == 3
        ranked = [f.file_id for f in state.ordered_files()]
        assert ranked == order.sort_ids(ids)


def test_a_file_is_done_only_when_its_shard_is_closed_or_uploaded(tmp_path):
    with State(tmp_path / "state.db") as state:
        state.add_files([source_file("small-1-b-x"), source_file("small-1-b-y")])
        state.open_shard("shard-000000")
        state.set_shard("small-1-b-x", "shard-000000")
        assert state.count_files(status="done") == 0

        state.close_shard(closed_shard())
        assert state.count_files(status="done") == 1
        assert state.shard_of("small-1-b-x") == "shard-000000"

        state.mark_uploaded("shard-000000")
        summary = state.summary()
        assert summary["shards_uploaded"] == 1
        assert summary["bytes_uploaded"] == 1024
        assert summary["chunks"] == 2
        assert summary["hours"] == pytest.approx(0.5)
        assert summary["files_done"] == 1
        assert summary["remaining_files"] == 1
        assert state.shards(status="uploaded")[0]["sha256"] == "aa"


def test_recover_discards_open_shards_and_their_files(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    tar = work / "shard-000007.tar"
    tar.write_bytes(b"partial")
    (work / "shard-000007.jsonl").write_text("{}\n")
    with State(tmp_path / "state.db", out_dir=work) as state:
        state.add_files([source_file("small-1-b-x"), source_file("small-1-b-y")])
        state.open_shard("shard-000007")
        state.set_shard("small-1-b-x", "shard-000007")
        assert state.last_shards()[-1]["status"] == "open"
        dropped = state.recover()
        assert dropped == ["shard-000007"]
        assert state.shards() == []
        assert state.shard_of("small-1-b-x") is None
        assert state.count_files(status="todo") == 2
        assert not tar.exists() and not (work / "shard-000007.jsonl").exists()


def test_recover_is_a_noop_without_open_shards(tmp_path):
    with State(tmp_path / "state.db") as state:
        state.close_shard(closed_shard())
        assert state.recover() == []
        assert len(state.shards(status="closed")) == 1


def test_next_shard_index_continues_after_a_restart(tmp_path):
    with State(tmp_path / "state.db") as state:
        assert state.next_shard_index() == 0
        state.close_shard(closed_shard("shard-000004"))
        state.close_shard(closed_shard("shard-000011"))
        assert state.next_shard_index() == 12


def test_meta_round_trips_json(tmp_path):
    with State(tmp_path / "state.db") as state:
        assert state.get_meta("missing", "fallback") == "fallback"
        state.set_meta("target_hours", 100)
        assert state.get_meta("target_hours") == 100
        state.set_meta("target_hours", "full")
        assert state.get_meta("target_hours") == "full"
        assert state.bump_meta("chunks", 2) == 2
        assert state.bump_meta("chunks", 3) == 5


def test_last_shards_returns_the_newest_first(tmp_path):
    with State(tmp_path / "state.db") as state:
        for index in range(12):
            state.close_shard(closed_shard(f"shard-{index:06d}"))
        newest = state.last_shards(3)
        assert [shard["name"] for shard in newest] == [
            "shard-000009",
            "shard-000010",
            "shard-000011",
        ]
