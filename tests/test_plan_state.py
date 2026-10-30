"""Plan-at-index-time: chunk counts, cum_hours, tiers and the final-order contract."""

from __future__ import annotations

import pytest

from hohere.librilight import SourceFile
from hohere.state import State, StateError


def source(file_id: str, split: str = "small", seconds: float = 120.0) -> SourceFile:
    spans = [(i * 11.0, i * 11.0 + 10.0) for i in range(max(1, int(seconds // 11)))]
    return SourceFile(
        file_id=file_id,
        split=split,
        speaker="1",
        book="b",
        archive_url="http://example/x.tar",
        flac_offset=0,
        flac_size=10,
        json_offset=10,
        json_size=10,
        duration_s=seconds,
        vad=spans,
    )


def test_plan_chunks_stores_counts_and_cumulative_hours(tmp_path):
    with State(tmp_path / "s.db") as state:
        state.add_files([source(f"small-1-b-f{i}") for i in range(4)])
        state.assign_order()
        report = state.plan_chunks()
        assert report["files"] == 4
        assert report["chunks"] >= 4
        assert report["chunk_hours"] > 0
        rows = state._all("SELECT n_chunks, chunk_hours, cum_hours FROM files ORDER BY order_rank")
        assert [row["n_chunks"] for row in rows] == sorted(row["n_chunks"] for row in rows)
        cums = [row["cum_hours"] for row in rows]
        assert cums == sorted(cums)
        assert cums[-1] == pytest.approx(report["chunk_hours"])
        assert state.get_meta("plan_hours") == pytest.approx(report["chunk_hours"])


def test_tier_hours_are_exact_before_any_encoding(tmp_path):
    with State(tmp_path / "s.db") as state:
        state.add_files([source(f"small-1-b-f{i}", seconds=3600) for i in range(3)])
        state.assign_order()
        state.plan_chunks()
        everything = state.tier_hours("full")
        assert everything["files"] == 3
        assert everything["hours"] > 2.5
        halfway = state.tier_hours(everything["hours"] / 2)
        assert 1 <= halfway["files"] < 3
        assert halfway["hours"] <= everything["hours"]


def test_select_until_is_the_cum_hours_prefix_with_the_crossing_file(tmp_path):
    with State(tmp_path / "s.db") as state:
        state.add_files([source(f"small-1-b-f{i}", seconds=3600) for i in range(4)])
        state.assign_order()
        state.plan_chunks()
        ordered = state.select_until("full")
        first_chunk_hours = state.cum_hours_of(ordered[0].file_id)
        tier = state.select_until(first_chunk_hours)
        assert [f.file_id for f in tier] == [ordered[0].file_id, ordered[1].file_id][: len(tier)]
        assert tier[0].file_id == ordered[0].file_id
        assert len(tier) <= 2  # the first file, plus the one that crosses the limit


def test_assign_order_is_final_and_extends_rather_than_shuffles(tmp_path):
    with State(tmp_path / "s.db") as state:
        state.add_files([source("small-1-b-a"), source("small-1-b-b")])
        state.assign_order()
        before = {row["file_id"]: row["order_rank"]
                  for row in state._all("SELECT file_id, order_rank FROM files")}
        state.add_files([source("small-1-b-c")])
        assert state.assign_order() == 1
        after = {row["file_id"]: row["order_rank"]
                 for row in state._all("SELECT file_id, order_rank FROM files")}
        assert {fid: after[fid] for fid in before} == before
        assert after["small-1-b-c"] == 2
        assert state.get_meta("plan_splits") == ["small"]


def test_a_split_that_is_not_indexed_is_an_error(tmp_path):
    with State(tmp_path / "s.db") as state:
        with pytest.raises(StateError, match="index"):
            state.assign_order()  # nothing indexed yet at all
        state.add_files([source("small-1-b-a")])
        state.assign_order()
        with pytest.raises(StateError, match="not indexed"):
            state.assign_order(["small", "large"])


def test_changing_the_split_set_after_planning_is_an_error(tmp_path):
    with State(tmp_path / "s.db") as state:
        state.add_files([source("small-1-b-a")])
        state.assign_order()
        state.add_files([source("medium-1-b-a", split="medium")])
        with pytest.raises(StateError, match="final"):
            state.assign_order()
