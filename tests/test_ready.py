"""A tier is whole on the hub, under one set of centroids, or it is not ready."""

from __future__ import annotations

import json
import types

import pytest

from hohere import ready

PLANNED = [("a", 2, 0), ("b", 1, 1), ("empty", 0, 2)]


def _line(record_id: str, seconds: float = 3600.0, k4096: str = "x", k8192: str = "y") -> dict:
    return {"id": record_id, "duration_s": seconds, "units_k4096": k4096, "units_k8192": k8192}


def test_a_whole_tier_is_ready_and_later_tiers_do_not_matter():
    manifests = [
        ("shard-000000", [_line("a-000"), _line("b-000")]),
        ("shard-000000", [_line("a-001"), _line("later-000", k4096="other")]),
    ]
    report = ready.check("500", PLANNED, manifests)
    assert report["ready"]
    assert report["chunks"] == report["chunks_found"] == 3
    assert report["hours"] == 3.0
    assert report["last_order_rank"] == 2
    assert report["units_k4096"] == ["x"]


def test_a_missing_chunk_is_named_and_can_be_allowed():
    manifests = [("shard-000000", [_line("a-000"), _line("b-000")])]
    report = ready.check("500", PLANNED, manifests)
    assert not report["ready"]
    assert report["missing_files"] == ["a"]
    assert ready.check("500", PLANNED, manifests, allow_missing=True)["ready"]


def test_duplicates_and_mixed_centroids_never_pass():
    whole = [_line("a-000"), _line("a-001"), _line("b-000")]
    twice = ready.check("500", PLANNED, [("s", whole), ("s", [_line("b-000")])], allow_missing=True)
    assert twice["duplicates"] == ["b-000"]
    assert not twice["ready"]
    mixed = [*whole[:2], _line("b-000", k8192="other")]
    report = ready.check("500", PLANNED, [("s", mixed)], allow_missing=True)
    assert report["units_k8192"] == ["other", "y"]
    assert not report["ready"]
    unnamed = [{"id": "a-000"}, {"id": "a-001"}, {"id": "b-000"}]
    assert not ready.check("500", PLANNED, [("s", unnamed)])["ready"]


def test_tier_names():
    assert ready.tier_name(500.0) == "500"
    assert ready.tier_name(0.5) == "0.5"
    assert ready.tier_name("full") == "full"


def test_publish_writes_only_a_passed_report():
    sent = {}
    uploader = types.SimpleNamespace(
        repo_id="r",
        path_in_repo=lambda name: f"librilight/{name}",
        api=lambda: types.SimpleNamespace(upload_file=lambda **kwargs: sent.update(kwargs)),
    )
    with pytest.raises(ValueError):
        ready.publish(uploader, {"ready": False, "tier": "500"})
    assert not sent
    report = {"ready": True, "tier": "500"}
    assert ready.publish(uploader, report) == "librilight/READY-500.json"
    assert json.loads(sent["path_or_fileobj"]) == report
    assert sent["repo_type"] == "dataset"
