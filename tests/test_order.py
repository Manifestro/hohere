"""Deterministic order and tier helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from hohere import order


@dataclass
class Fake:
    file_id: str
    duration_s: float


def test_order_key_is_the_sha256_of_the_id():
    import hashlib

    assert order.order_key("small-1-a-b") == hashlib.sha256(b"small-1-a-b").hexdigest()
    assert order.order_key("a") != order.order_key("b")


def test_rank_is_stable_and_zero_based():
    ids = ["small-3-x-y", "small-1-a-b", "small-2-c-d"]
    ranks = order.rank(ids)
    assert sorted(ranks.values()) == [0, 1, 2]
    assert ranks == order.rank(list(reversed(ids)))
    assert order.sort_ids(ids) == sorted(ids, key=order.order_key)


def test_tier_of_returns_the_first_reached_tier():
    assert order.tier_of(0) == 100
    assert order.tier_of(99.9) == 100
    assert order.tier_of(100) == 200
    assert order.tier_of(500) == 1000
    assert order.tier_of(4999) == 5000
    assert order.tier_of(5000) == "full"
    assert order.tier_of(10_000) == "full"


def test_tier_helpers_reject_unknown_tiers():
    assert order.tier_index(100) == 0
    assert order.tier_index("full") == len(order.TIERS) - 1
    assert order.tier_hours("full") is None
    assert order.tier_hours(200) == 200.0
    with pytest.raises(ValueError):
        order.tier_index(42)


def test_files_until_counts_the_prefix_that_covers_the_hours():
    files = [Fake(f"f{i}", 1800.0) for i in range(10)]  # every file is 30 minutes
    assert order.files_until(files, 0) == 0
    assert order.files_until(files, 0.5) == 1
    assert order.files_until(files, 1.0) == 2
    assert order.files_until(files, 100) == len(files)
    assert order.files_for_tier(files, "full") == len(files)
    assert order.files_for_tier(files, 100) == len(files)  # a tier beyond the data


def test_sort_files_uses_the_file_id_digest():
    files = [Fake("b", 1.0), Fake("a", 2.0), Fake("c", 3.0)]
    assert [f.file_id for f in order.sort_files(files)] == order.sort_ids(["a", "b", "c"])
