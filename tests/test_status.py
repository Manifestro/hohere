"""Status payload and the HTTP page (a real server on port 0)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest
from test_state import source_file

from hohere.state import State
from hohere.status import Progress, StatusServer, iso


def get(url: str) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def test_snapshot_has_the_documented_shape():
    progress = Progress(device="cuda", target=100)
    progress.set_totals(files=10, source_hours=20.0)
    progress.add("download", 1.0)
    progress.add("download", 3.0)
    progress.done_chunk(0.25)
    progress.done_file(2.0)
    payload = progress.snapshot()

    assert payload["device"] == "cuda"
    assert payload["target"] == 100
    assert payload["totals"] == {"files": 10, "source_hours": 20.0}
    assert payload["done"]["chunks"] == 1
    assert payload["done"]["hours"] == 0.25
    assert payload["remaining"] == {"files": 9, "hours": 18.0}
    assert payload["stage_times"]["download"] == {"n": 2, "total_s": 4.0, "mean_s": 2.0}
    assert payload["stage_times"]["mimi"]["n"] == 0
    assert payload["tier"]["reached"] == 100
    assert set(payload["speed"]) == {"window_s", "rolling_10min", "overall"}
    assert set(payload["eta_s"]) == {"target", "full"}
    assert payload["errors"] == {"count": 0, "last": []}
    assert payload["shards"] == []
    assert payload["progress"]["fraction"] == pytest.approx(0.0025)


def test_errors_keep_the_last_twenty_messages():
    progress = Progress()
    for index in range(25):
        progress.error(f"boom {index}")
    assert progress.snapshot()["errors"]["count"] == 25
    last = progress.snapshot()["errors"]["last"]
    assert len(last) == 20
    assert last[-1] == "boom 24"


def test_rolling_speed_covers_the_last_ten_minutes():
    progress = Progress()
    progress.done_chunk(0.5)
    progress.done_chunk(0.5)
    with progress.lock:
        progress._window.clear()
        progress._window.append((0.0, 9.0))  # a stale sample must fall out of the window
        progress._window.append((1000.0, 0.5))
        progress._window.append((1300.0, 0.5))
    speed = progress.rolling_speed(now=1600.0)
    # the sample from t=0 falls out, the one exactly 10 minutes old still counts
    assert speed == pytest.approx(1.0 / (600.0 / 3600.0))


def test_recent_shards_are_capped_and_counted():
    progress = Progress()
    for index in range(12):
        progress.shard_uploaded({"name": f"shard-{index:06d}", "bytes": 100})
    payload = progress.snapshot()
    assert payload["done"]["shards_uploaded"] == 12
    assert payload["done"]["bytes_uploaded"] == 1200
    assert len(payload["shards"]) == 10
    assert payload["shards"][-1]["name"] == "shard-000011"


def test_snapshot_merges_the_durable_state(tmp_path):
    with State(tmp_path / "state.db") as state:
        state.add_files([source_file("small-1-b-x", 3600)])
        state.set_meta("started_at", iso(0.0))
        state.set_meta("target_hours", 100)
        state.set_shard("small-1-b-x", "shard-000000", status="done")
        state.close_shard(
            type(
                "S",
                (),
                {
                    "name": "shard-000000",
                    "bytes": 2048,
                    "sha256": "aa",
                    "jsonl_sha256": "bb",
                    "n_records": 4,
                    "hours": 0.75,
                },
            )()
        )
        payload = Progress().snapshot(state)
    assert payload["target"] == 100
    assert payload["totals"] == {"files": 1, "source_hours": 1.0}
    assert payload["done"]["files"] == 1
    assert payload["done"]["chunks"] == 4
    assert payload["done"]["hours"] == 0.75
    assert payload["remaining"] == {"files": 0, "hours": 0.0}
    assert payload["shards"][-1]["name"] == "shard-000000"


def test_the_server_serves_json_and_the_page():
    progress = Progress(device="cuda")
    progress.set_totals(1, 1.0)
    server = StatusServer(progress, None, host="127.0.0.1", port=0).start()
    try:
        code, body = get(f"http://127.0.0.1:{server.port}/api/status")
        assert code == 200
        assert json.loads(body)["device"] == "cuda"
        code, body = get(f"http://127.0.0.1:{server.port}/")
        assert code == 200
        assert "/api/status" in body and "viewport" in body
        assert get(f"http://127.0.0.1:{server.port}/nope")[0] == 404
    finally:
        server.stop()


def test_a_token_is_required_on_every_request():
    progress = Progress()
    server = StatusServer(progress, None, host="127.0.0.1", port=0, token="s3cret").start()
    try:
        base = f"http://127.0.0.1:{server.port}"
        assert get(f"{base}/api/status")[0] == 403
        assert get(f"{base}/")[0] == 403
        assert get(f"{base}/api/status?token=wrong")[0] == 403
        assert get(f"{base}/api/status?token=s3cret")[0] == 200
        assert get(f"{base}/?token=s3cret")[0] == 200
    finally:
        server.stop()
