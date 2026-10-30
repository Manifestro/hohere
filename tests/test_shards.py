"""Shard writer and uploader."""

from __future__ import annotations

import json
import tarfile

import pytest
from helpers import record

from hohere import shards
from hohere.records import write_record
from hohere.shards import ClosedShard, ShardWriter, Uploader, iter_record_ids, read_member


def make_shard(writer: ShardWriter, chunk_id: str, seconds: float = 45.0) -> ClosedShard | None:
    raw = write_record(**record())
    return writer.add(chunk_id, raw, {"duration_s": seconds, "id": chunk_id})


def test_add_and_close_produce_a_tar_and_a_jsonl(tmp_path):
    writer = ShardWriter(tmp_path, target_bytes=1 << 30)
    for index in range(3):
        assert make_shard(writer, f"small-1-a-b-{index:03d}") is None
    closed = writer.close()
    assert closed is not None
    assert closed.name == "shard-000000"
    assert closed.n_records == 3
    assert closed.hours == pytest.approx(3 * 45 / 3600)
    assert closed.tar_path.exists() and closed.jsonl_path.exists()
    assert closed.bytes == closed.tar_path.stat().st_size + closed.jsonl_path.stat().st_size
    assert closed.sha256 == shards.sha256_file(closed.tar_path)

    ids = list(iter_record_ids(closed.tar_path))
    assert ids == [f"small-1-a-b-{index:03d}" for index in range(3)]
    assert read_member(closed.tar_path, ids[0]) == write_record(**record())

    lines = [json.loads(line) for line in closed.jsonl_path.read_text().splitlines()]
    assert [line["id"] for line in lines] == ids
    assert list(lines[0]) == ["duration_s", "id"]  # sort_keys=True


def test_members_are_deterministic(tmp_path):
    writer = ShardWriter(tmp_path, target_bytes=1 << 30)
    make_shard(writer, "small-1-a-b-000")
    closed = writer.close()
    with tarfile.open(closed.tar_path) as tar:
        info = tar.getmembers()[0]
        assert info.name == "small-1-a-b-000.npz"
        assert (info.mtime, info.uid, info.gid, info.uname, info.gname) == (0, 0, 0, "", "")
        assert info.mode == 0o644


def test_long_member_names_survive_the_gnu_format(tmp_path):
    writer = ShardWriter(tmp_path, target_bytes=1 << 30)
    long_id = "small-1-" + "b" * 90 + "-" + "c" * 90 + "-000"
    make_shard(writer, long_id)
    closed = writer.close()
    assert list(iter_record_ids(closed.tar_path)) == [long_id]


def test_the_writer_rolls_over_at_the_target_size(tmp_path):
    writer = ShardWriter(tmp_path, target_bytes=4000)
    rolled = []
    for index in range(12):
        shard = make_shard(writer, f"small-1-a-b-{index:03d}")
        if shard is not None:
            rolled.append(shard)
    writer.close()
    assert len(rolled) >= 2
    assert [shard.name for shard in rolled] == [f"shard-{i:06d}" for i in range(len(rolled))]
    every = []
    for shard in writer.closed:
        assert shard.tar_path.stat().st_size >= 4000 or shard is writer.closed[-1]
        every.extend(iter_record_ids(shard.tar_path))
    assert every == [f"small-1-a-b-{index:03d}" for index in range(12)]


class FakeApi:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.created: list[tuple[str, dict]] = []
        self.uploads: list[tuple[str, str, str, str]] = []

    def create_repo(self, repo_id: str, **kwargs) -> None:
        self.created.append((repo_id, kwargs))

    def upload_file(self, path_or_fileobj: str, path_in_repo: str, repo_id: str, repo_type: str):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("hub is down")
        self.uploads.append((path_or_fileobj, path_in_repo, repo_id, repo_type))


def closed_shard(tmp_path) -> ClosedShard:
    writer = ShardWriter(tmp_path, target_bytes=1 << 30)
    make_shard(writer, "small-1-a-b-000")
    closed = writer.close()
    assert closed is not None
    return closed


def test_uploader_uploads_both_files_then_deletes_them(tmp_path):
    shard = closed_shard(tmp_path)
    api = FakeApi()
    Uploader("bagzhan/hohere", "v2", api=api, sleep=lambda _s: None).upload(shard)
    assert api.created == [
        ("bagzhan/hohere", {"repo_type": "dataset", "private": True, "exist_ok": True})
    ]
    assert [path for _, path, _, _ in api.uploads] == [
        "v2/shard-000000.tar",
        "v2/shard-000000.jsonl",
    ]
    assert {(repo_id, repo_type) for _, _, repo_id, repo_type in api.uploads} == {
        ("bagzhan/hohere", "dataset")
    }
    assert not shard.tar_path.exists() and not shard.jsonl_path.exists()


def test_uploader_retries_and_can_keep_the_files(tmp_path):
    shard = closed_shard(tmp_path)
    api = FakeApi(failures=1)
    uploader = Uploader("bagzhan/hohere", api=api, retries=2, sleep=lambda _s: None, delete=False)
    uploader.upload(shard)
    assert len(api.uploads) == 2
    assert shard.tar_path.exists() and shard.jsonl_path.exists()


def test_uploader_gives_up_after_the_retry_budget(tmp_path):
    shard = closed_shard(tmp_path)
    api = FakeApi(failures=10)
    uploader = Uploader("bagzhan/hohere", api=api, retries=1, sleep=lambda _s: None)
    with pytest.raises(RuntimeError, match="cannot upload"):
        uploader.upload(shard)
    assert shard.tar_path.exists()
