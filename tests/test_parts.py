"""A tier split across machines, and a machine that continues from the hub."""

from __future__ import annotations

import functools
import json
import operator
import types

import pytest

from hohere import order, pipeline, records, shards


def test_parts_cover_the_tier_once():
    files = list(range(10))
    parts = [order.part_of(files, i, 3) for i in range(3)]
    assert sorted(functools.reduce(operator.iadd, parts, [])) == files
    assert max(map(len, parts)) - min(map(len, parts)) <= 1
    with pytest.raises(ValueError):
        order.part_of(files, 3, 3)


def _manifest(tmp_path, name, ids):
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps({"id": i}) + "\n" for i in ids))
    return path


def test_uploaded_manifests_need_both_files_and_the_prefix(tmp_path):
    good = _manifest(tmp_path, "shard-000002", ["a-000"])
    api = types.SimpleNamespace(
        list_repo_files=lambda repo, repo_type: [
            "lb/part-001/shard-000002.jsonl",
            "lb/part-001/shard-000002.tar",
            "lb/part-001/shard-000003.jsonl",  # its tar never arrived
            "lb/part-002/shard-000009.jsonl",
            "lb/part-002/shard-000009.tar",
        ],
        hf_hub_download=lambda repo, name, repo_type: str(good),
    )
    uploader = shards.Uploader("r", "lb/part-001", api=api)
    assert uploader.uploaded_manifests() == [("shard-000002", [{"id": "a-000"}])]


def test_resume_marks_complete_files_and_finds_the_next_shard():
    done = {}
    state = types.SimpleNamespace(
        n_chunks_of=lambda file_id: {"whole": 2, "partial": 3}[file_id],
        set_status=lambda file_id, status: done.__setitem__(file_id, status),
    )
    ids = [
        records.chunk_id("whole", 0),
        records.chunk_id("whole", 1),
        records.chunk_id("partial", 0),
    ]
    uploader = types.SimpleNamespace(uploaded_manifests=lambda: [
        ("shard-000004", [{"id": i} for i in ids])
    ])
    files = [types.SimpleNamespace(file_id="whole"), types.SimpleNamespace(file_id="partial")]
    chunks, first = pipeline.resume_from_hub(uploader, state, files)
    assert done == {"whole": "done"}
    assert chunks == set(ids)
    assert first == 5
