from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from hohere.librilight.encode import manifest_line
from hohere.librilight.finalize import (
    Entry,
    check,
    merge,
    pack_shards,
    run_finalize,
    shard_index,
    silence_codes,
    upload,
)
from hohere.librilight.ids import segment_id

CODEC = "cd" * 32


def line(speaker: str, chapter: str, index: int, frames: int = 100) -> dict:
    return manifest_line(
        id=segment_id(speaker, chapter, index),
        source=f"medium/{speaker}/{chapter.split('_')[0]}/{chapter}.flac",
        speaker=speaker,
        frames=frames,
        sha256="00" * 32,
        codec_sha256=CODEC,
        start=0.0,
        end=frames / 12.5,
    )


def make_work(root: Path, parts: dict[int, list[dict]], payload: bytes = b"x" * 400) -> Path:
    """A `WORK` folder with `part-K/done/<chapter>.json` markers and fake token files."""
    for part, lines in parts.items():
        done = root / f"part-{part}" / "done"
        done.mkdir(parents=True)
        tokens_dir = root / f"part-{part}" / "tokens"
        tokens_dir.mkdir()
        by_chapter: dict[str, list[dict]] = {}
        for entry in lines:
            (tokens_dir / f"{entry['id']}.pt").write_bytes(payload)
            by_chapter.setdefault(entry["id"].rsplit("-", 1)[0], []).append(entry)
        for chapter, chunk in by_chapter.items():
            (done / f"{chapter}.json").write_text(json.dumps(chunk))
    return root


def test_merge_sorts_by_speaker_chapter_and_index(tmp_path: Path) -> None:
    work = make_work(
        tmp_path,
        {
            1: [line("0007", "book_01_b", 0), line("0064", "book_01_a", 1)],
            0: [line("0064", "book_01_a", 0), line("0007", "book_01_a", 2)],
        },
    )
    entries = merge(work)
    assert [e.line["id"] for e in entries] == [
        "0007-book_01_a-2",
        "0007-book_01_b-0",
        "0064-book_01_a-0",
        "0064-book_01_a-1",
    ]
    assert all(isinstance(e, Entry) and e.root.name.startswith("part-") for e in entries)
    assert entries[0].parts == (7, "book_01_a", 2)


def test_merge_is_empty_for_a_missing_part_folder(tmp_path: Path) -> None:
    assert merge(tmp_path) == []


def test_check_accepts_a_consistent_release(tmp_path: Path) -> None:
    work = make_work(tmp_path, {0: [line("0007", "book_01_a", 0), line("0007", "book_01_a", 1)]})
    report = check(merge(work), check_tensor=False)
    assert report["ok"] and report["problems"] == []
    assert (report["segments"], report["speakers"], report["chapters"]) == (2, 1, 1)
    assert report["hours"] == pytest.approx(round(200 / 12.5 / 3600, 4))


def test_check_reports_duplicate_ids_bad_indices_and_missing_files(tmp_path: Path) -> None:
    lines = [line("0007", "book_01_a", 1), line("0007", "book_01_a", 1)]
    work = make_work(tmp_path, {0: lines[:1]})
    entries = [*merge(work), Entry(lines[1], work / "part-0")]
    (work / "part-0" / "tokens" / f"{lines[1]['id']}.pt").unlink(missing_ok=True)
    report = check(entries, check_tensor=False)
    assert not report["ok"]
    text = "\n".join(report["problems"])
    assert "duplicate id" in text and "missing" in text
    assert "book_01_a" in text  # indices are not 0..n-1


def test_check_reports_excluded_speakers_and_recordings(tmp_path: Path) -> None:
    work = make_work(tmp_path, {0: [line("0064", "book_01_a", 0)]})
    report = check(
        merge(work), exclude_speakers=["0064"], exclude_recordings=[["0064", "book_01_a"]]
    )
    problems = "\n".join(report["problems"])
    assert "excluded speaker 0064" in problems
    assert "excluded recording book_01_a" in problems


def test_pack_shards_keeps_manifest_order_and_shard_size(tmp_path: Path) -> None:
    lines = [line("0007", "book_01_a", i) for i in range(4)]
    work = make_work(tmp_path, {0: lines}, payload=b"x" * 400)
    entries = merge(work)
    shards = pack_shards(entries, tmp_path / "release", shard_bytes=1000)
    assert [p.name for p in shards] == ["shard-0000.tar", "shard-0001.tar"]
    assert shard_index(shards[0]) == [f"tokens/{e.line['id']}.pt" for e in entries[:2]]
    assert shard_index(shards[1]) == [f"tokens/{e.line['id']}.pt" for e in entries[2:]]
    for path in shards:  # members fit the limit, the tar is padded to its 10 KiB record
        with tarfile.open(path) as tar:
            assert sum(m.size for m in tar.getmembers()) <= 1000
            assert path.stat().st_size % 10240 == 0
    with tarfile.open(shards[0]) as tar:
        assert tar.getmember(f"tokens/{entries[0].line['id']}.pt").mtime == 0


def test_pack_shards_puts_every_file_once(tmp_path: Path) -> None:
    lines = [line("0007", "book_01_a", i) for i in range(3)]
    work = make_work(tmp_path, {0: lines}, payload=b"x" * 4000)
    shards = pack_shards(merge(work), tmp_path / "release", shard_bytes=10_000)
    members = [name for path in shards for name in shard_index(path)]
    assert members == [f"tokens/{entry.line['id']}.pt" for entry in merge(work)]
    assert len(shards) == 2  # two 4000-byte files fit, the third starts a shard


def test_run_finalize_writes_the_release_and_the_report(tmp_path: Path) -> None:
    work = make_work(tmp_path / "work", {0: [line("0007", "book_01_a", i) for i in range(2)]})
    (work / "index_summary.json").write_text(json.dumps({"exclude_speakers": []}))
    result = run_finalize(work, check_tensor=False)
    folder = work / "release" / "librilight.medium"
    assert (folder / "manifest.jsonl").exists() and (folder / "silence_codes.json").exists()
    assert (work / "release" / "checks.json").exists()
    assert result["report"]["ok"]
    assert json.loads((folder / "silence_codes.json").read_text()) == silence_codes()
    assert [json.loads(x)["id"] for x in (folder / "manifest.jsonl").read_text().splitlines()] == [
        "0007-book_01_a-0",
        "0007-book_01_a-1",
    ]


def test_run_finalize_refuses_a_bad_release(tmp_path: Path) -> None:
    work = make_work(tmp_path / "work", {0: [line("0007", "book_01_a", 0)]})
    (work / "part-0" / "tokens" / "0007-book_01_a-0.pt").unlink()
    with pytest.raises(RuntimeError, match="problems"):
        run_finalize(work, check_tensor=False)
    assert json.loads((work / "release" / "checks.json").read_text())["ok"] is False


class FakeApi:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.created: list[str] = []
        self.commits: list[list[str]] = []

    def create_repo(self, repo: str, repo_type: str, private: bool, exist_ok: bool) -> None:
        self.created.append(f"{repo}:{private}")

    def upload_folder(self, folder_path: str, repo_id: str, **kwargs) -> None:
        if self.failures:
            self.failures -= 1
            raise RuntimeError("429 Client Error: rate limit exceeded")
        self.commits.append(kwargs["allow_patterns"])


def test_upload_batches_commits_and_retries_a_rate_limit(tmp_path: Path, monkeypatch) -> None:
    folder = tmp_path / "release"
    folder.mkdir()
    for i in range(25):
        (folder / f"file{i:02d}.json").write_text("{}")
    fake = FakeApi(failures=2)
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub", SimpleNamespace(HfApi=lambda **_: fake)
    )
    result = upload(folder, "someone/dataset", per_commit=20, sleep=lambda _: None)
    assert result == {"repo": "someone/dataset", "private": True, "files": 25, "commits": 2}
    assert fake.commits == [
        [f"file{i:02d}.json" for i in range(20)],
        [f"file{i:02d}.json" for i in range(20, 25)],
    ]
    assert fake.created == ["someone/dataset:True"]


def test_token_checks_need_torch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    codes = torch.tensor([1049, 1700, 1626, 546, 306, 1443, 1871, 2008], dtype=torch.int16)[
        :, None
    ].expand(-1, 4)
    payload = {
        "user": codes,
        "target": codes,
        "user_activity": torch.zeros(4, dtype=torch.uint8),
        "activity": torch.zeros(4, dtype=torch.uint8),
    }
    work = make_work(tmp_path, {0: [line("0007", "book_01_a", 0, frames=4)]})
    path = work / "part-0" / "tokens" / "0007-book_01_a-0.pt"
    torch.save(payload, path)
    entries = merge(work)
    from hohere.storage.manifest import sha256_file

    entries[0].line["sha256"] = sha256_file(path)
    report = check(entries)
    assert report["ok"], report["problems"]
    path.write_bytes(path.read_bytes())  # unchanged bytes, still valid
    entries[0].line["sha256"] = "00" * 32
    report = check(entries)
    assert any("checksum mismatch" in p for p in report["problems"])
