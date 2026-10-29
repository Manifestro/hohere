from __future__ import annotations

import json
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hohere.librilight import finalize
from hohere.librilight.activity import SILENCE_CODES
from hohere.librilight.encode import manifest_line
from hohere.librilight.finalize import (
    CHECK_CHUNK,
    Entry,
    chapter_stems,
    check,
    check_entry,
    merge,
    pack_shards,
    run_finalize,
    shard_index,
    silence_codes,
    upload,
)
from hohere.librilight.ids import chapter_id, segment_id
from hohere.librilight.plan import PlanFile, write_plan

CODEC = "cd" * 32


def line(speaker: str, chapter: str, index: int, frames: int = 100) -> dict:
    return manifest_line(
        id=segment_id(speaker, chapter, index),
        speaker=speaker,
        frames=frames,
        sha256="00" * 32,
        codec_sha256=CODEC,
        start=0.0,
        end=frames / 12.5,
    )


def plan_file(speaker: str, stem: str) -> PlanFile:
    return PlanFile(
        speaker=speaker,
        book_dir="book-01",
        stem=stem,
        chapter=chapter_id("book-01", stem),
        flac_offset=1024,
        flac_size=1000,
        duration=20.0,
        snr=12.5,
        segments=((0.5, 14.5),),
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
        merge(work),
        exclude_speakers=["0064"],
        exclude_recordings=[["0064", "a"]],
        stems={"0064-book_01_a": {"a"}},
    )
    problems = "\n".join(report["problems"])
    assert "excluded speaker 0064" in problems
    assert "excluded recording a in 0064-book_01_a-0" in problems


def test_chapter_stems_reads_the_plan(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    files = [plan_file("0007", "a"), plan_file("0008", "b")]
    write_plan(work / "plan.jsonl", files)
    assert chapter_stems(work) == {"0007-book_01_a": {"a"}, "0008-book_01_b": {"b"}}
    assert chapter_stems(tmp_path / "missing") == {}


def test_run_finalize_looks_recordings_up_in_the_plan(tmp_path: Path) -> None:
    work = make_work(tmp_path / "work", {0: [line("0007", "book_01_a", 0)]})
    write_plan(work / "plan.jsonl", [plan_file("0007", "a")])
    (work / "index_summary.json").write_text(json.dumps({"exclude_recordings": [["0007", "a"]]}))
    with pytest.raises(RuntimeError, match="excluded recording a"):
        run_finalize(work, check_tensor=False)


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


def test_silence_codes_is_the_plain_list_of_codes() -> None:
    assert silence_codes() == [1049, 1700, 1626, 546, 306, 1443, 1871, 2008]
    assert silence_codes() == list(SILENCE_CODES)


def test_run_finalize_writes_the_release_and_the_report(tmp_path: Path) -> None:
    work = make_work(tmp_path / "work", {0: [line("0007", "book_01_a", i) for i in range(2)]})
    (work / "index_summary.json").write_text(json.dumps({"exclude_speakers": []}))
    result = run_finalize(work, check_tensor=False)
    folder = work / "release" / "librilight.medium"
    assert (folder / "manifest.jsonl").exists() and (folder / "silence_codes.json").exists()
    assert (work / "release" / "checks.json").exists()
    assert result["report"]["ok"]
    assert json.loads((folder / "silence_codes.json").read_text()) == silence_codes()
    assert (folder / "silence_codes.json").read_text() == (
        "[1049, 1700, 1626, 546, 306, 1443, 1871, 2008]"  # a plain list, not an object
    )
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


def many_work(tmp_path: Path, count: int, missing: int = 0) -> list[Entry]:
    """A `part-0` folder of `count` one-segment chapters, the last `missing` of them gone."""
    lines = [line("0007", f"book_{i:04d}_a", 0) for i in range(count)]
    work = make_work(tmp_path / "work", {0: lines})
    for entry in lines[count - missing :]:
        (work / "part-0" / "tokens" / f"{entry['id']}.pt").unlink()
    return merge(work)


def test_entry_check_is_one_picklable_result_per_entry(tmp_path: Path) -> None:
    work = make_work(tmp_path, {0: [line("0007", "book_01_a", 0, frames=7)]})
    entry = merge(work)[0]
    result = check_entry(entry, check_tensor=False)
    assert (result.problems, result.frames, result.missing) == ((), 7, False)
    (work / "part-0" / "tokens" / f"{entry.line['id']}.pt").unlink()
    missing = check_entry(entry, check_tensor=False)
    assert missing.missing and "missing" in missing.problems[0] and missing.frames == 0


def test_a_big_release_is_checked_in_a_process_pool(tmp_path: Path, monkeypatch) -> None:
    entries = many_work(tmp_path, CHECK_CHUNK + 44, missing=2)
    pools: list[int] = []

    class Spy(ProcessPoolExecutor):
        def map(self, *args, **kwargs):
            pools.append(self._max_workers)
            return super().map(*args, **kwargs)

    monkeypatch.setattr(finalize, "ProcessPoolExecutor", Spy)
    inline = check(entries, check_tensor=False, workers=1)  # explicit: no pool
    assert pools == []
    pooled = check(entries, check_tensor=False, workers=2)
    assert pools == [2]
    assert pooled == inline  # same aggregation, just spread over the workers
    assert pooled["segments"] == CHECK_CHUNK + 44
    assert pooled["missing_files"] == 2
    kept = len(entries) - 2  # a missing file contributes no frames
    assert pooled["hours"] == pytest.approx(round(kept * 100 / 12.5 / 3600, 4))
    assert sum("missing" in problem for problem in pooled["problems"]) == 2


def test_the_pool_skips_the_tensors_when_torch_is_missing(tmp_path: Path) -> None:
    try:
        import torch  # noqa: F401 - the pod has it, so there is nothing to skip
    except ImportError:
        pass
    else:
        pytest.skip("torch is installed: the pool checks the tensors instead of skipping")
    entries = many_work(tmp_path, CHECK_CHUNK)
    report = check(entries, workers=2)
    assert report["tensors_skipped"] == len(entries) and report["ok"]


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
