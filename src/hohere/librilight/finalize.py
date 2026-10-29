"""`hohere librilight finalize`: merge the per-part manifests, check them, pack shards.

Acceptance is a hard gate: ids, per-chapter numbering, exclusions, checksums, tensor
shape and the silent channel are all verified before anything is packed, and the
report goes to `release/checks.json` whether it passed or not. The shards are plain
tars of `tokens/<id>.pt` in manifest order, plus `manifest.jsonl` and
`silence_codes.json`, so a release is a folder plus one upload per few dozen files.
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hohere.librilight.activity import FRAME_SAMPLES, SAMPLE_RATE, SILENCE_CODES
from hohere.librilight.ids import parse_segment_id
from hohere.storage.manifest import sha256_file

log = logging.getLogger(__name__)

DATASET_SPLIT = "librilight.medium"
FRAME_RATE = SAMPLE_RATE / FRAME_SAMPLES
SHARD_BYTES = 1_000_000_000
PER_COMMIT = 20
UPLOAD_RETRIES = 5


@dataclass(frozen=True)
class Entry:
    """One manifest line and the `WORK/part-K` folder that holds its token file."""

    line: dict
    root: Path

    @property
    def path(self) -> Path:
        return self.root / self.line["tokens_path"]

    @property
    def parts(self) -> tuple[int, str, int]:
        speaker, chapter, index = parse_segment_id(self.line["id"])
        return int(speaker), chapter, index


def merge(work: Path) -> list[Entry]:
    """Every `part-*/done/*.json` line, sorted by speaker, chapter and index."""
    entries: list[Entry] = []
    for root in sorted(work.glob("part-*")):
        for marker in sorted((root / "done").glob("*.json")):
            for line in json.loads(marker.read_text()):
                entries.append(Entry(line, root))
    entries.sort(key=lambda e: e.parts)
    return entries


def write_manifest(path: Path, entries: Sequence[Entry]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry.line) + "\n")
    tmp.replace(path)


def _check_ids(entries: Sequence[Entry], problems: list[str]) -> dict[str, list[int]]:
    """Ids are unique and round-trip; per chapter the indices are exactly 0..n-1."""
    seen: set[str] = set()
    by_chapter: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        sid = entry.line["id"]
        if sid in seen:
            problems.append(f"duplicate id {sid}")
            continue
        seen.add(sid)
        try:
            _, chapter, index = parse_segment_id(sid)
        except ValueError as e:
            problems.append(str(e))
            continue
        by_chapter[chapter].append(index)
    for chapter, indices in by_chapter.items():
        if sorted(indices) != list(range(len(indices))):
            problems.append(f"chapter {chapter!r} has indices {sorted(indices)[:10]}")
    return by_chapter


def _check_sources(
    entries: Sequence[Entry],
    problems: list[str],
    exclude_speakers: Iterable[str],
    exclude_recordings: Iterable[Iterable[str]],
) -> None:
    speakers = {str(s) for s in exclude_speakers}
    recordings = {(str(s), str(t)) for s, t in exclude_recordings}
    for entry in entries:
        line = entry.line
        if line["user_voice"] in speakers:
            problems.append(f"excluded speaker {line['user_voice']} in {line['id']}")
        stem = Path(line["source"]).name.rpartition(".")[0]
        if (line["user_voice"], stem) in recordings:
            problems.append(f"excluded recording {stem} in {line['id']}")


def check_tokens(entry: Entry, problems: list[str]) -> None:
    """Checksum, keys, dtypes, shapes and the silent channel of one token file."""
    import torch  # lazy: the release machine has it, a Mac may not

    path = entry.path
    frames = int(entry.line["frames"])
    codes = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != entry.line["sha256"]:
        problems.append(f"checksum mismatch {path}")
    missing = {"user", "target", "user_activity", "activity"} - set(codes)
    if missing:
        problems.append(f"{path}: missing keys {sorted(missing)}")
        return
    for key in ("user", "target"):
        tensor = codes[key]
        if tensor.dtype != torch.int16:
            problems.append(f"{path}: {key} is {tensor.dtype}, expected int16")
        if tuple(tensor.shape) != (len(SILENCE_CODES), frames):
            problems.append(f"{path}: {key} is {tuple(tensor.shape)}, expected {(8, frames)}")
    for key in ("user_activity", "activity"):
        tensor = codes[key]
        if tensor.dtype != torch.uint8:
            problems.append(f"{path}: {key} is {tensor.dtype}, expected uint8")
        if int(tensor.shape[0]) != frames:
            problems.append(f"{path}: {key} has {int(tensor.shape[0])} frames, expected {frames}")
    silent = "A" if int(codes["user_activity"].sum()) == 0 else "B"
    codes_key = "user" if silent == "A" else "target"
    act_key = "user_activity" if silent == "A" else "activity"
    expected = torch.tensor(SILENCE_CODES, dtype=torch.int16)[:, None].expand(-1, frames)
    if not bool(torch.equal(codes[codes_key], expected)):
        problems.append(f"{path}: channel {silent} is not the silence codes")
    if int(codes[act_key].sum()) != 0:
        problems.append(f"{path}: channel {silent} has activity")


def check(
    entries: Sequence[Entry],
    exclude_speakers: Iterable[str] = (),
    exclude_recordings: Iterable[Iterable[str]] = (),
    check_tensor: bool = True,
) -> dict:
    """The acceptance report: `ok`, `problems` and the release statistics."""
    problems: list[str] = []
    chapters = _check_ids(entries, problems)
    _check_sources(entries, problems, exclude_speakers, exclude_recordings)
    missing_files = 0
    frames = 0
    skipped_tensors = 0
    for entry in entries:
        if not entry.path.exists():
            problems.append(f"missing {entry.path}")
            missing_files += 1
            continue
        frames += int(entry.line["frames"])
        if not check_tensor:
            continue
        try:
            check_tokens(entry, problems)
        except ImportError:
            skipped_tensors += 1
        except Exception as e:  # a corrupt file is a problem, not a crash mid-release
            problems.append(f"{entry.path}: {e}")
    speakers = Counter(entry.line["user_voice"] for entry in entries)
    chapter_frames: dict[str, int] = defaultdict(int)
    for entry in entries:
        chapter_frames[f"{entry.line['user_voice']}-{entry.parts[1]}"] += int(entry.line["frames"])
    report = {
        "ok": not problems,
        "problems": problems[:200],
        "problem_count": len(problems),
        "segments": len(entries),
        "speakers": len(speakers),
        "chapters": len(chapters),
        "hours": round(frames / FRAME_RATE / 3600, 4),
        "chapters_at_least_2048_frames": sum(1 for f in chapter_frames.values() if f >= 2048),
        "missing_files": missing_files,
    }
    if skipped_tensors:
        report["tensors_skipped"] = skipped_tensors
    return report


def pack_shards(
    entries: Sequence[Entry], out_dir: Path, shard_bytes: int = SHARD_BYTES
) -> list[Path]:
    """Tar the token files in manifest order, about `shard_bytes` per shard."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    tar: tarfile.TarFile | None = None
    written = 0
    for entry in entries:
        size = entry.path.stat().st_size
        if tar is None or (written and written + size > shard_bytes):
            if tar is not None:
                tar.close()
            path = out_dir / f"shard-{len(shards):04d}.tar"
            shards.append(path)
            # Shards are closed as soon as they are full, so a release never holds
            # hundreds of open file handles.
            tar = tarfile.open(path, "w")  # noqa: SIM115
            written = 0
        info = tarfile.TarInfo(f"tokens/{entry.line['id']}.pt")
        info.size = size
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        with open(entry.path, "rb") as f:
            tar.addfile(info, f)
        written += size
    if tar is not None:
        tar.close()
    return shards


def shard_index(path: Path) -> list[str]:
    """Member names of a shard, for the report and the tests."""
    with tarfile.open(path) as tar:
        return tar.getnames()


def silence_codes() -> dict:
    return {
        "codes": list(SILENCE_CODES),
        "codebooks": len(SILENCE_CODES),
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "frame_rate": FRAME_RATE,
    }


def _is_rate_limit(error: Exception) -> bool:
    text = str(error).lower()
    return getattr(error, "status_code", None) == 429 or "429" in text or "rate limit" in text


def upload(
    folder: Path,
    repo: str,
    *,
    private: bool = True,
    token_env: str = "HF_TOKEN",
    per_commit: int = PER_COMMIT,
    retries: int = UPLOAD_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Create the dataset repo if needed and upload `folder` in few-file commits."""
    from huggingface_hub import HfApi  # lazy: only the publishing machine has it

    api = HfApi(token=os.environ.get(token_env) or None)
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    names = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
    commits = 0
    for start in range(0, len(names), per_commit):
        chunk = names[start : start + per_commit]
        _upload_chunk(api, repo, folder, chunk, retries, sleep)
        commits += 1
    return {"repo": repo, "private": private, "files": len(names), "commits": commits}


def _upload_chunk(
    api,
    repo: str,
    folder: Path,
    names: Sequence[str],
    retries: int,
    sleep: Callable[[float], None],
) -> None:
    delay = 2.0
    for attempt in range(retries + 1):
        try:
            api.upload_folder(
                folder_path=str(folder),
                repo_id=repo,
                repo_type="dataset",
                allow_patterns=list(names),
                commit_message=f"{len(names)} files",
            )
            return
        except Exception as e:  # only a rate limit is retried
            if attempt == retries or not _is_rate_limit(e):
                raise
            log.warning("upload limited (%s), retrying in %.1f s", e, delay)
            sleep(delay)
            delay *= 2


def run_finalize(
    work: Path,
    out: Path | None = None,
    *,
    shard_bytes: int = SHARD_BYTES,
    split: str = DATASET_SPLIT,
    repo: str | None = None,
    private: bool = True,
    token_env: str = "HF_TOKEN",
    check_tensor: bool = True,
) -> dict:
    """Merge, check, pack and optionally upload one release."""
    work = Path(work)
    out = Path(out) if out is not None else work / "release"
    summary = _read_json(work / "index_summary.json")
    entries = merge(work)
    log.info("merged %d segments from %s", len(entries), work)
    report = check(
        entries,
        exclude_speakers=summary.get("exclude_speakers", ()),
        exclude_recordings=summary.get("exclude_recordings", ()),
        check_tensor=check_tensor,
    )
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "checks.tmp"
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out / "checks.json")
    if not report["ok"]:
        raise RuntimeError(
            f"{report['problem_count']} problems, see {out / 'checks.json'}:\n"
            + "\n".join(report["problems"][:10])
        )
    folder = out / split
    shards = pack_shards(entries, folder, shard_bytes)
    write_manifest(folder / "manifest.jsonl", entries)
    (folder / "silence_codes.json").write_text(json.dumps(silence_codes(), indent=2))
    result = {"report": report, "shards": [p.name for p in shards], "folder": str(folder)}
    if repo:
        result["upload"] = upload(folder, repo, private=private, token_env=token_env)
    return result


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)
