"""Shard writer and uploader.

A shard is a pair of files: `shard-NNNNNN.tar` (uncompressed, members added in a
fixed order, `mtime=0`) and `shard-NNNNNN.jsonl` (one manifest line per member). The
writer rolls over as soon as the tar reaches `target_bytes`, so the disk footprint is
one shard at a time.

The uploader pushes a closed shard to a private Hugging Face dataset repo and only
then deletes the local files; with `--no-upload` the caller simply skips it.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

GIB = 1 << 30
UPLOAD_RETRIES = 5
UPLOAD_BACKOFF = 2.0


@dataclass(frozen=True)
class ClosedShard:
    """A finalised shard pair on disk, with the digests the state DB records."""

    name: str
    tar_path: Path
    jsonl_path: Path
    bytes: int
    sha256: str
    jsonl_sha256: str
    n_records: int
    hours: float

    def paths(self) -> tuple[Path, Path]:
        return self.tar_path, self.jsonl_path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk):
            digest.update(data)
    return digest.hexdigest()


def shard_name(index: int) -> str:
    return f"shard-{index:06d}"


def manifest_line(meta: dict) -> str:
    """One deterministic JSON line for a record."""
    return json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n"


class ShardWriter:
    """Append records until the tar is big enough, then finalise and start the next."""

    def __init__(self, out_dir: str | os.PathLike[str], target_bytes: int = GIB, index: int = 0):
        self.out_dir = Path(out_dir)
        self.target_bytes = target_bytes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._index = index
        self._tar: tarfile.TarFile | None = None
        self._jsonl: io.TextIOWrapper | None = None
        self.name = ""
        self.tar_bytes = 0
        self.n_records = 0
        self.hours = 0.0
        self.closed: list[ClosedShard] = []

    # -- lifecycle ---------------------------------------------------------------
    @property
    def tar_path(self) -> Path:
        return self.out_dir / f"{self.name}.tar"

    @property
    def jsonl_path(self) -> Path:
        return self.out_dir / f"{self.name}.jsonl"

    @property
    def is_open(self) -> bool:
        return self._tar is not None

    def open(self) -> str:
        """Start writing `shard-NNNNNN`; returns its name (idempotent while open)."""
        if self._tar is None:
            self._open_new()
        return self.name

    def _open_new(self) -> None:
        self.name = shard_name(self._index)
        self._index += 1
        # both handles stay open across add() calls and are closed by close()
        self._tar = tarfile.open(  # noqa: SIM115
            self.tar_path, mode="w", format=tarfile.GNU_FORMAT, encoding="utf-8"
        )
        self._jsonl = self.jsonl_path.open("w", encoding="utf-8", newline="\n")
        self.tar_bytes = 0
        self.n_records = 0
        self.hours = 0.0

    def add(self, record_id: str, data: bytes, meta: dict) -> ClosedShard | None:
        """Append `<record_id>.npz` and its manifest line; returns a shard if it rolled."""
        if self._tar is None or self._jsonl is None:
            self._open_new()
        assert self._tar is not None and self._jsonl is not None
        info = tarfile.TarInfo(f"{record_id}.npz")
        info.size = len(data)
        info.mtime = 0
        info.mode = 0o644
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        self._tar.addfile(info, io.BytesIO(data))
        payload = dict(meta)
        payload.setdefault("id", record_id)
        self._jsonl.write(manifest_line(payload))
        self.n_records += 1
        self.hours += float(payload.get("duration_s") or 0.0) / 3600.0
        self.tar_bytes = self.tar_path.stat().st_size
        if self.tar_bytes >= self.target_bytes:
            return self.close()
        return None

    def close(self) -> ClosedShard | None:
        """Finalise the open shard (if any) and return it."""
        if self._tar is None:
            return None
        self._tar.close()
        assert self._jsonl is not None
        self._jsonl.close()
        self._tar = None
        self._jsonl = None
        shard = ClosedShard(
            name=self.name,
            tar_path=self.tar_path,
            jsonl_path=self.jsonl_path,
            bytes=self.tar_path.stat().st_size + self.jsonl_path.stat().st_size,
            sha256=sha256_file(self.tar_path),
            jsonl_sha256=sha256_file(self.jsonl_path),
            n_records=self.n_records,
            hours=self.hours,
        )
        self.closed.append(shard)
        log.info(
            "closed %s: %d records, %.2f h, %.2f GB",
            shard.name,
            shard.n_records,
            shard.hours,
            shard.bytes / GIB,
        )
        return shard

    def __enter__(self) -> ShardWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def iter_record_ids(tar_path: Path) -> Iterator[str]:
    """Record ids of a shard tar, in member order (used by checks and tests)."""
    with tarfile.open(tar_path, mode="r") as tar:
        for member in tar:
            name = member.name
            yield name[: -len(".npz")] if name.endswith(".npz") else name


def read_member(tar_path: Path, record_id: str) -> bytes:
    with tarfile.open(tar_path, mode="r") as tar:
        handle = tar.extractfile(f"{record_id}.npz")
        if handle is None:  # pragma: no cover - defensive
            raise KeyError(record_id)
        return handle.read()


class Uploader:
    """Upload closed shards to `repo_id` (dataset repo, private if created here)."""

    def __init__(
        self,
        repo_id: str,
        prefix: str = "",
        *,
        token: str | None = None,
        api: object | None = None,
        retries: int = UPLOAD_RETRIES,
        backoff: float = UPLOAD_BACKOFF,
        sleep: Callable[[float], None] = time.sleep,
        delete: bool = True,
    ) -> None:
        self.repo_id = repo_id
        self.prefix = prefix.strip("/")
        self.token = token
        self._api = api
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.delete = delete
        self._repo_ready = False

    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.token)
        return self._api

    def path_in_repo(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def ensure_repo(self) -> None:
        if self._repo_ready:
            return
        self.api().create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
        self._repo_ready = True

    def _upload(self, path: Path) -> None:
        self.ensure_repo()
        delay = self.backoff
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self.api().upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=self.path_in_repo(path.name),
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
                return
            except Exception as exc:  # network and hub errors alike are worth a retry
                last = exc
                if attempt == self.retries:
                    break
                log.warning("upload %s failed (%s), retry %d", path.name, exc, attempt + 1)
                self.sleep(delay)
                delay *= 2
        raise RuntimeError(f"cannot upload {path} after {self.retries + 1} tries: {last}")

    def upload(self, shard: ClosedShard) -> None:
        """Upload both files, then delete them locally."""
        for path in shard.paths():
            self._upload(path)
        if self.delete:
            for path in shard.paths():
                path.unlink(missing_ok=True)
        log.info("uploaded %s (%d records)", shard.name, shard.n_records)

