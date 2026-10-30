"""SQLite resume state.

Three tables:

* `files`   — the archive index (one row per utterance) plus `order_rank`, the shard
  that holds its last chunk and the `todo|done` status,
* `shards`  — `open|closed|uploaded`, bytes, digests, record and hour counters,
* `meta`    — small key/value pairs (run start time, counters).

A file only becomes `done` when the shard holding its last chunk is uploaded (or
closed, with `--no-upload`). On start, every `open` shard is garbage: its local files
are deleted and its files go back to `todo`.

WAL mode, one connection behind a re-entrant lock: many reader threads, one writer.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from . import order
from .librilight import SourceFile
from .shards import ClosedShard

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_id      TEXT PRIMARY KEY,
    split        TEXT NOT NULL,
    speaker      TEXT NOT NULL,
    book         TEXT NOT NULL,
    archive_url  TEXT NOT NULL,
    flac_offset  INTEGER NOT NULL,
    flac_size    INTEGER NOT NULL,
    json_offset  INTEGER NOT NULL,
    json_size    INTEGER NOT NULL,
    duration_s   REAL NOT NULL,
    vad          TEXT NOT NULL,
    order_rank   INTEGER,
    status       TEXT NOT NULL DEFAULT 'todo',
    shard        TEXT
);
CREATE INDEX IF NOT EXISTS files_split ON files(split);
CREATE INDEX IF NOT EXISTS files_rank ON files(order_rank);
CREATE INDEX IF NOT EXISTS files_status ON files(status);
CREATE TABLE IF NOT EXISTS shards (
    name         TEXT PRIMARY KEY,
    status       TEXT NOT NULL,
    bytes        INTEGER NOT NULL DEFAULT 0,
    sha256       TEXT,
    jsonl_sha256 TEXT,
    n_records    INTEGER NOT NULL DEFAULT 0,
    hours        REAL NOT NULL DEFAULT 0,
    created_at   REAL,
    updated_at   REAL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

SHARD_STATUSES = ("open", "closed", "uploaded")
FILE_STATUSES = ("todo", "done")
_COLUMNS = (
    "file_id",
    "split",
    "speaker",
    "book",
    "archive_url",
    "flac_offset",
    "flac_size",
    "json_offset",
    "json_size",
    "duration_s",
    "vad",
)


def _row_to_file(row: Sequence[Any]) -> SourceFile:
    data = dict(zip(_COLUMNS, row[: len(_COLUMNS)], strict=True))
    vad = [(float(a), float(b)) for a, b in json.loads(data["vad"])]
    return SourceFile(vad=vad, **{k: v for k, v in data.items() if k != "vad"})


class State:
    """The resume state; safe to use from several threads (writes are serialised)."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        out_dir: str | os.PathLike[str] | None = None,
    ):
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.executescript(SCHEMA)
            self._db.commit()

    # -- plumbing ----------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> State:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _write(self, sql: str, params: Sequence[Any] | Iterable[Sequence[Any]] = ()) -> None:
        with self._lock:
            self._db.execute(sql, params)
            self._db.commit()

    def _write_many(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self._lock:
            self._db.executemany(sql, rows)
            self._db.commit()

    def _all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(sql, params))

    def _one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._db.execute(sql, params).fetchone()

    # -- files -------------------------------------------------------------------
    def add_files(self, files: Iterable[SourceFile]) -> int:
        rows = [
            (
                f.file_id,
                f.split,
                f.speaker,
                f.book,
                f.archive_url,
                f.flac_offset,
                f.flac_size,
                f.json_offset,
                f.json_size,
                f.duration_s,
                json.dumps([[a, b] for a, b in f.vad]),
            )
            for f in files
        ]
        if not rows:
            return 0
        with self._lock:
            before = self._db.total_changes
            self._db.executemany(
                "INSERT OR IGNORE INTO files ("
                + ", ".join(_COLUMNS)
                + ") VALUES ("
                + ", ".join("?" * len(_COLUMNS))
                + ")",
                rows,
            )
            self._db.commit()
            return self._db.total_changes - before

    def files(self, split: str | None = None, status: str | None = None) -> list[SourceFile]:
        sql = f"SELECT {', '.join(_COLUMNS)} FROM files"  # fixed column list
        clauses, params = [], []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY order_rank IS NULL, order_rank, file_id"
        return [_row_to_file(tuple(row)) for row in self._all(sql, params)]

    def has_files(self, split: str | None = None) -> bool:
        row = self._one(
            "SELECT COUNT(*) AS n FROM files" + (" WHERE split = ?" if split else ""),
            (split,) if split else (),
        )
        return bool(row and row["n"])

    def count_files(self, split: str | None = None, status: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM files"
        clauses, params = [], []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        row = self._one(sql, params)
        return int(row["n"]) if row else 0

    def source_hours(self, split: str | None = None, status: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(duration_s), 0) AS s FROM files"
        clauses, params = [], []
        if split is not None:
            clauses.append("split = ?")
            params.append(split)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        row = self._one(sql, params)
        return float(row["s"]) / 3600.0 if row else 0.0

    def set_status(self, file_id: str, status: str) -> None:
        if status not in FILE_STATUSES:
            raise ValueError(f"unknown file status {status!r}")
        self._write("UPDATE files SET status = ? WHERE file_id = ?", (status, file_id))

    def set_shard(self, file_id: str, shard: str | None, *, status: str = "todo") -> None:
        if status not in FILE_STATUSES:
            raise ValueError(f"unknown file status {status!r}")
        self._write(
            "UPDATE files SET shard = ?, status = ? WHERE file_id = ?", (shard, status, file_id)
        )

    def shard_of(self, file_id: str) -> str | None:
        row = self._one("SELECT shard FROM files WHERE file_id = ?", (file_id,))
        return row["shard"] if row else None

    # -- order -------------------------------------------------------------------
    def assign_order(self) -> int:
        """Rank every indexed file by `sha256(file_id)` (idempotent)."""
        ids = [row["file_id"] for row in self._all("SELECT file_id FROM files")]
        ranks = order.rank(ids)
        self._write_many(
            "UPDATE files SET order_rank = ? WHERE file_id = ?",
            [(rank, fid) for fid, rank in ranks.items()],
        )
        return len(ranks)

    def ordered_files(self, status: str | None = None) -> list[SourceFile]:
        """Indexed files in the fixed order (assigning ranks first if needed)."""
        if self._one("SELECT COUNT(*) AS n FROM files WHERE order_rank IS NULL")["n"]:
            self.assign_order()
        return self.files(status=status)

    # -- shards ------------------------------------------------------------------
    def open_shard(self, name: str) -> None:
        self._write(
            "INSERT OR REPLACE INTO shards (name, status, created_at, updated_at) "
            "VALUES (?, 'open', ?, ?)",
            (name, _now(), _now()),
        )

    def close_shard(self, shard: ClosedShard) -> None:
        self._write(
            "INSERT OR REPLACE INTO shards "
            "(name, status, bytes, sha256, jsonl_sha256, n_records, hours, created_at, updated_at) "
            "VALUES (?, 'closed', ?, ?, ?, ?, ?, ?, ?)",
            (
                shard.name,
                shard.bytes,
                shard.sha256,
                shard.jsonl_sha256,
                shard.n_records,
                shard.hours,
                _now(),
                _now(),
            ),
        )
        self._mark_files_done(shard.name)

    def mark_uploaded(self, name: str) -> None:
        self._write(
            "UPDATE shards SET status = 'uploaded', updated_at = ? WHERE name = ?",
            (_now(), name),
        )
        self._mark_files_done(name)

    def _mark_files_done(self, name: str) -> None:
        self._write("UPDATE files SET status = 'done' WHERE shard = ?", (name,))

    def shards(self, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM shards"
        params: Sequence[Any] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status,)
        sql += " ORDER BY name"
        return [dict(row) for row in self._all(sql, params)]

    def last_shards(self, count: int = 10) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM shards ORDER BY name DESC LIMIT ?", (count,))
        return [dict(row) for row in reversed(rows)]

    def next_shard_index(self) -> int:
        highest = -1
        for shard in self.shards():
            try:
                highest = max(highest, int(shard["name"].rsplit("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        return highest + 1

    # -- resume ------------------------------------------------------------------
    def recover(self, out_dir: str | os.PathLike[str] | None = None) -> list[str]:
        """Drop every `open` shard: its files are garbage, its files go back to TODO."""
        directory = Path(out_dir) if out_dir is not None else self.out_dir
        dropped: list[str] = []
        for shard in self.shards(status="open"):
            name = shard["name"]
            if directory is not None:
                for suffix in (".tar", ".jsonl"):
                    (directory / f"{name}{suffix}").unlink(missing_ok=True)
            self._write(
                "UPDATE files SET status = 'todo', shard = NULL WHERE shard = ?", (name,)
            )
            self._write("DELETE FROM shards WHERE name = ?", (name,))
            dropped.append(name)
        return dropped

    # -- meta --------------------------------------------------------------------
    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._one("SELECT value FROM meta WHERE key = ?", (key,))
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:  # pragma: no cover - meta is always JSON
            return row["value"]

    def set_meta(self, key: str, value: Any) -> None:
        self._write(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, json.dumps(value))
        )

    def bump_meta(self, key: str, amount: float) -> float:
        total = float(self.get_meta(key, 0) or 0) + amount
        self.set_meta(key, total)
        return total

    # -- reporting ---------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Everything the status page reports that lives in the DB."""
        files_total = self.count_files()
        files_done = self.count_files(status="done")
        shard_rows = self.shards()
        closed = [s for s in shard_rows if s["status"] in ("closed", "uploaded")]
        uploaded = [s for s in shard_rows if s["status"] == "uploaded"]
        hours = sum(float(s["hours"] or 0.0) for s in closed)
        chunks = sum(int(s["n_records"] or 0) for s in closed)
        source_hours = self.source_hours()
        done_hours = self.source_hours(status="done")
        return {
            "files_total": files_total,
            "files_done": files_done,
            "source_hours": source_hours,
            "source_hours_done": done_hours,
            "remaining_files": max(0, files_total - files_done),
            "remaining_hours": max(0.0, source_hours - done_hours),
            "chunks": chunks,
            "hours": hours,
            "shards_total": len(shard_rows),
            "shards_uploaded": len(uploaded),
            "shards_closed": len(closed),
            "bytes_uploaded": sum(int(s["bytes"] or 0) for s in uploaded),
            "started_at": self.get_meta("started_at"),
            "target_hours": self.get_meta("target_hours", "full"),
        }


def _now() -> float:
    return time.time()
