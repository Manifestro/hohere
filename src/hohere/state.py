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
    shard        TEXT,
    n_chunks     INTEGER,
    chunk_hours  REAL,
    cum_hours    REAL
);
CREATE INDEX IF NOT EXISTS files_split ON files(split);
CREATE INDEX IF NOT EXISTS files_rank ON files(order_rank);
CREATE INDEX IF NOT EXISTS files_status ON files(status);
CREATE INDEX IF NOT EXISTS files_cum ON files(cum_hours);
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

class StateError(RuntimeError):
    """The state DB is not in the shape the requested operation needs."""


SHARD_STATUSES = ("open", "closed", "uploaded")
FILE_STATUSES = ("todo", "done", "error")
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

    def file_status(self, file_id: str) -> str | None:
        row = self._one("SELECT status FROM files WHERE file_id = ?", (file_id,))
        return row["status"] if row else None

    def set_shard(self, file_id: str, shard: str | None, *, status: str = "todo") -> None:
        if status not in FILE_STATUSES:
            raise ValueError(f"unknown file status {status!r}")
        self._write(
            "UPDATE files SET shard = ?, status = ? WHERE file_id = ?", (shard, status, file_id)
        )

    def rank_of(self, file_id: str) -> int | None:
        row = self._one("SELECT order_rank FROM files WHERE file_id = ?", (file_id,))
        return int(row["order_rank"]) if row and row["order_rank"] is not None else None

    def checkpoint(self) -> None:
        """Fold the WAL into the main file, which can then be copied alone."""
        self._write("PRAGMA wal_checkpoint(TRUNCATE)")

    def status_of(self, file_id: str) -> str | None:
        row = self._one("SELECT status FROM files WHERE file_id = ?", (file_id,))
        return row["status"] if row else None

    def n_chunks_of(self, file_id: str) -> int:
        row = self._one("SELECT n_chunks FROM files WHERE file_id = ?", (file_id,))
        return int(row["n_chunks"] or 0) if row else 0

    def cum_hours_of(self, file_id: str) -> float | None:
        row = self._one("SELECT cum_hours FROM files WHERE file_id = ?", (file_id,))
        return float(row["cum_hours"]) if row and row["cum_hours"] is not None else None

    def shard_of(self, file_id: str) -> str | None:
        row = self._one("SELECT shard FROM files WHERE file_id = ?", (file_id,))
        return row["shard"] if row else None

    # -- plan --------------------------------------------------------------------
    def assign_order(self, splits: Sequence[str] | None = None) -> int:
        """Freeze the ranks of the indexed splits, by `sha256(file_id)`.

        Ranks are final: a run only ever addresses files through them, so a new split
        appearing later would silently reshuffle the order of a finished tier.
        Re-ranking therefore only ever *extends* the order — the files already ranked
        keep their rank, new files are appended after them — and asking for a split
        that was never indexed is an error, not an empty tier.
        """
        indexed = sorted({row["split"] for row in self._all("SELECT DISTINCT split FROM files")})
        if not indexed:
            raise StateError("no files are indexed; run `hohere index` first")
        wanted = list(splits) if splits is not None else indexed
        missing = [split for split in wanted if split not in indexed]
        if missing:
            raise StateError(
                f"split(s) {missing} are not indexed (indexed: {indexed}); "
                "run `hohere index --splits " + ",".join(missing) + "` first"
            )
        planned = self.get_meta("plan_splits")
        if planned is not None and sorted(planned) != indexed:
            raise StateError(
                f"the plan covers splits {planned} but the DB now indexes {indexed}; "
                "the order is final once encoding started"
            )
        ranked = self._all(
            "SELECT file_id, order_rank FROM files WHERE order_rank IS NOT NULL "
            "ORDER BY order_rank"
        )
        known = {row["file_id"] for row in ranked}
        next_rank = (max((row["order_rank"] for row in ranked), default=-1)) + 1
        fresh = [
            row["file_id"]
            for row in self._all("SELECT file_id FROM files WHERE order_rank IS NULL")
            if row["file_id"] not in known
        ]
        for file_id, rank in order.rank(fresh).items():
            self._write(
                "UPDATE files SET order_rank = ? WHERE file_id = ?", (next_rank + rank, file_id)
            )
        self.set_meta("plan_splits", indexed)
        return len(fresh)

    def plan_chunks(self, min_s: float = 30.0, max_s: float = 60.0) -> dict[str, Any]:
        """Chunk every file from its VAD spans and store `n_chunks`/`cum_hours`.

        Chunking needs nothing but the Libri-Light VAD, so it happens at index time
        and a tier is exact before a single chunk is encoded. `cum_hours` is the
        cumulative *kept* audio over `order_rank`, so `--until-hours N` is just
        `cum_hours <= N` (the first file crossing N included).
        """
        from .chunk import plan as plan_file

        if self._one("SELECT COUNT(*) AS n FROM files WHERE order_rank IS NULL")["n"]:
            self.assign_order()
        rows = self._all(
            "SELECT file_id, vad FROM files WHERE order_rank IS NOT NULL ORDER BY order_rank"
        )
        total = 0.0
        updates: list[tuple[int, float, float, str]] = []
        counted = 0
        for row in rows:
            vad = [(float(a), float(b)) for a, b in json.loads(row["vad"])]
            duration = max((end for _start, end in vad), default=0.0)
            result = plan_file(vad, duration, min_s=min_s, max_s=max_s)
            hours = result.kept_seconds / 3600.0
            total += hours
            updates.append((len(result.chunks), hours, total, row["file_id"]))
            counted += len(result.chunks)
        self._write_many(
            "UPDATE files SET n_chunks = ?, chunk_hours = ?, cum_hours = ? WHERE file_id = ?",
            updates,
        )
        self.set_meta("plan_hours", total)
        return {"files": len(rows), "chunks": counted, "chunk_hours": total}

    def select_until(self, hours: float | str | None = None) -> list[SourceFile]:
        """The files of a tier: everything with `cum_hours <= N`, plus the first crossing.

        `hours` may be a number, `"full"`/None (everything) or a tier name
        (`100`, `"full"`). Files are returned in `order_rank` order.
        """
        limit = None if hours in (None, "full") else float(hours)
        if limit is not None and self._one(
            "SELECT COUNT(*) AS n FROM files WHERE cum_hours IS NULL"
        )["n"]:
            self.plan_chunks()
        sql = (
            f"SELECT {', '.join(_COLUMNS)} FROM files WHERE order_rank IS NOT NULL "
            f"AND (? IS NULL OR cum_hours <= ?) ORDER BY order_rank"
        )
        rows = self._all(sql, (limit, limit))
        if limit is not None:
            # the first file crossing N is part of the tier, so its chunks are kept
            passing = self._one(
                "SELECT file_id FROM files WHERE cum_hours > ? ORDER BY order_rank LIMIT 1",
                (limit,),
            )
            if passing is not None:
                extra = self._all(
                    f"SELECT {', '.join(_COLUMNS)} FROM files WHERE file_id = ?",
                    (passing["file_id"],),
                )
                rows = [*rows, *extra]
        return [_row_to_file(tuple(row)) for row in rows]

    def tier_hours(self, hours: float | str | None = None) -> dict[str, float]:
        """Planned totals of a tier, so the status page can show remaining/ETA."""
        limit = None if hours in (None, "full") else float(hours)
        if self._one("SELECT COUNT(*) AS n FROM files WHERE cum_hours IS NULL")["n"]:
            self.plan_chunks()
        row = self._one(
            "SELECT COUNT(*) AS files, COALESCE(SUM(n_chunks), 0) AS chunks, "
            "COALESCE(MAX(cum_hours), 0.0) AS hours FROM files "
            "WHERE order_rank IS NOT NULL AND (? IS NULL OR cum_hours <= ?)",
            (limit, limit),
        )
        extra = (
            self._one(
                "SELECT chunk_hours FROM files WHERE cum_hours > ? ORDER BY order_rank LIMIT 1",
                (limit,),
            )
            if limit is not None
            else None
        )
        hours_total = float(row["hours"]) + (float(extra["chunk_hours"]) if extra else 0.0)
        return {
            "files": int(row["files"]) + (1 if extra else 0),
            "chunks": int(row["chunks"]),
            "hours": hours_total,
        }

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
