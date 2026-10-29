from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from hohere.librilight.files import pair_entries
from hohere.librilight.index import (
    check_chain,
    index_headers,
    part_headers,
    read_index,
    read_meta,
    run_index,
    write_index,
)
from hohere.librilight.tarindex import next_offset, walk

SPEAKER = "0007"
OTHER = "0008"
RATE = 24000


def streaminfo(total: int, rate: int = RATE) -> bytes:
    packed = (rate << 44) | (15 << 36) | total
    body = b"\x10\x00\x10\x00" + b"\x00\x00\x00" * 2 + packed.to_bytes(8, "big") + b"\x00" * 16
    return b"fLaC" + b"\x80" + len(body).to_bytes(3, "big") + body


def utterance(duration: float = 20.0, vad: list[list[float]] | None = None) -> tuple[bytes, bytes]:
    """A FLAC big enough to have a STREAMINFO and a JSON with a usable VAD."""
    flac = streaminfo(int(duration * RATE)) + b"\x00" * 4096
    meta = {"snr": 12.5, "voice_activity": vad or [[1.0, 4.0], [4.5, 9.0], [12.0, 14.0]]}
    return flac, json.dumps(meta).encode()


def build(entries: list[tuple[str, bytes]], fmt: int = tarfile.USTAR_FORMAT) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=fmt) as tf:
        for path, payload in entries:
            info = tarfile.TarInfo(path)
            info.mtime = 0
            if path.endswith("/"):
                info.type = tarfile.DIRTYPE
                info.size = 0
            else:
                info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def layout(speakers: tuple[str, ...] = (SPEAKER, OTHER)) -> list[tuple[str, bytes]]:
    entries: list[tuple[str, bytes]] = []
    for speaker in speakers:
        directory = f"medium/{speaker}/book-01"
        entries.append((f"{directory}/", b""))
        for stem in ("a", "b"):
            flac, meta = utterance()
            entries.append((f"{directory}/{stem}.flac", flac))
            entries.append((f"{directory}/{stem}.json", meta))
    return entries


class Archive:
    """A `RangeReader` stand-in over an in-memory tar."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.reads = 0

    def read(self, offset: int, n: int) -> bytes:
        self.reads += 1
        return self.data[offset : offset + n]

    def size(self) -> int:
        return len(self.data)


class Zeroed(Archive):
    """An archive whose block at `offset` decodes as zeros (a damaged region)."""

    def __init__(self, data: bytes, offset: int) -> None:
        super().__init__(data)
        self.bad = slice(offset, offset + 512)

    def read(self, offset: int, n: int) -> bytes:
        buffer = bytearray(super().read(offset, n))
        start = max(self.bad.start - offset, 0)
        stop = min(self.bad.stop - offset, len(buffer))
        if start < stop:
            buffer[start:stop] = bytes(stop - start)
        return bytes(buffer)


def test_walks_the_whole_archive_in_any_number_of_parts() -> None:
    data = build(layout())
    whole = [h.path for h in walk(Archive(data).read, 0, len(data))]
    for parts in (1, 2, 3, 7, 64):
        assert [h.path for h in index_headers(Archive(data).read, len(data), parts)] == whole
    assert whole == [
        "medium/0007/book-01/a.flac",
        "medium/0007/book-01/a.json",
        "medium/0007/book-01/b.flac",
        "medium/0007/book-01/b.json",
        "medium/0008/book-01/a.flac",
        "medium/0008/book-01/a.json",
        "medium/0008/book-01/b.flac",
        "medium/0008/book-01/b.json",
    ]


def test_part_headers_starts_at_the_first_header_after_a_payload() -> None:
    data = build(layout())
    headers = list(walk(Archive(data).read, 0, len(data)))
    middle = headers[2].data_offset + 100  # inside the payload of b.flac
    rest = [h.path for h in part_headers(Archive(data).read, middle, len(data))]
    assert rest[0] == "medium/0007/book-01/b.json"
    assert len(rest) == len(headers) - 3


def test_check_chain_ignores_directory_entries() -> None:
    data = build(layout())
    headers = list(walk(Archive(data).read, 0, len(data)))
    assert check_chain(Archive(data).read, headers) == []


def test_check_chain_reports_a_missing_entry_as_a_hole() -> None:
    data = build(layout())
    headers = list(walk(Archive(data).read, 0, len(data)))
    problems = check_chain(Archive(data).read, headers[:2] + headers[3:])
    assert len(problems) == 1 and "hole" in problems[0]


def test_check_chain_reports_an_entry_that_overruns_the_next_one() -> None:
    data = build(layout())
    headers = list(walk(Archive(data).read, 0, len(data)))
    problems = check_chain(Archive(data).read, [headers[0], headers[3]])
    assert len(problems) == 1 and "hole" in problems[0]


def test_index_file_round_trips(tmp_path: Path) -> None:
    data = build(layout())
    headers = list(walk(Archive(data).read, 0, len(data)))
    path = tmp_path / "tar_index.jsonl"
    assert write_index(path, headers) == len(headers)
    assert read_index(path) == headers


def test_read_meta_reads_json_and_streaminfo_and_caches_them(tmp_path: Path) -> None:
    data = build(layout())
    files, _ = pair_entries(walk(Archive(data).read, 0, len(data)))
    archive = Archive(data)
    meta = read_meta(archive.read, files[0], tmp_path)
    assert meta["duration"] == pytest.approx(20.0) and meta["snr"] == 12.5
    assert meta["voice_activity"][0] == [1.0, 4.0]
    reads = archive.reads
    assert read_meta(archive.read, files[0], tmp_path) == meta
    assert archive.reads == reads  # the second call came from WORK/meta


def test_run_index_pairs_excludes_and_plans(tmp_path: Path) -> None:
    data = build(layout())
    archive = Archive(data)
    summary = run_index(
        tmp_path,
        parts=4,
        workers=2,
        exclude_speakers=[OTHER],
        reader=archive,
    )
    assert summary["files_seen"] == 4
    assert summary["paired"] == 4
    assert summary["excluded"] == {"speaker": 2, "recording": 0}
    assert summary["kept"] == 2
    assert summary["speakers"] == 1
    assert summary["segments"] == 2
    lines = [json.loads(line) for line in (tmp_path / "plan.jsonl").read_text().splitlines()]
    assert [line["chapter"] for line in lines] == ["book_01_a", "book_01_b"]
    assert [line["segments"] for line in lines] == [[[0.5, 14.5]], [[0.5, 14.5]]]
    assert [line["flac_offset"] for line in lines] == [1024, 7168]
    assert summary["hours"] == pytest.approx(round(2 * 14.0 / 3600, 4))
    assert (tmp_path / "index_summary.json").exists()


def test_run_index_resumes_from_an_existing_index(tmp_path: Path) -> None:
    data = build(layout())
    archive = Archive(data)
    run_index(tmp_path, parts=4, workers=1, reader=archive)
    reads = archive.reads
    run_index(tmp_path, parts=4, workers=1, reader=Archive(b""))  # no archive at all
    assert reads > 0
    assert (tmp_path / "tar_index.jsonl").exists()
    assert (tmp_path / "plan.jsonl").exists()


def test_run_index_fails_on_a_damaged_region(tmp_path: Path) -> None:
    data = build(layout())
    headers = list(walk(Archive(data).read, 0, len(data)))
    damaged = Zeroed(data, headers[5].header_offset)  # a header block reads as zeros
    with pytest.raises(RuntimeError, match="does not form one contiguous chain"):
        run_index(tmp_path, parts=4, workers=1, reader=damaged)


def test_a_flac_payload_holding_ustar_bytes_is_not_a_header() -> None:
    payload = b"fLaC" + b"ustar\x0000" + b"\x01" * 600
    data = build([(f"medium/{SPEAKER}/book-01/a.flac", payload)])
    headers = index_headers(Archive(data).read, len(data), 3)
    assert [h.path for h in headers] == [f"medium/{SPEAKER}/book-01/a.flac"]
    assert next_offset(headers[0]) == headers[0].data_offset + 1024
