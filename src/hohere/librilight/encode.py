"""`hohere librilight encode`: planned segments to Mimi token files, one process per GPU.

Downloads, ffmpeg and Mimi form a pipeline of bounded queues: at most `ahead` FLAC
files are in memory, segments are batched across files by length (at most
`batch_items` items and `max_padded_frames` padded frames, so a batch of 15 s items
fits the 5.6 GB of a 4090), and each file is written on a writer thread as soon as all
of its segments are encoded, so the GPU never waits for a `torch.save`. A
`done/<chapter>.json` marker makes the run resumable at file granularity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import Lock

import numpy as np

from hohere.librilight.activity import (
    FRAME_SAMPLES,
    SAMPLE_RATE,
    SILENCE_CODES,
    activity,
    has_pause,
    is_silence,
    num_frames,
)
from hohere.librilight.ids import segment_id
from hohere.librilight.index import DEFAULT_URL, TAR_MD5
from hohere.librilight.plan import PlanFile, assign_plan_parts, read_plan
from hohere.librilight.remote import RangeReader
from hohere.storage.manifest import sha256_file

log = logging.getLogger(__name__)

NUM_CODEBOOKS = 8
FFMPEG = (
    "ffmpeg",
    "-nostdin",
    "-loglevel",
    "error",
    "-i",
    "pipe:0",
    "-ac",
    "1",
    "-ar",
    str(SAMPLE_RATE),
    "-f",
    "f32le",
    "pipe:1",
)
DATASET = "libri-light"
DATASET_CONFIG = "medium"
DATASET_SPLIT = "librilight.medium"
SKILL = "single_speech"
PROGRESS_EVERY = 30.0
WRITE_WORKERS = 8


@dataclass(frozen=True)
class _Item:
    file: int  # index into the part's file list
    index: int  # index of the planned segment in that file
    audio: np.ndarray
    act: np.ndarray


def cut(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    """Samples of `[start, end)` seconds, clamped to the file."""
    first = max(0, round(start * SAMPLE_RATE))
    last = min(audio.shape[0], round(end * SAMPLE_RATE))
    return audio[first:last].copy() if last > first else audio[:0].copy()


def numbering(seg_indices: Iterable[int]) -> list[tuple[int, int]]:
    """`(number, segment index)` of the kept segments, in time order.

    Dropped segments never reach this point, so the numbering has no holes.
    """
    return list(enumerate(sorted(seg_indices)))


def channel(segment: str) -> str:
    """Speech of a segment goes to channel A or B, half and half, by its id."""
    return "A" if hashlib.sha256(segment.encode()).digest()[0] % 2 == 0 else "B"


def tokens(codes: np.ndarray, act: np.ndarray, side: str) -> dict:
    """The `.pt` payload: codes and activity on `side`, silence and zeros on the other."""
    silence = np.tile(np.asarray(SILENCE_CODES, np.int16)[:, None], (1, codes.shape[1]))
    zeros = np.zeros(codes.shape[1], np.uint8)
    return {
        "user": codes if side == "A" else silence,
        "target": codes if side == "B" else silence,
        "user_activity": act if side == "A" else zeros,
        "activity": act if side == "B" else zeros,
    }


def source_name(side: str) -> str:
    """`source` of a manifest line: which channel of the `.pt` carries the speech."""
    return f"1a_on_{side}"


def manifest_line(
    *,
    id: str,
    speaker: str,
    frames: int,
    sha256: str,
    codec_sha256: str,
    start: float,
    end: float,
) -> dict:
    """One manifest line; keys are fixed by the dataset schema.

    `source` names the layout of the token file rather than the archive it came from:
    the speech sits on the A or the B channel, which `channel` picks from the id, so
    the line cannot name a layout the `.pt` does not have.
    """
    return {
        "id": id,
        "source": source_name(channel(id)),
        "skill": SKILL,
        "family": f"librilight/{speaker}",
        "user_voice": speaker,
        "frames": frames,
        "tokens_path": f"tokens/{id}.pt",
        "sha256": sha256,
        "dataset": DATASET,
        "dataset_revision": f"md5:{TAR_MD5}",
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "codec_sha256": codec_sha256,
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
    }


def pack_batches(
    lengths: Sequence[int], batch_items: int, max_padded_frames: int
) -> list[list[int]]:
    """Indices into `lengths`, shortest first, packed into batches of padded frames."""
    if batch_items < 1:
        raise ValueError("batch_items must be >= 1")
    batches: list[list[int]] = []
    current: list[int] = []
    widest = 0
    for i in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        frames = num_frames(lengths[i])
        wider = max(widest, frames)
        full = len(current) >= batch_items or wider * (len(current) + 1) > max_padded_frames
        if current and full:
            batches.append(current)
            current, wider = [], frames
        current.append(i)
        widest = wider
    if current:
        batches.append(current)
    return batches


def encode_batch(mimi, audios: Sequence[np.ndarray], device: str) -> list[np.ndarray]:
    """Codes `[8, frames]` of every item, cut to its own frame count."""
    import torch

    frames = [num_frames(len(a)) for a in audios]
    samples = max(frames) * FRAME_SAMPLES
    x = np.zeros((len(audios), 1, samples), np.float32)
    for i, audio in enumerate(audios):
        x[i, 0, : audio.shape[0]] = audio
    with torch.no_grad():
        codes = mimi.encode(torch.from_numpy(x).to(device))
    out = codes.cpu().numpy().astype(np.int16)
    return [out[i, :, : frames[i]] for i in range(len(audios))]


def resolve_checkpoint(checkpoint: str) -> Path:
    """A local file, or `<repo>/<file>` fetched from the Hub into the HF cache."""
    path = Path(checkpoint)
    if path.exists():
        return path
    repo, _, name = checkpoint.rpartition("/")
    if not repo or not name:
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    from huggingface_hub import hf_hub_download  # lazy: only the encoding extra has it

    return Path(hf_hub_download(repo, name))


def load_mimi(checkpoint: Path, device: str):
    """Mimi with 8 codebooks; TF32 off so batched and single-item codes are identical."""
    import torch
    from moshi.models import loaders  # lazy: only the encoding machine has moshi

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    mimi = loaders.get_mimi(str(checkpoint), device=device)
    mimi.set_num_codebooks(NUM_CODEBOOKS)
    return mimi


def check_silence(mimi, device: str) -> None:
    """Refuse to run if Mimi no longer maps digital silence to `SILENCE_CODES`."""
    codes = encode_batch(mimi, [np.zeros(FRAME_SAMPLES, np.float32)], device)[0]
    if not is_silence(codes):
        raise RuntimeError(
            f"Mimi encodes silence to {codes[:, 0].tolist()}, expected {list(SILENCE_CODES)}"
        )


def decode_flac(data: bytes) -> np.ndarray:
    """Mono float32 at 24 kHz, decoded by ffmpeg (runs in a worker process)."""
    proc = subprocess.run(FFMPEG, input=data, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip()[-500:]
        raise RuntimeError(f"ffmpeg failed with {proc.returncode}: {tail}")
    audio = np.frombuffer(proc.stdout, np.float32)
    if audio.shape[0] == 0:
        raise RuntimeError("ffmpeg produced no samples")
    return audio.copy()


def save_tokens(path: Path, payload: dict) -> str:
    """`torch.save` atomically (tmp + rename); returns the sha256 of the file.

    The numpy arrays of `tokens` become tensors here, so the file is a plain
    `{name: tensor}` dict that `torch.load(weights_only=True)` can read back. There is
    no fsync: the `done/` marker is the durable record, and a `.pt` that does not
    survive a crash is encoded again on the next run.
    """
    import torch

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value)) for name, value in payload.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        torch.save(tensors, f)
    tmp.replace(path)
    return sha256_file(path)


def write_file(
    file: PlanFile,
    payloads: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    tokens_dir: Path,
    done_dir: Path,
    codec_sha256: str,
    save: Callable[[Path, dict], str] = save_tokens,
) -> list[dict]:
    """Save every token file of one source file, then its `done/<chapter>.json` marker.

    The marker is the resume point and is written last and fsynced, so a chapter is
    either complete on disk or redone. Runs on a writer thread.
    """
    lines = []
    for number, seg_index in numbering(payloads):
        codes, act = payloads[seg_index]
        start, end = file.segments[seg_index]
        sid = segment_id(file.speaker, file.chapter, number)
        digest = save(tokens_dir / f"{sid}.pt", tokens(codes, act, channel(sid)))
        lines.append(
            manifest_line(
                id=sid,
                speaker=file.speaker,
                frames=int(codes.shape[1]),
                sha256=digest,
                codec_sha256=codec_sha256,
                start=start,
                end=end,
            )
        )
    marker = done_dir / f"{file.chapter}.json"
    tmp = marker.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(lines))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(marker)
    return lines


class Writers:
    """Token writing on `workers` threads, with at most `pending` files in the queue.

    Mimi encodes ~2600x real time, so a few hundred kilobytes per segment cannot be
    written on the thread that feeds the GPU: `submit` hands the job over and blocks
    only once `pending` writes are in flight, which bounds memory. A failed write is
    re-raised by `submit`, `wait` or `close`, so a full disk stops the run instead of
    leaving holes in the release.
    """

    def __init__(self, workers: int = WRITE_WORKERS, pending: int | None = None) -> None:
        self.workers = max(1, workers)
        self.limit = max(1, pending if pending is not None else 2 * self.workers)
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="write")
        self.pending: set[Future] = set()
        self.lock = Lock()
        self.done = 0

    def __enter__(self) -> Writers:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc is None:
            self.close()
        else:  # a failing run cancels what has not started and keeps its first error
            self.pool.shutdown(wait=True, cancel_futures=True)

    def submit(self, job: Callable[[], object]) -> None:
        """Queue `job`; a finished failure stops the run, a full queue waits for a slot."""
        finished = [future for future in self.pending if future.done()]
        if finished:
            self._harvest(finished)
        if len(self.pending) >= self.limit:
            self._harvest(wait_futures(self.pending, return_when=FIRST_COMPLETED).done)
        self.pending.add(self.pool.submit(self._run, job))

    def wait(self) -> None:
        """Block until every queued write is on disk; re-raise the first failure."""
        while self.pending:
            self._harvest(wait_futures(self.pending, return_when=FIRST_COMPLETED).done)

    def close(self) -> None:
        self.wait()
        self.pool.shutdown(wait=True)

    def _run(self, job: Callable[[], object]) -> None:
        job()
        with self.lock:
            self.done += 1

    def _harvest(self, finished: Iterable[Future]) -> None:
        error: BaseException | None = None
        for future in finished:
            self.pending.discard(future)
            if error is None:
                error = future.exception()
        if error is not None:
            raise error


def _decoded(
    files: Sequence[PlanFile],
    reader: RangeReader,
    download_workers: int,
    decode_workers: int,
    ahead: int = 8,
) -> Iterator[tuple[int, PlanFile, np.ndarray]]:
    """Yield `(index, file, audio)` keeping at most `ahead` files in flight."""
    stream = iter(enumerate(files))
    downloads: dict[Future, tuple[int, PlanFile]] = {}
    decodes: dict[Future, tuple[int, PlanFile]] = {}

    def start_downloads(pool: ThreadPoolExecutor) -> None:
        while len(downloads) < ahead:
            pair = next(stream, None)
            if pair is None:
                return
            index, file = pair
            downloads[pool.submit(reader.read, file.flac_offset, file.flac_size)] = (index, file)

    with (
        ThreadPoolExecutor(max_workers=download_workers) as download_pool,
        ProcessPoolExecutor(max_workers=decode_workers) as decode_pool,
    ):
        start_downloads(download_pool)
        while downloads or decodes:
            while downloads and len(decodes) < ahead:
                done = wait_futures(downloads, return_when=FIRST_COMPLETED).done
                future = min(done, key=lambda f: downloads[f][0])
                index, file = downloads.pop(future)
                decodes[decode_pool.submit(decode_flac, future.result())] = (index, file)
                start_downloads(download_pool)
            if not decodes:
                continue
            done = wait_futures(decodes, return_when=FIRST_COMPLETED).done
            future = min(done, key=lambda f: decodes[f][0])
            index, file = decodes.pop(future)
            yield index, file, future.result()


class _Progress:
    """One JSON line every `every` seconds on stdout."""

    def __init__(
        self,
        total: int,
        done: int = 0,
        *,
        every: float = PROGRESS_EVERY,
        clock=time.monotonic,
        stream=None,
    ) -> None:
        self.total = total
        self.done = done
        self.every = every
        self.clock = clock
        self.stream = stream or sys.stdout
        self.started = clock()
        self.last = self.started

    def tick(self, segments: int, seconds: float, dropped: Counter, force: bool = False) -> None:
        now = self.clock()
        if not force and now - self.last < self.every:
            return
        wall = max(now - self.started, 1e-9) / 3600
        self.last = now
        line = {
            "files_done": self.done,
            "files_total": self.total,
            "segments": segments,
            "hours": round(seconds / 3600, 4),
            "audio_hours_per_wall_hour": round(seconds / 3600 / wall, 2),
            "dropped": dict(dropped),
        }
        self.stream.write(json.dumps(line) + "\n")
        self.stream.flush()


def run_encode(
    *,
    plan: Path,
    part: int,
    parts: int,
    out: Path,
    checkpoint: str,
    batch_items: int = 16,
    max_padded_frames: int = 3000,
    download_workers: int = 4,
    decode_workers: int = 12,
    write_workers: int = WRITE_WORKERS,
    url: str = DEFAULT_URL,
    device: str = "cuda",
) -> dict:
    """Encode part `part` of `plan` into `out/tokens` and `out/done`."""
    if not 0 <= part < parts:
        raise ValueError(f"part must be in [0, {parts}), got {part}")
    out = Path(out)
    tokens_dir = out / "tokens"
    done_dir = out / "done"
    tokens_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    files = assign_plan_parts(read_plan(plan), parts)[part]
    todo = [f for f in files if not (done_dir / f"{f.chapter}.json").exists()]
    skipped = len(files) - len(todo)
    weights = resolve_checkpoint(checkpoint)
    codec_sha = sha256_file(weights)
    mimi = load_mimi(weights, device)
    check_silence(mimi, device)

    progress = _Progress(len(files), skipped)
    dropped: Counter[str] = Counter()
    held: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    expected: dict[int, int] = {}
    planned: dict[int, PlanFile] = {}
    seconds = 0.0
    window = max(4 * batch_items, batch_items)
    items: list[_Item] = []
    encoded = 0

    def write(file: PlanFile, payloads: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
        """Hand one file to the writer threads: its `.pt` files, then its marker."""
        writers.submit(
            partial(
                write_file,
                file,
                payloads,
                tokens_dir=tokens_dir,
                done_dir=done_dir,
                codec_sha256=codec_sha,
            )
        )

    def drain() -> None:
        for index in [i for i, left in expected.items() if left == 0]:
            payloads = held.pop(index)
            expected.pop(index)
            write(planned.pop(index), payloads)

    def flush() -> None:
        nonlocal items, encoded
        if not items:
            return
        lengths = [item.audio.shape[0] for item in items]
        for batch in pack_batches(lengths, batch_items, max_padded_frames):
            chosen = [items[k] for k in batch]
            codes = encode_batch(mimi, [c.audio for c in chosen], device)
            for item, code in zip(chosen, codes, strict=True):
                held[item.file][item.index] = (code, item.act)
                expected[item.file] -= 1
                encoded += 1
        items = []
        drain()

    reader = RangeReader(url)
    with Writers(write_workers) as writers:
        for index, file, audio in _decoded(todo, reader, download_workers, decode_workers):
            planned[index] = file
            kept = 0
            for seg_index, (start, end) in enumerate(file.segments):
                clip = cut(audio, start, end)
                act = activity(clip)
                if not act.any():
                    dropped["silent"] += 1
                    continue
                if not has_pause(act):
                    dropped["no_pause"] += 1
                    continue
                items.append(_Item(index, seg_index, clip, act))
                kept += 1
                seconds += end - start
            expected[index] = kept
            if kept == 0:
                expected.pop(index)
                planned.pop(index)
                write(file, {})
            if len(items) >= window:
                flush()
            progress.done = skipped + writers.done
            progress.tick(encoded, seconds, dropped)
        flush()
        writers.wait()  # every token file and marker is on disk before the next stage
    progress.done = skipped + writers.done
    progress.tick(encoded, seconds, dropped, force=True)
    log.info("part %d/%d: %d files, %d segments", part, parts, progress.done, encoded)
    return {
        "part": part,
        "parts": parts,
        "files": len(files),
        "skipped": skipped,
        "done": progress.done,
        "segments": encoded,
        "hours": round(seconds / 3600, 4),
        "dropped": dict(dropped),
        "out": str(out),
    }
