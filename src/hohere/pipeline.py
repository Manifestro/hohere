"""The streaming run: download -> decode -> one model worker -> shard -> upload.

Shape of the pipeline, and why:

* every stage is a thread pool behind a *bounded* queue, so a fast downloader cannot
  fill RAM while the GPU is busy — the whole run has a fixed memory ceiling;
* exactly one thread owns the device (VAD, WavLM, units, Mimi). Model loading is slow
  and a second copy on the same GPU is a mistake, not a speedup;
* the writer and the uploader are separate threads, so a slow `upload_file` never
  stalls the models; a file only becomes `done` when its shard is uploaded;
* every stage reports its own seconds to `Progress`, and `--bench-minutes` uses those
  same numbers, so the status page and the benchmark can never disagree.

All model work is behind injected callables (`Stages`), which is what makes the whole
pipeline testable on a machine without torch (`--no-upload` writes real shards).

Crash safety: a shard is only ever `open` while it is being written; on start every
`open` shard is deleted and its files go back to `todo` (`State.recover`), so a
restart redoes exactly the files whose records were not uploaded.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import queue
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import chunk as chunk_mod
from . import kmeans, records, shards, vad
from .librilight import SourceFile
from .shards import ClosedShard, ShardWriter, Uploader
from .state import State
from .status import Progress

log = logging.getLogger(__name__)

QUEUE_FILES = 8  # files in flight: download -> decode
QUEUE_CHUNKS = 32  # chunks in flight: decode -> model
QUEUE_RECORDS = 32  # records in flight: model -> writer
QUEUE_SHARDS = 2  # closed shards waiting for upload
BUFFER_BATCHES = 4  # chunks gathered per flush, in batches


@dataclass
class Stages:
    """Everything the pipeline needs from the outside world, injected."""

    download: Callable[[SourceFile], bytes]
    decode: Callable[[bytes, SourceFile], np.ndarray]
    speech: Callable[[np.ndarray], list[tuple[int, int]]]
    encode_units: Callable[[np.ndarray, list[tuple[int, int]]], Any]
    encode_mimi: Callable[[list[np.ndarray]], list[np.ndarray]]
    assign: Callable[[np.ndarray, int], np.ndarray]
    mimi_batch: int = 16
    # several chunks of similar length at once; without it `encode_units` runs per chunk
    encode_units_batch: Callable[[list[np.ndarray], list[list]], list] | None = None
    wavlm_batch: int = 4
    # optional: fraction of boundaries that agree with a float32 reference
    agreement: Callable[[np.ndarray, list[tuple[int, int]]], float] | None = None

    def with_overrides(self, **kwargs: Any) -> Stages:
        return Stages(**{**self.__dict__, **kwargs})


def encode_many(stages: Stages, wavs: list[np.ndarray], spans: list[list]) -> list:
    """Units of many chunks, batched by similar length; results keep the input order."""
    if stages.encode_units_batch is None:
        return [stages.encode_units(w, s) for w, s in zip(wavs, spans, strict=True)]
    order = sorted(range(len(wavs)), key=lambda i: wavs[i].size)
    out: list[Any] = [None] * len(wavs)
    for start in range(0, len(order), stages.wavlm_batch):
        picked = order[start : start + stages.wavlm_batch]
        done = stages.encode_units_batch([wavs[i] for i in picked], [spans[i] for i in picked])
        for i, segments in zip(picked, done, strict=True):
            out[i] = segments
    return out


@dataclass
class FileState:
    """Bookkeeping of one file while its chunks flow through the pipeline."""

    file: SourceFile
    chunks: list[tuple[float, float]] = field(default_factory=list)
    remaining: int = 0
    index: int = 0


@dataclass
class Stats:
    """Counters a bench run needs; the runtime pipeline only feeds `Progress`."""

    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    queue_wait_seconds: float = 0.0
    files: int = 0
    chunks: int = 0
    errors: int = 0
    agreements: list[float] = field(default_factory=list)


class Runner:
    """One run of the pipeline, from the state DB to uploaded shards."""

    def __init__(
        self,
        *,
        state: State,
        stages: Stages,
        out_dir: str | os.PathLike[str] = "work/shards",
        progress: Progress | None = None,
        uploader: Uploader | None = None,
        target_bytes: int = shards.GIB,
        download_workers: int = 8,
        decode_workers: int = 4,
        vad_workers: int = 8,
        mark_done: bool = True,
        keep_files: bool = True,
        crash_after_files: int | None = None,
        audio_limit_seconds: float | None = None,
    ) -> None:
        self.state = state
        self.stages = stages
        self.out_dir = Path(out_dir)
        self.progress = progress or Progress()
        self.uploader = uploader
        self.target_bytes = target_bytes
        self.download_workers = max(1, download_workers)
        self.decode_workers = max(1, decode_workers)
        self.vad_workers = max(1, vad_workers)
        self.mark_done = mark_done
        self.keep_files = keep_files
        self.crash_after_files = crash_after_files
        self.audio_limit_seconds = audio_limit_seconds
        self.stats = Stats()
        self.stop = threading.Event()
        self.failure: BaseException | None = None
        self._files_done = 0
        self._errors: list[str] = []
        self._lock = threading.Lock()
        # how many workers of each stage actually run; set once per run
        self._n_downloaders = self.download_workers
        self._n_decoders = self.decode_workers
        self._downloaders_active = self.download_workers
        self._decoders_active = self.decode_workers
        self._vads_active = self.vad_workers
        self._models_active = 1
        self._writers_active = 1
        self._uploads_active = 1
        self._file_queue: queue.Queue = queue.Queue(maxsize=QUEUE_FILES)
        self._decode_queue: queue.Queue = queue.Queue(maxsize=QUEUE_FILES)
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=QUEUE_CHUNKS)
        self._model_queue: queue.Queue = queue.Queue(maxsize=QUEUE_CHUNKS)
        self._record_queue: queue.Queue = queue.Queue(maxsize=QUEUE_RECORDS)
        self._shard_queue: queue.Queue = queue.Queue(maxsize=QUEUE_SHARDS)
        self._threads: list[threading.Thread] = []
        self._pending: dict[str, FileState] = {}
        self._pending_lock = threading.Lock()
        self._open_files = 0
        self._fed = threading.Event()
        self._closed: list[ClosedShard] = []

    # -- feed --------------------------------------------------------------------
    def feed(self, files: Sequence[SourceFile]) -> None:
        """Queue the tier, then one sentinel per downloader (they count themselves out)."""
        for file in files:
            if self.stop.is_set():
                break
            self._file_queue.put(file)
        for _ in range(self._n_downloaders):
            self._file_queue.put(None)
        self._fed.set()

    def _join_out(self, stage: str, countdown: str, downstream: queue.Queue) -> None:
        """A worker that saw its sentinel leaves; the last one wakes the next stage."""
        with self._lock:
            setattr(self, countdown, getattr(self, countdown) - 1)
            last = getattr(self, countdown) <= 0
        if last:
            for _ in range(self._fanout(stage)):
                downstream.put(None)

    def _fanout(self, stage: str) -> int:
        return {
            "download": self._n_decoders,
            "decode": self.vad_workers,
            "vad": 1,  # exactly one model worker
            "model": 1,  # exactly one writer
            "write": 1,  # exactly one uploader
        }[stage]

    # -- stages ------------------------------------------------------------------
    def _downloader(self) -> None:
        while True:
            file = self._file_queue.get()
            if file is None:
                self._join_out("download", "_downloaders_active", self._decode_queue)
                return
            if self.stop.is_set():
                continue
            started = time.perf_counter()
            try:
                raw = self.stages.download(file)
            except Exception as exc:  # a dead file never kills the run
                self._error(file, f"download failed: {exc}")
                continue
            self.progress.add("download", time.perf_counter() - started)
            self._decode_queue.put((file, raw))

    def _decoder(self) -> None:
        while True:
            item = self._decode_queue.get()
            if item is None:
                self._join_out("decode", "_decoders_active", self._chunk_queue)
                return
            file, raw = item
            started = time.perf_counter()
            try:
                wav = self.stages.decode(raw, file)
            except Exception as exc:
                self._error(file, f"decode failed: {exc}")
                continue
            self.progress.add("decode", time.perf_counter() - started)
            plan = chunk_mod.plan(file.vad, _duration(wav))
            self._register(file, plan.chunks)
            for index, (start_s, end_s) in enumerate(plan.chunks):
                if self.stop.is_set():
                    break
                self._chunk_queue.put(("chunk", (file, index, start_s, end_s),
                                       _slice(wav, start_s, end_s)))

    def _vad_worker(self) -> None:
        while True:
            item = self._chunk_queue.get()
            if item is None:
                self._join_out("vad", "_vads_active", self._model_queue)
                return
            ctx, wav = item[1], item[2]
            started = time.perf_counter()
            try:
                spans = self.stages.speech(wav)
            except Exception as exc:
                self._error(ctx[0], f"vad failed: {exc}")
                continue
            self.progress.add("vad", time.perf_counter() - started)
            self._model_queue.put((ctx, wav, spans))

    def _model_worker(self) -> None:
        # A buffer of several batches, so that the chunks of a batch can be close in length.
        size = max(self.stages.mimi_batch, self.stages.wavlm_batch) * BUFFER_BATCHES
        buffer: list[tuple[Any, np.ndarray, list]] = []
        while True:
            item = self._model_queue.get()
            if item is None:
                if buffer:
                    self._flush(buffer)
                self._join_out("model", "_models_active", self._record_queue)
                return
            buffer.append(item)
            if len(buffer) >= size:
                self._flush(buffer)
                buffer = []

    def _flush(self, buffer: list[tuple[Any, np.ndarray, list]]) -> None:
        wavs = [item[1] for item in buffer]
        spans = [item[2] for item in buffer]
        try:
            started = time.perf_counter()
            units = encode_many(self.stages, wavs, spans)
            per = (time.perf_counter() - started) / len(buffer)
            for _ in buffer:
                self.progress.add("wavlm", per)
            started = time.perf_counter()
            codes = self._encode_mimi(wavs)
            per = (time.perf_counter() - started) / len(buffer)
            for _ in buffer:
                self.progress.add("mimi", per)
        except Exception as exc:  # one bad batch: name its files, keep the run alive
            for ctx in {id(item[0][0]): item[0] for item in buffer}.values():
                self._error(ctx[0], f"model failed: {exc}")
            return
        for (ctx, wav, chunk_spans), segments, code in zip(buffer, units, codes, strict=True):
            try:
                self._encode_chunk(ctx, wav, chunk_spans, segments, code)
            except Exception as exc:
                self._error(ctx[0], f"model failed: {exc}")

    def _encode_mimi(self, wavs: list[np.ndarray]) -> list[np.ndarray]:
        order = sorted(range(len(wavs)), key=lambda i: wavs[i].size)
        out: list[Any] = [None] * len(wavs)
        for start in range(0, len(order), self.stages.mimi_batch):
            picked = order[start : start + self.stages.mimi_batch]
            for i, code in zip(picked, self.stages.encode_mimi([wavs[i] for i in picked]),
                               strict=True):
                out[i] = code
        return out

    def _encode_chunk(
        self, ctx: Any, wav: np.ndarray, spans: list, segments: Any, codes: np.ndarray
    ) -> None:
        file, index, start_s, end_s = ctx
        started = time.perf_counter()
        ids = {k: self.stages.assign(segments.embeddings, k) for k in (4096, 8192)}
        self.progress.add("assign", time.perf_counter() - started)
        if self.stages.agreement is not None:
            self.stats.agreements.append(float(self.stages.agreement(wav, spans)))
        n_frames = int(codes.shape[-1])
        record = {
            "codes": codes,
            "unit_start": segments.unit_start,
            "unit_end": segments.unit_end,
            "unit_k4096": ids[4096].astype(np.int16),
            "unit_k8192": ids[8192].astype(np.int16),
            "sil_start": np.asarray(vad.pauses(spans), dtype=np.int32),
        }
        meta = {
            "id": records.chunk_id(file.file_id, index),
            "file_id": file.file_id,
            "split": file.split,
            "speaker": file.speaker,
            "book": file.book,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "frames": n_frames,
            "n_units": int(segments.unit_start.shape[0]),
            "n_sil": int(record["sil_start"].shape[0]),
            "order_rank": self.state.rank_of(file.file_id),
            "cum_hours": self.state.cum_hours_of(file.file_id),
        }
        self._record_queue.put((file, record, meta))

    def _writer(self) -> None:
        writer = ShardWriter(
            self.out_dir, target_bytes=self.target_bytes, index=self.state.next_shard_index()
        )
        writer.open()
        self.state.open_shard(writer.name)
        while True:
            item = self._record_queue.get()
            if item is None:
                break
            file, record, meta = item
            started = time.perf_counter()
            data = records.write_record(**record)
            meta["shard"] = writer.name
            closed = writer.add(meta["id"], data, meta)
            if closed is not None:
                self._finish_shard(closed)
                writer.open()
                self.state.open_shard(writer.name)
            self.progress.add("write", time.perf_counter() - started)
            self.progress.done_chunk(meta["duration_s"] / 3600.0)
            with self._lock:
                self.stats.chunks += 1
                self.stats.audio_seconds += meta["duration_s"]
                reached = (
                    self.audio_limit_seconds is not None
                    and self.stats.audio_seconds >= self.audio_limit_seconds
                )
            if reached:
                self.stop.set()
            self._chunk_written(file)
        closed = writer.close()
        if closed is not None:
            self._finish_shard(closed)
        self._join_out("write", "_writers_active", self._shard_queue)

    def _finish_shard(self, closed: ClosedShard) -> None:
        record = {
            "name": closed.name,
            "bytes": closed.bytes,
            "n_records": closed.n_records,
            "hours": closed.hours,
            "sha256": closed.sha256,
        }
        if self.mark_done:
            self.state.close_shard(closed)
        if self.uploader is not None:
            self._shard_queue.put(closed)
        else:
            self._after_shard(record)

    def _after_shard(self, record: dict) -> None:
        self.progress.shard_uploaded(record)
        if self.mark_done:
            self.state.mark_uploaded(record["name"])

    def _uploader_loop(self) -> None:
        while True:
            shard = self._shard_queue.get()
            if shard is None:
                return
            started = time.perf_counter()
            try:
                assert self.uploader is not None
                self.uploader.upload(shard)
            except Exception as exc:
                self._error_text(f"upload of {shard.name} failed: {exc}")
                continue
            self.progress.add("upload", time.perf_counter() - started)
            if self.mark_done:
                self.state.mark_uploaded(shard.name)
            self._after_shard(
                {
                    "name": shard.name,
                    "bytes": shard.bytes,
                    "n_records": shard.n_records,
                    "hours": shard.hours,
                }
            )

    # -- bookkeeping -------------------------------------------------------------
    def _register(self, file: SourceFile, chunks: Sequence[tuple[float, float]]) -> None:
        with self._pending_lock:
            self._pending[file.file_id] = FileState(
                file=file, chunks=list(chunks), remaining=len(chunks)
            )
            self._open_files += 1
        if not chunks:  # nothing to encode (a file shorter than 30 s)
            self._chunk_written(file)

    def _chunk_written(self, file: SourceFile) -> None:
        with self._pending_lock:
            state = self._pending.get(file.file_id)
            if state is None:
                return
            state.remaining -= 1
            if state.remaining > 0:
                return
            self._pending.pop(file.file_id, None)
            self._open_files -= 1
        self._file_done(file)

    def _file_done(self, file: SourceFile) -> None:
        self.progress.done_file(file.duration_s / 3600.0)
        with self._lock:
            self.stats.files += 1
            self._files_done += 1
            done = self._files_done
        if self.mark_done:
            self.state.set_status(file.file_id, "done")
        if self.crash_after_files is not None and done >= self.crash_after_files:
            self.stop.set()

    def _error(self, file: SourceFile, message: str) -> None:
        self._error_text(f"{file.file_id}: {message}")
        if self.mark_done:
            self.state.set_status(file.file_id, "error")

    def _error_text(self, message: str) -> None:
        log.warning("%s", message)
        self.progress.error(message)
        with self._lock:
            self.stats.errors += 1
            self._errors.append(message)

    # -- run ---------------------------------------------------------------------
    def run(self, files: Sequence[SourceFile], timeout: float | None = None) -> Stats:
        started = time.perf_counter()
        self._n_downloaders = min(self.download_workers, max(1, len(files)))
        self._n_decoders = min(self.decode_workers, max(1, len(files)))
        self._downloaders_active = self._n_downloaders
        self._decoders_active = self._n_decoders
        self._vads_active = self.vad_workers
        self.progress.set_totals(
            len(files), sum(file.duration_s for file in files) / 3600.0
        )
        threads = [
            *(
                threading.Thread(target=self._downloader, name=f"dl{i}", daemon=True)
                for i in range(self._n_downloaders)
            ),
            *(
                threading.Thread(target=self._decoder, name=f"dec{i}", daemon=True)
                for i in range(self._n_decoders)
            ),
            *(
                threading.Thread(target=self._vad_worker, name=f"vad{i}", daemon=True)
                for i in range(self.vad_workers)
            ),
            threading.Thread(target=self._model_worker, name="model", daemon=True),
            threading.Thread(target=self._writer, name="writer", daemon=True),
        ]
        if self.uploader is not None:
            threads.append(
                threading.Thread(target=self._uploader_loop, name="uploader", daemon=True)
            )
        self._threads = threads
        for thread in threads:
            thread.start()
        feeder = threading.Thread(target=self.feed, args=(files,), name="feed", daemon=True)
        feeder.start()
        deadline = None if timeout is None else started + timeout
        try:
            for thread in [*threads, feeder]:
                remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
                thread.join(remaining)
                if deadline is not None and thread.is_alive():
                    self.stop.set()
                    log.warning("stopping %s: timeout", thread.name)
        finally:
            self.stats.wall_seconds = time.perf_counter() - started
            self.stop.set()
        return self.stats


def _duration(wav: np.ndarray) -> float:
    return float(np.asarray(wav).shape[0]) / vad.SAMPLE_RATE


def _slice(wav: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    start = round(start_s * vad.SAMPLE_RATE)
    end = min(round(end_s * vad.SAMPLE_RATE), wav.shape[0])
    return np.ascontiguousarray(wav[max(0, start) : max(0, end)])


def install_stop_handlers(runner: Runner) -> Callable[[], None]:
    """SIGINT/SIGTERM ask the run to stop taking files; nothing partial is kept."""
    previous: dict[int, Any] = {}

    def handler(signum, frame):
        del frame
        log.warning("signal %d: stopping after the files in flight", signum)
        runner.stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(ValueError):
            previous[sig] = signal.signal(sig, handler)

    def restore() -> None:
        for sig, old in previous.items():
            with contextlib.suppress(ValueError):
                signal.signal(sig, old)

    return restore


# -- real stages -------------------------------------------------------------------
def reader_for(file: SourceFile, cache: dict[str, Any]) -> Any:
    """One `RangeReader` per archive URL, shared by the download threads."""
    from .remote import RangeReader

    reader = cache.get(file.archive_url)
    if reader is None:
        reader = RangeReader(file.archive_url)
        cache[file.archive_url] = reader
    return reader


def real_stages(
    *,
    device: str = "cpu",
    models_dir: str = "work/models",
    wavlm_dtype: str = "float32",
    units_dir: str = "work/units",
    mimi_batch: int = 16,
    wavlm_batch: int = 4,
    vad_workers: int = 8,
    load_codebooks: bool = True,
) -> Stages:
    """The production stages: SoundFile, Silero, WavLM/ZeroSyl, Mimi, codebooks.

    `fit-units` passes `load_codebooks=False`: it is what creates them.
    """
    readers: dict[str, Any] = {}
    codebooks = {k: kmeans.load(units_dir, k) for k in (4096, 8192)} if load_codebooks else {}
    encoder = {"wavlm": None, "mimi": None}

    def download(file: SourceFile) -> bytes:
        reader = reader_for(file, readers)
        return reader.read(file.flac_offset, file.flac_size)

    def decode(raw: bytes, file: SourceFile) -> np.ndarray:
        import soundfile

        audio, rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        if rate != vad.SAMPLE_RATE:
            raise ValueError(f"{file.file_id}: {rate} Hz, expected 16000")
        if audio.shape[1] != 1:
            raise ValueError(f"{file.file_id}: {audio.shape[1]} channels, expected mono")
        return np.ascontiguousarray(audio[:, 0])

    from . import zerosyl

    speech = vad.speech_pool(vad_workers)
    encoder["wavlm"] = zerosyl.load_encoder(device=device, dtype=wavlm_dtype, models_dir=models_dir)
    if load_codebooks:  # a run needs Mimi from the first chunk; fit-units never does
        from . import mimi

        encoder["mimi"] = mimi.load_encoder(device=device, models_dir=models_dir)

    def encode_units(wav: np.ndarray, spans):
        return encoder["wavlm"].encode(wav, spans)

    def encode_units_batch(wavs: list[np.ndarray], spans: list[list]):
        return encoder["wavlm"].encode_batch(wavs, spans)

    def encode_mimi(chunks: list[np.ndarray]) -> list[np.ndarray]:
        if encoder["mimi"] is None:
            from . import mimi

            encoder["mimi"] = mimi.load_encoder(device=device, models_dir=models_dir)
        return encoder["mimi"].encode_batch(chunks)

    def assign(emb: np.ndarray, k: int) -> np.ndarray:
        if emb.shape[0] == 0:
            return np.zeros(0, dtype=np.int16)
        return kmeans.assign(emb, codebooks[k], device=device)

    return Stages(
        download=download,
        decode=decode,
        speech=speech,
        encode_units=encode_units,
        encode_units_batch=encode_units_batch,
        encode_mimi=encode_mimi,
        assign=assign,
        mimi_batch=mimi_batch,
        wavlm_batch=wavlm_batch,
    )
