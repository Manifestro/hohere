"""`hohere fit-units`: embed a tier with ZeroSyl, then fit the k-means codebooks.

Two phases, both resumable:

1. **embed** — walk the tier in `order_rank` order, run download -> decode -> Silero ->
   WavLM, and write the segment embeddings of every chunk as an fp16 `.npy` part in
   `work/units/emb/` with one line per unit in `emb/index.jsonl` (chunk id,
   `unit_start`, `unit_end`). A part that is already on disk is skipped, so a killed
   job continues where it stopped;
2. **fit** — load every part, fit each requested `k` with spherical k-means and write
   `work/units/k{K}.npy` + `k{K}.json`.

Only embeddings are kept: the raw audio is never stored, and the codebooks are the
only artefact a `run` needs (`k4096.npy`, `k8192.npy`).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import kmeans, records
from .chunk import plan as plan_chunks
from .pipeline import Stages, _duration, _slice
from .status import Progress

log = logging.getLogger(__name__)

PART_FILES = 200  # files per embedding part; ~1 GB of fp16 per part at k=8192 scale
EMB_DTYPE = np.float16


def part_name(index: int) -> str:
    return f"part-{index:05d}.npy"


def _load_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def embed(
    files: list,
    stages: Stages,
    *,
    out_dir: str | os.PathLike[str] = "work/units/emb",
    part_files: int = PART_FILES,
    progress: Progress | None = None,
    max_units_per_part: int = 2_000_000,
) -> dict[str, Any]:
    """Phase 1: write fp16 embedding parts; already written parts are skipped."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.jsonl"
    progress = progress or Progress()
    parts = sorted(p for p in directory.glob("part-*.npy"))
    done = {p.name for p in parts}
    written = len(done)
    units_total = 0
    with index_path.open("a", encoding="utf-8") as index_handle:
        buffer: list[np.ndarray] = []
        pending: list[dict[str, Any]] = []
        part = 0

        def flush() -> None:
            nonlocal buffer, pending, written, part
            if not buffer:
                return
            name = part_name(part)
            if name not in done:
                np.save(directory / name, np.concatenate(buffer).astype(EMB_DTYPE))
                for row in pending:
                    index_handle.write(json.dumps(row, sort_keys=True) + "\n")
                index_handle.flush()
                done.add(name)
                written += 1
            part += 1
            buffer, pending = [], []

        for count, file in enumerate(files):
            name = part_name(part)
            if name in done and count < len(files):  # skip what a previous run wrote
                part += 1
                continue
            raw = stages.download(file)
            wav = stages.decode(raw, file)
            plan = plan_chunks(file.vad, _duration(wav))
            for chunk_index, (start_s, end_s) in enumerate(plan.chunks):
                piece = _slice(wav, start_s, end_s)
                started = time.perf_counter()
                spans = stages.speech(piece)
                progress.add("vad", time.perf_counter() - started)
                started = time.perf_counter()
                segments = stages.encode_units(piece, spans)
                progress.add("wavlm", time.perf_counter() - started)
                if len(segments) == 0:
                    continue
                buffer.append(np.asarray(segments.embeddings, dtype=np.float16))
                chunk_id = records.chunk_id(file.file_id, chunk_index)
                for start, end in zip(segments.unit_start, segments.unit_end, strict=True):
                    pending.append(
                        {"chunk": chunk_id, "unit_start": int(start), "unit_end": int(end)}
                    )
                units_total += len(segments)
                progress.done_chunk((end_s - start_s) / 3600.0)
            if (count + 1) % part_files == 0 or units_total >= max_units_per_part:
                flush()
                units_total = 0
        flush()
        if index_path.exists() and index_path.stat().st_size and not done:
            index_handle.write("")
    return {
        "parts": written,
        "files": len(files),
        "index": str(index_path),
        "directory": str(directory),
    }


def load_embeddings(out_dir: str | os.PathLike[str] = "work/units/emb") -> np.ndarray:
    """All fp16 parts, stacked as float32 [N, D]."""
    directory = Path(out_dir)
    parts = sorted(directory.glob("part-*.npy"))
    if not parts:
        raise FileNotFoundError(f"no embedding parts in {directory}; run `fit-units` first")
    return np.concatenate([np.load(part).astype(np.float32) for part in parts])


def fit_all(
    embeddings: np.ndarray,
    ks: tuple[int, ...] = (4096, 8192),
    *,
    out_dir: str | os.PathLike[str] = "work/units",
    hours: float | None = None,
    device: str = "cpu",
    iters: int = kmeans.ITERS,
    seed: int = kmeans.SEED,
) -> dict[str, Any]:
    """Phase 2: fit every k and write `k{K}.npy`/`k{K}.json`."""
    report: dict[str, Any] = {}
    for k in ks:
        started = time.perf_counter()
        result = kmeans.fit(embeddings, k, iters=iters, seed=seed, device=device)
        npy, meta = kmeans.save(result, out_dir, hours=hours, device=str(device))
        report[str(k)] = {
            "npy": str(npy),
            "json": str(meta),
            "seconds": round(time.perf_counter() - started, 1),
            "objective": result.objective[-1] if result.objective else None,
        }
    return report


def upload(units_dir: str | os.PathLike[str], repo_id: str, *, ks=(4096, 8192),
           upload_embeddings: bool = False, embeddings_dir: str | os.PathLike[str] | None = None,
           api: Any = None) -> list[str]:
    """Push the codebooks (and optionally the embeddings) to the dataset repo."""
    from huggingface_hub import HfApi

    api = api or HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    directory = Path(units_dir)
    files = [directory / f"k{k}.npy" for k in ks] + [directory / f"k{k}.json" for k in ks]
    if upload_embeddings:
        source = Path(embeddings_dir) if embeddings_dir else directory / "emb"
        files += sorted(source.glob("part-*.npy"))
    uploaded: list[str] = []
    for path in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"units/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        uploaded.append(path.name)
    return uploaded
