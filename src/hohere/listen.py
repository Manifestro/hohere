"""`hohere listen`: chunks with their boundaries, for a person to check by ear.

Shards hold no audio, so this reads the same files as a run and writes, per chunk:

* `NNN.wav` — the chunk as it is;
* `NNN-clicks.wav` — the same with a high click at each unit start and a low one at
  each pause (SIL), so the boundaries can be heard in any player;
* `NNN.txt` — the same events as Audacity labels (File > Import > Labels).

`index.json` has the numbers of each chunk. Codebooks are not needed: boundaries come
before k-means, so this can run before `fit-units`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import chunk, pipeline, vad
from .pipeline import Stages

WAVLM_S = vad.WAVLM_HOP / vad.SAMPLE_RATE
CLICK_S = 0.008
CLICK_UNIT_HZ = 2000.0
CLICK_SIL_HZ = 500.0
CLICK_LEVEL = 0.35


def pick_files(files: list, count: int) -> list:
    """`count` files spread evenly over the tier."""
    if count <= 0 or not files:
        return []
    count = min(count, len(files))
    return [files[index * len(files) // count] for index in range(count)]


def with_clicks(wav: np.ndarray, units: np.ndarray, pauses: np.ndarray) -> np.ndarray:
    """The waveform with a short tone mixed in at each event (frames at 50 Hz)."""
    out = np.array(wav, dtype=np.float32)
    length = int(CLICK_S * vad.SAMPLE_RATE)
    time = np.arange(length) / vad.SAMPLE_RATE
    for frames, hertz in ((units, CLICK_UNIT_HZ), (pauses, CLICK_SIL_HZ)):
        tone = CLICK_LEVEL * np.sin(2 * np.pi * hertz * time).astype(np.float32)
        for frame in np.asarray(frames).reshape(-1):
            start = int(frame) * vad.WAVLM_HOP
            part = out[start : start + length]
            part += tone[: part.size]
    return np.clip(out, -1.0, 1.0)


def labels(unit_start: np.ndarray, unit_end: np.ndarray, pauses: np.ndarray) -> str:
    """Audacity label lines: start, end, text, in time order."""
    rows = [
        (float(start) * WAVLM_S, float(end) * WAVLM_S, f"u{index}")
        for index, (start, end) in enumerate(zip(unit_start, unit_end, strict=True))
    ]
    rows += [(float(frame) * WAVLM_S, float(frame) * WAVLM_S, "SIL") for frame in pauses]
    return "".join(f"{a:.3f}\t{b:.3f}\t{text}\n" for a, b, text in sorted(rows))


def write_chunk(
    out_dir: Path, index: int, wav: np.ndarray, segments: Any, pauses: np.ndarray, meta: dict
) -> dict:
    """The three files of one chunk; returns its line of `index.json`."""
    import soundfile

    stem = f"{index:03d}"
    soundfile.write(out_dir / f"{stem}.wav", wav, vad.SAMPLE_RATE)
    soundfile.write(
        out_dir / f"{stem}-clicks.wav",
        with_clicks(wav, segments.unit_start, pauses),
        vad.SAMPLE_RATE,
    )
    (out_dir / f"{stem}.txt").write_text(labels(segments.unit_start, segments.unit_end, pauses))
    seconds = wav.shape[0] / vad.SAMPLE_RATE
    return {
        "name": stem,
        **meta,
        "seconds": round(seconds, 2),
        "units": len(segments),
        "units_per_s": round(len(segments) / seconds, 2) if seconds else 0.0,
        "sil": int(pauses.shape[0]),
    }


def run(files: list, stages: Stages, out_dir: str | os.PathLike[str], count: int = 20) -> dict:
    """Write `count` chunks, the middle chunk of each picked file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for file in pick_files(files, count):
        wav = stages.decode(stages.download(file), file)
        chunks = chunk.chunks(file.vad, wav.shape[0] / vad.SAMPLE_RATE)
        if not chunks:
            continue
        middle = len(chunks) // 2
        start_s, end_s = chunks[middle]
        piece = pipeline._slice(wav, start_s, end_s)
        spans = stages.speech(piece)
        segments = pipeline.encode_many(stages, [piece], [spans])[0]
        pauses = np.asarray(vad.pauses(spans), dtype=np.int32)
        meta = {"file_id": file.file_id, "chunk": middle, "start_s": start_s}
        rows.append(write_chunk(out, len(rows), piece, segments, pauses, meta))
    seconds = sum(row["seconds"] for row in rows)
    report = {
        "chunks": rows,
        "units_per_s": round(sum(row["units"] for row in rows) / seconds, 2) if seconds else 0.0,
    }
    (out / "index.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
