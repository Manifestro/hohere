"""`hohere label`: arbitrary audio -> the same records a run writes.

For test stimuli (sWUGGY, sBLIMP, StoryCloze) and later CANDOR: Silero, WavLM cut at
pauses, the same SIL rule and the same centroids as `run`, so `build_m` reads the result
unchanged. What differs is only how a file becomes pieces, since there is no Libri-Light
VAD and most stimuli are far under 30 s:

* a file up to 60 s is one piece, with edge silence trimmed to 0.5 s;
* a longer file is chunked as `run` chunks, on the Silero spans of the whole file;
* a file with no speech stays whole and gets a record without units.

Silero runs again on each piece, as `run` does on a chunk. A file that fails does not
stop the others. `manifest.jsonl` gets all the lines of a file at once after its records,
so a restart skips exactly the files that are complete.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import chunk, pipeline, records, vad
from .pipeline import Stages

log = logging.getLogger(__name__)

AUDIO_SUFFIXES = (".wav", ".flac")
GROUP_PIECES = 64  # pieces gathered before the models run, so that batches can form


@dataclass
class Source:
    """One input file, cut into the pieces that become records."""

    name: str  # path under the input directory, without the suffix
    rate: int
    pieces: list[tuple[float, float]]
    wavs: list[np.ndarray]

    def record_id(self, index: int) -> str:
        return self.name if len(self.pieces) == 1 else records.chunk_id(self.name, index)


def find_audio(in_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(in_dir)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Mono float32 at 16 kHz, and the rate of the file."""
    import soundfile

    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        raise ValueError(f"{audio.shape[1]} channels, expected mono: label each channel apart")
    wav = np.ascontiguousarray(audio[:, 0])
    if rate != vad.SAMPLE_RATE:
        import torch
        import torchaudio

        wav = torchaudio.functional.resample(
            torch.from_numpy(wav), rate, vad.SAMPLE_RATE
        ).numpy()
    return wav, rate


def plan_pieces(wav: np.ndarray, spans: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """`(start_s, end_s)` of the pieces of one file, from its Silero spans."""
    duration = wav.shape[0] / vad.SAMPLE_RATE
    if not spans:
        return [(0.0, duration)] if wav.size else []
    seconds = [(start / vad.SAMPLE_RATE, end / vad.SAMPLE_RATE) for start, end in spans]
    if duration > chunk.MAX_S:
        return chunk.chunks(seconds, duration)
    return [
        (
            max(0.0, seconds[0][0] - chunk.EDGE_S),
            min(duration, seconds[-1][1] + chunk.EDGE_S),
        )
    ]


def load_source(path: Path, root: Path, stages: Stages) -> Source:
    wav, rate = load_audio(path)
    pieces = plan_pieces(wav, stages.speech(wav))
    wavs = [pipeline._slice(wav, start, end) for start, end in pieces]
    return Source(path.relative_to(root).with_suffix("").as_posix(), rate, pieces, wavs)


def encode_sources(sources: list[Source], stages: Stages) -> list[list[dict[str, Any]]]:
    """The arrays of every piece, grouped by source."""
    wavs = [wav for source in sources for wav in source.wavs]
    spans = [stages.speech(wav) for wav in wavs]
    units = pipeline.encode_many(stages, wavs, spans)
    codes = pipeline.encode_mimi_sorted(stages, wavs)
    arrays = iter(
        zip(units, codes, spans, strict=True)
    )
    out = []
    for source in sources:
        out.append(
            [
                pipeline.unit_record(stages, segments, code, span)
                for segments, code, span in (next(arrays) for _ in source.wavs)
            ]
        )
    return out


def write_source(
    source: Source, arrays: list[dict[str, Any]], out_dir: Path, version: dict[str, str]
) -> str:
    """Write the records of one file; returns its manifest lines."""
    lines = []
    for index, (record, (start_s, end_s)) in enumerate(zip(arrays, source.pieces, strict=True)):
        record_id = source.record_id(index)
        path = out_dir / f"{record_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(records.write_record(**record))
        line = {
            "id": record_id,
            "source": source.name,
            "source_rate": source.rate,
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": end_s - start_s,
            "frames": int(record["codes"].shape[-1]),
            "n_units": int(record["unit_start"].shape[0]),
            "n_sil": int(record["sil_start"].shape[0]),
            **version,
        }
        lines.append(json.dumps(line, sort_keys=True) + "\n")
    return "".join(lines)


def run(
    in_dir: str | os.PathLike[str], out_dir: str | os.PathLike[str], stages: Stages
) -> dict[str, Any]:
    """Label every audio file under `in_dir`; returns counts and the failed files."""
    root, out = Path(in_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    finished = (
        {json.loads(line)["source"] for line in manifest.read_text().splitlines() if line}
        if manifest.exists()
        else set()
    )
    summary: dict[str, Any] = {"files": 0, "records": 0, "units": 0, "no_units": 0,
                               "skipped": 0, "errors": {}}

    with manifest.open("a", encoding="utf-8") as handle:

        def finish(sources: list[Source]) -> None:
            try:
                arrays = encode_sources(sources, stages)
            except Exception as exc:
                if len(sources) == 1:  # one bad file: name it, keep the others
                    summary["errors"][sources[0].name] = str(exc)
                    log.warning("%s: %s", sources[0].name, exc)
                    return
                for source in sources:
                    finish([source])
                return
            for source, records_of in zip(sources, arrays, strict=True):
                handle.write(write_source(source, records_of, out, stages.units_version))
                handle.flush()
                units = [int(record["unit_start"].shape[0]) for record in records_of]
                summary["files"] += 1
                summary["records"] += len(records_of)
                summary["units"] += sum(units)
                summary["no_units"] += sum(count == 0 for count in units)

        pending: list[Source] = []
        for path in find_audio(root):
            name = path.relative_to(root).with_suffix("").as_posix()
            if name in finished:
                summary["skipped"] += 1
                continue
            try:
                pending.append(load_source(path, root, stages))
            except Exception as exc:
                summary["errors"][name] = str(exc)
                log.warning("%s: %s", name, exc)
                continue
            if sum(len(source.wavs) for source in pending) >= GROUP_PIECES:
                finish(pending)
                pending = []
        if pending:
            finish(pending)
    return summary
