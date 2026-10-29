"""Plan lines: one source file with its duration, `snr` and the segments to encode.

`WORK/plan.jsonl` is the hand-off between `index` (which touches the archive) and
`encode` (which touches a GPU). One line per file that survived segmentation, sorted
by speaker and chapter, so the file is stable across runs and diffable.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from hohere.librilight.files import SourceFile, balance_by_speaker
from hohere.librilight.ids import chapter_id
from hohere.librilight.segment import DEFAULT_CONFIG, SegmentConfig, segment


@dataclass(frozen=True)
class PlanFile:
    """A source file and the segments `index` wants from it, in time order."""

    speaker: str
    book_dir: str
    stem: str
    chapter: str
    flac_offset: int
    flac_size: int
    duration: float
    snr: float
    segments: tuple[tuple[float, float], ...]

    @property
    def key(self) -> tuple[int, str]:
        return int(self.speaker), self.chapter

    @property
    def archive_path(self) -> str:
        """The path of the FLAC inside the tar this file was cut from."""
        return f"medium/{self.speaker}/{self.book_dir}/{self.stem}.flac"

    def line(self) -> dict:
        return {
            "speaker": self.speaker,
            "book_dir": self.book_dir,
            "stem": self.stem,
            "chapter": self.chapter,
            "flac_offset": self.flac_offset,
            "flac_size": self.flac_size,
            "duration": round(self.duration, 6),
            "snr": self.snr,
            "segments": [[round(s, 6), round(e, 6)] for s, e in self.segments],
        }


def parse_line(raw: dict) -> PlanFile:
    """Inverse of `PlanFile.line`; a line from `WORK/plan.jsonl`."""
    return PlanFile(
        speaker=str(raw["speaker"]),
        book_dir=raw["book_dir"],
        stem=raw["stem"],
        chapter=raw["chapter"],
        flac_offset=int(raw["flac_offset"]),
        flac_size=int(raw["flac_size"]),
        duration=float(raw["duration"]),
        snr=float(raw["snr"]),
        segments=tuple((float(s), float(e)) for s, e in raw["segments"]),
    )


def build(
    file: SourceFile,
    duration: float,
    snr: float,
    voice_activity: Iterable[tuple[float, float]],
    cfg: SegmentConfig = DEFAULT_CONFIG,
) -> PlanFile | None:
    """One plan entry, or None when the VAD yields no usable segment."""
    cuts = segment(voice_activity, duration, cfg)
    if not cuts:
        return None
    return PlanFile(
        speaker=file.speaker,
        book_dir=file.book_dir,
        stem=file.stem,
        chapter=chapter_id(file.book_dir, file.stem),
        flac_offset=file.flac.data_offset,
        flac_size=file.flac.size,
        duration=duration,
        snr=snr,
        segments=tuple((s.start, s.end) for s in cuts),
    )


def filter_files(
    files: Iterable[SourceFile],
    exclude_speakers: Iterable[str] = (),
    exclude_recordings: Iterable[Iterable[str]] = (),
) -> tuple[list[SourceFile], dict[str, int]]:
    """Drop excluded speakers and recordings; the second value counts the drops."""
    speakers = {str(s) for s in exclude_speakers}
    recordings = {(str(s), str(t)) for s, t in exclude_recordings}
    kept: list[SourceFile] = []
    dropped: Counter[str] = Counter()
    for f in files:
        if f.speaker in speakers:
            dropped["speaker"] += 1
        elif (f.speaker, f.stem) in recordings:
            dropped["recording"] += 1
        else:
            kept.append(f)
    return kept, dict(dropped)


def assign_plan_parts(files: Sequence[PlanFile], parts: int) -> list[list[PlanFile]]:
    """Shards of whole speakers, balanced by FLAC bytes; see `files.assign_parts`."""
    by_speaker: dict[str, list[PlanFile]] = defaultdict(list)
    for f in files:
        by_speaker[f.speaker].append(f)
    sizes = {s: sum(f.flac_size for f in group) for s, group in by_speaker.items()}
    return balance_by_speaker(by_speaker, sizes, parts, order=lambda f: f.flac_offset)


def read_plan(path: Path) -> list[PlanFile]:
    with open(path) as f:
        return [parse_line(json.loads(line)) for line in f if line.strip()]


def write_plan(path: Path, files: Iterable[PlanFile]) -> int:
    """Write sorted plan lines, atomically; returns how many were written."""
    ordered = sorted(files, key=lambda f: f.key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for plan_file in ordered:
            f.write(json.dumps(plan_file.line(), sort_keys=False) + "\n")
    tmp.replace(path)
    return len(ordered)
