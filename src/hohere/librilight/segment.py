"""Cut a speech file into utterance-length segments from its voice activity.

The VAD of a Libri-Light JSON is a list of `[start_s, end_s]` speech intervals. The
cutting rules (see `segment`) grow a segment run by run until it reaches `min_len`,
never cut inside a pause shorter than `cut_gap`, never cut inside a single long run,
and only keep segments that have a real internal pause with some speech on both sides
of it — otherwise a segment is mostly silence and worthless for training.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

Bounds = tuple[float, float]


@dataclass(frozen=True)
class SegmentConfig:
    """`min_a`/`min_b` are how much speech must sit before/after the internal pause."""

    min_len: float = 6.0
    max_len: float = 30.0
    cut_gap: float = 0.3
    edge_pad: float = 0.5
    pause: float = 0.24
    min_a: float = 3.0
    min_b: float = 2.0


DEFAULT_CONFIG = SegmentConfig()


@dataclass(frozen=True)
class Segment:
    """A cut into the source file, in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class _Candidate:
    start: float
    end: float
    intervals: list[Bounds]


def merge_intervals(voice_activity: Iterable[Bounds], duration: float) -> list[Bounds]:
    """Clip to [0, duration], drop empty intervals, sort and merge touching ones."""
    clipped: list[Bounds] = []
    for start, end in voice_activity:
        s, e = max(0.0, min(float(start), duration)), max(0.0, min(float(end), duration))
        if e > s:
            clipped.append((s, e))
    clipped.sort()
    merged: list[Bounds] = []
    for s, e in clipped:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _groups(intervals: list[Bounds], cut_gap: float) -> list[list[Bounds]]:
    """Consecutive intervals separated by less than `cut_gap` form one run."""
    groups: list[list[Bounds]] = []
    for interval in intervals:
        if groups and interval[0] - groups[-1][-1][1] < cut_gap:
            groups[-1].append(interval)
        else:
            groups.append([interval])
    return groups


def runs(intervals: list[Bounds], cut_gap: float) -> list[Bounds]:
    """Maximal stretches of speech joined by gaps shorter than `cut_gap`.

    A segment boundary may only fall between two runs, never inside one.
    """
    return [(group[0][0], group[-1][1]) for group in _groups(intervals, cut_gap)]


def _padded_spans(span_list: list[Bounds], duration: float, edge_pad: float) -> list[Bounds]:
    """Runs widened by the edge padding: at most half of the gap to the neighbouring
    run, and at most the room up to the edge of the file for the first and last one."""
    out: list[Bounds] = []
    last = len(span_list) - 1
    for i, (start, end) in enumerate(span_list):
        before = start if i == 0 else (start - span_list[i - 1][1]) / 2
        after = duration - end if i == last else (span_list[i + 1][0] - end) / 2
        out.append((start - min(edge_pad, before), end + min(edge_pad, after)))
    return out


def _has_internal_pause(cand: _Candidate, cfg: SegmentConfig) -> bool:
    """True if some gap inside the segment is long enough and far enough from both edges."""
    for (_, gap_start), (gap_end, _) in pairwise(cand.intervals):
        if (
            gap_end - gap_start >= cfg.pause
            and gap_start - cand.start >= cfg.min_a
            and cand.end - gap_end >= cfg.min_b
        ):
            return True
    return False


def segment(
    voice_activity: Iterable[Bounds], duration: float, cfg: SegmentConfig = DEFAULT_CONFIG
) -> list[Segment]:
    """Cut a file of `duration` seconds into segments, in time order."""
    if duration <= 0:
        return []
    groups = _groups(merge_intervals(voice_activity, duration), cfg.cut_gap)
    if not groups:
        return []
    spans = _padded_spans([(g[0][0], g[-1][1]) for g in groups], duration, cfg.edge_pad)

    candidates: list[_Candidate] = []
    current: list[int] = []
    dirty = False  # a stretch was discarded since the last kept candidate

    def close() -> None:
        """Finish the running accumulation: keep it if it is long enough."""
        nonlocal current, dirty
        if not current:
            return
        start, end = spans[current[0]][0], spans[current[-1]][1]
        if end - start >= cfg.min_len:
            intervals = [iv for i in current for iv in groups[i]]
            candidates.append(_Candidate(start, end, intervals))
            dirty = False
        else:
            dirty = True
        current = []

    def _ready(indices: list[int]) -> bool:
        start, end = spans[indices[0]][0], spans[indices[-1]][1]
        intervals = [iv for i in indices for iv in groups[i]]
        return _has_internal_pause(_Candidate(start, end, intervals), cfg)

    for i, (start, end) in enumerate(spans):
        if end - start > cfg.max_len:
            # Rule 4: a run longer than max_len is dropped, never cut inside. The padded
            # span is compared, so no kept segment can exceed max_len because of padding.
            close()
            dirty = True
            continue
        # Libri-Light VAD runs are long (median ~10 s), so one run often reaches min_len
        # alone and has no pause inside. Keep growing until the segment also has a usable
        # internal pause; on overflow drop runs from the front instead of the whole stretch.
        while current and end - spans[current[0]][0] > cfg.max_len:
            current.pop(0)
            dirty = True
        current.append(i)
        if end - spans[current[0]][0] >= cfg.min_len and _ready(current):
            close()

    if current:
        # Only a segment shorter than min_len can still be open here (rule 3 closes
        # the others immediately): merge it into its neighbour if that is adjacent and
        # still fits, otherwise drop it.
        start, end = spans[current[0]][0], spans[current[-1]][1]
        tail = [iv for i in current for iv in groups[i]]
        if candidates and not dirty and end - candidates[-1].start <= cfg.max_len:
            prev = candidates.pop()
            candidates.append(_Candidate(prev.start, end, prev.intervals + tail))
        current = []

    return [Segment(c.start, c.end) for c in candidates if _has_internal_pause(c, cfg)]


def voiced_fraction(voice_activity: Iterable[Bounds], duration: float) -> float:
    """Share of the file that is speech."""
    if duration <= 0:
        return 0.0
    return sum(e - s for s, e in merge_intervals(voice_activity, duration)) / duration
