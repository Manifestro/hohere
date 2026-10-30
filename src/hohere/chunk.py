"""Cut a recording into 30–60 s chunks on the Libri-Light VAD.

The VAD gives the speech spans of a file; every chunk boundary has to fall in a gap
between two spans. A chunk is `[first span start - lead, last span end + tail]` where
`lead`/`tail` are at most 0.5 s and at most half of the surrounding gap, so edge
silence is short and two neighbouring chunks never overlap.

The rule, in order:

1. grow a chunk span by span; once its speech extent reaches 30 s, look ahead for the
   widest gap that still keeps the whole chunk (edges included) within 60 s and cut
   there;
2. if even the shortest chunk that reaches 30 s would overshoot 60 s (25 s of spans
   followed by a 40 s span, say), drop the chunk's leading span(s) — counting each in
   `dropped_lead` — and keep growing from the span that follows;
3. a span longer than 60 s on its own cannot be chunked and is dropped
   (`dropped_long`);
4. a tail shorter than 30 s is merged into the previous chunk when the result still
   fits in 60 s (`merged_tail`), and dropped otherwise (`dropped_tail`).

So every chunk `plan` emits is within [30, 60] s, boundaries only ever fall in a VAD
gap, and neighbouring chunks never overlap. Dropped pieces are counted, never silently
ignored.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

MIN_S = 30.0
MAX_S = 60.0
EDGE_S = 0.5

Span = tuple[float, float]


@dataclass
class ChunkPlan:
    """Chunks of one file plus the counters of what was thrown away."""

    chunks: list[Span] = field(default_factory=list)
    dropped_long: int = 0  # a span too long to ever fit
    dropped_tail: int = 0  # a tail shorter than min_s that could not be merged
    dropped_lead: int = 0  # leading spans of an overshooting chunk
    merged_tail: int = 0  # tails folded into the previous chunk

    @property
    def kept_seconds(self) -> float:
        return sum(end - start for start, end in self.chunks)


def normalize(spans: Iterable[Sequence[float]], duration: float) -> list[Span]:
    """Clip spans to the file, drop empties, sort and merge overlaps."""
    clipped: list[Span] = []
    for span in spans:
        start, end = float(span[0]), float(span[1])
        start, end = max(0.0, start), min(float(duration), end)
        if end > start:
            clipped.append((start, end))
    clipped.sort()
    merged: list[Span] = []
    for start, end in clipped:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _lead(spans: list[Span], index: int, edge: float) -> float:
    previous = 0.0 if index == 0 else spans[index - 1][1]
    gap = spans[index][0] - previous
    return min(gap / 2, edge) if index else min(gap, edge)


def _tail(spans: list[Span], index: int, duration: float, edge: float) -> float:
    following = float(duration) if index == len(spans) - 1 else spans[index + 1][0]
    gap = following - spans[index][1]
    return min(gap / 2, edge) if index < len(spans) - 1 else min(gap, edge)


def _bounds(
    spans: list[Span], first: int, last: int, duration: float, edge: float
) -> Span:
    start = max(0.0, spans[first][0] - _lead(spans, first, edge))
    end = min(float(duration), spans[last][1] + _tail(spans, last, duration, edge))
    return start, end


def plan(
    vad: Iterable[Sequence[float]],
    duration: float,
    *,
    min_s: float = MIN_S,
    max_s: float = MAX_S,
    edge_s: float = EDGE_S,
) -> ChunkPlan:
    """Chunk one file: `vad` is a list of `(start_s, end_s)` speech spans."""
    spans = normalize(vad, duration)
    result = ChunkPlan()
    too_long = [end - start > max_s for start, end in spans]
    count = len(spans)
    index = 0
    while index < count:
        if too_long[index]:  # a span that cannot be cut down to size
            result.dropped_long += 1
            index += 1
            continue
        stop = index  # a too-long span ends the run of usable spans
        while stop < count and not too_long[stop]:
            stop += 1
        while index < stop:
            # The first span that brings the speech extent of the chunk to >= min_s.
            last = index
            while last < stop and spans[last][1] - spans[index][0] < min_s:
                last += 1
            if last >= stop:  # short tail of this run
                end = _bounds(spans, index, stop - 1, duration, edge_s)[1]
                if result.chunks and end - result.chunks[-1][0] <= max_s:
                    result.chunks[-1] = (result.chunks[-1][0], end)
                    result.merged_tail += 1
                else:
                    result.dropped_tail += 1
                index = stop
                break

            # Look ahead for the widest gap that still keeps the chunk within max_s.
            cut: int | None = None
            best_gap = -1.0
            candidate = last
            while candidate < stop:
                start, end = _bounds(spans, index, candidate, duration, edge_s)
                if end - start > max_s:  # every later candidate is only longer
                    break
                if candidate + 1 < stop:
                    gap = spans[candidate + 1][0] - spans[candidate][1]
                    if gap > best_gap:
                        best_gap, cut = gap, candidate
                candidate += 1
            if cut is None:
                start, end = _bounds(spans, index, last, duration, edge_s)
                if end - start <= max_s and last < stop:
                    cut = last  # the shortest chunk that reaches min_s does fit
                else:
                    # It overshoots 60 s: drop this leading span and grow again from
                    # the next one, so the chunk never grows past max_s.
                    result.dropped_lead += 1
                    index += 1
                    continue
            result.chunks.append(_bounds(spans, index, cut, duration, edge_s))
            index = cut + 1
    return result


def chunks(
    vad: Iterable[Sequence[float]],
    duration: float,
    *,
    min_s: float = MIN_S,
    max_s: float = MAX_S,
    edge_s: float = EDGE_S,
) -> list[Span]:
    """Just the chunk boundaries of `plan`."""
    return plan(vad, duration, min_s=min_s, max_s=max_s, edge_s=edge_s).chunks
