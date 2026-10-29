"""Libri-Light ingestion: tar indexing, source pairing, ids and VAD segmentation.

The archive is read over HTTP Range requests, so `tarindex` never seeks: a worker
splits the archive with `split_points`, finds a safe start with `find_first_header`
and walks its slice. `files` pairs the FLAC of an utterance with its JSON and gives
each speaker exactly one worker, `ids` names segments, and `segment` cuts a file
into utterance-length pieces. Encoding to Mimi tokens is not part of this package.
"""

from hohere.librilight.files import SourceFile, assign_parts, pair_entries
from hohere.librilight.ids import chapter_id, check_unique_chapters, parse_segment_id, segment_id
from hohere.librilight.segment import (
    Segment,
    SegmentConfig,
    merge_intervals,
    voiced_fraction,
)

# `segment` names both the module and its main function, so the package exports the
# function under a name that cannot be shadowed by the submodule.
from hohere.librilight.segment import segment as cut_segments
from hohere.librilight.tarindex import (
    TarHeader,
    find_first_header,
    next_offset,
    parse_header,
    split_points,
    walk,
)

__all__ = [
    "Segment",
    "SegmentConfig",
    "SourceFile",
    "TarHeader",
    "assign_parts",
    "chapter_id",
    "check_unique_chapters",
    "cut_segments",
    "find_first_header",
    "merge_intervals",
    "next_offset",
    "pair_entries",
    "parse_header",
    "parse_segment_id",
    "segment_id",
    "split_points",
    "voiced_fraction",
    "walk",
]
