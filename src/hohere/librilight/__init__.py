"""Libri-Light ingestion: tar indexing, source pairing, ids, segmentation and encoding.

The archive is read over HTTP Range requests, so `tarindex` never seeks: a worker
splits the archive with `split_points`, finds a safe start with `find_first_header`
and walks its slice. `files` pairs the FLAC of an utterance with its JSON and gives
each speaker exactly one worker, `ids` names segments, and `segment` cuts a file into
utterance-length pieces.

Part two adds the archive-to-tokens pipeline: `index` walks the tar and writes a plan,
`plan` holds its lines, `activity` has the frame/pause rules, `encode` runs the Mimi
codec (torch is imported lazily, so importing this package needs only numpy), and
`finalize` merges, checks and packs the release.
"""

from hohere.librilight.activity import (
    SILENCE_CODES,
    activity,
    has_pause,
    is_silence,
    num_frames,
)
from hohere.librilight.files import SourceFile, assign_parts, pair_entries
from hohere.librilight.flac import StreamInfo, parse_streaminfo
from hohere.librilight.ids import chapter_id, check_unique_chapters, parse_segment_id, segment_id
from hohere.librilight.plan import PlanFile, assign_plan_parts, filter_files, read_plan
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
    "SILENCE_CODES",
    "PlanFile",
    "Segment",
    "SegmentConfig",
    "SourceFile",
    "StreamInfo",
    "TarHeader",
    "activity",
    "assign_parts",
    "assign_plan_parts",
    "chapter_id",
    "check_unique_chapters",
    "cut_segments",
    "filter_files",
    "find_first_header",
    "has_pause",
    "is_silence",
    "merge_intervals",
    "next_offset",
    "num_frames",
    "pair_entries",
    "parse_header",
    "parse_segment_id",
    "parse_streaminfo",
    "read_plan",
    "segment_id",
    "split_points",
    "voiced_fraction",
    "walk",
]
