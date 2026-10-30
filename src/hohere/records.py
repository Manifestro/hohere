"""The on-disk record of one chunk.

One chunk is one `.npz` member of a shard tar:

    codes        int16 [32, T]   Mimi codes, T = ceil(samples_24k / 1920)
    unit_start   int32 [N]       WavLM 50 Hz frame of the first speech frame
    unit_end     int32 [N]       WavLM 50 Hz frame of the last speech frame
    unit_k4096   int16 [N]       ZeroSyl unit id, 4096-entry vocabulary
    unit_k8192   int16 [N]       ZeroSyl unit id, 8192-entry vocabulary
    sil_start    int32 [S]       WavLM frame where an internal pause >= 200 ms starts

Everything else about the chunk (its id, source file, times, order rank, shard) is a
line in the manifest, not a member of the npz.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np

CODECS = 32
K4096 = 4096
K8192 = 8192
CODE_MAX = 2048  # Mimi codebook size
KEYS = ("codes", "unit_start", "unit_end", "unit_k4096", "unit_k8192", "sil_start")
DTYPES: dict[str, str] = {
    "codes": "int16",
    "unit_start": "int32",
    "unit_end": "int32",
    "unit_k4096": "int16",
    "unit_k8192": "int16",
    "sil_start": "int32",
}


class RecordError(ValueError):
    """A record violates the format contract."""


def chunk_id(file_id: str, index: int) -> str:
    """`id = f"{file_id}-{idx:03d}"`."""
    if index < 0:
        raise ValueError("chunk index must be >= 0")
    return f"{file_id}-{index:03d}"


def validate(record: dict[str, Any], *, codebooks: int = CODECS) -> dict[str, np.ndarray]:
    """Check shapes, dtypes, ordering and ranges; returns the normalised record."""
    arrays: dict[str, np.ndarray] = {}
    for key in KEYS:
        array = np.asarray(record[key]) if key in record else None
        if array is None:
            raise RecordError(f"missing key {key!r}")
        if array.dtype != np.dtype(DTYPES[key]):
            raise RecordError(f"{key} must be {DTYPES[key]}, got {array.dtype}")
        arrays[key] = array

    codes = arrays["codes"]
    if codes.ndim != 2 or codes.shape[0] != codebooks:
        raise RecordError(f"codes must be [{codebooks}, T], got {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= CODE_MAX):
        raise RecordError(f"codes out of range [0, {CODE_MAX})")

    lengths = {key: arrays[key].shape for key in KEYS if key != "codes"}
    for key, shape in lengths.items():
        if len(shape) != 1:
            raise RecordError(f"{key} must be 1-D, got {shape}")
    n_units = arrays["unit_start"].shape[0]
    for key in ("unit_end", "unit_k4096", "unit_k8192"):
        if arrays[key].shape[0] != n_units:
            raise RecordError(f"{key} has {arrays[key].shape[0]} rows, expected {n_units}")

    start = arrays["unit_start"]
    end = arrays["unit_end"]
    if n_units:
        if start.min() < 0 or end.min() < 0:
            raise RecordError("unit frames must be >= 0")
        if np.any(start >= end):
            raise RecordError("unit_start must be < unit_end")
        if np.any(np.diff(start) < 0):
            raise RecordError("unit_start must be sorted")
        if arrays["unit_k4096"].min() < 0 or arrays["unit_k4096"].max() >= K4096:
            raise RecordError(f"unit_k4096 out of range [0, {K4096})")
        if arrays["unit_k8192"].min() < 0 or arrays["unit_k8192"].max() >= K8192:
            raise RecordError(f"unit_k8192 out of range [0, {K8192})")

    sil = arrays["sil_start"]
    if sil.size:
        if sil.min() < 0:
            raise RecordError("sil_start frames must be >= 0")
        if np.any(np.diff(sil) < 0):
            raise RecordError("sil_start must be sorted")
    return arrays


def write_record(
    *,
    codes: Any,
    unit_start: Any,
    unit_end: Any,
    unit_k4096: Any,
    unit_k8192: Any,
    sil_start: Any,
    codebooks: int = CODECS,
) -> bytes:
    """Serialise one chunk into npz bytes, validating it first."""
    raw = {
        "codes": codes,
        "unit_start": unit_start,
        "unit_end": unit_end,
        "unit_k4096": unit_k4096,
        "unit_k8192": unit_k8192,
        "sil_start": sil_start,
    }
    record = {
        key: np.ascontiguousarray(np.asarray(value, dtype=DTYPES[key]))
        for key, value in raw.items()
    }
    validate(record, codebooks=codebooks)
    buffer = io.BytesIO()
    np.savez(buffer, **record)
    return buffer.getvalue()


def read_record(
    data: bytes, *, codebooks: int = CODECS, check: bool = True
) -> dict[str, np.ndarray]:
    """Parse npz bytes back into a record (and validate it unless `check=False`)."""
    with np.load(io.BytesIO(data), allow_pickle=False) as loaded:
        record = {key: loaded[key] for key in KEYS if key in loaded.files}
    missing = [key for key in KEYS if key not in record]
    if missing:
        raise RecordError(f"missing keys: {missing}")
    return validate(record, codebooks=codebooks) if check else record


def record_frames(record: dict[str, Any]) -> int:
    """`T` of a record."""
    return int(np.asarray(record["codes"]).shape[1])
