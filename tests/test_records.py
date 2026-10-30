"""Record format: round trip and the validator."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import record

from hohere import records
from hohere.records import RecordError, chunk_id, read_record, validate, write_record


def test_round_trip_preserves_every_array():
    rec = record()
    raw = write_record(**rec)
    back = read_record(raw)
    for key in records.KEYS:
        assert back[key].dtype == np.dtype(records.DTYPES[key])
        assert np.array_equal(back[key], rec[key])


def test_chunk_id_is_zero_padded():
    assert chunk_id("small-1-a-b", 3) == "small-1-a-b-003"
    assert chunk_id("small-1-a-b", 1234) == "small-1-a-b-1234"
    with pytest.raises(ValueError):
        chunk_id("x", -1)


def test_an_empty_chunk_is_valid():
    rec = record(frames=0, unit_starts=(), sil_starts=())
    raw = write_record(**rec)
    back = read_record(raw)
    assert back["codes"].shape == (32, 0)
    assert back["unit_start"].shape == (0,)


def test_validator_rejects_wrong_dtypes_and_shapes():
    rec = record()
    rec["unit_start"] = rec["unit_start"].astype(np.int64)
    with pytest.raises(RecordError, match="int32"):
        validate(rec)

    rec = record()
    rec["codes"] = rec["codes"][:16]
    with pytest.raises(RecordError, match=r"\[32, T\]"):
        validate(rec)

    rec = record()
    rec["unit_end"] = rec["unit_end"][:1]
    with pytest.raises(RecordError, match="expected"):
        validate(rec)


def test_validator_rejects_unsorted_or_out_of_range_frames():
    rec = record(unit_starts=(8, 4))
    with pytest.raises(RecordError, match="sorted"):
        validate(rec)

    rec = record(unit_starts=(5,), sil_starts=())
    rec["unit_end"] = np.asarray([5], dtype=np.int32)
    with pytest.raises(RecordError, match="unit_start must be < unit_end"):
        validate(rec)

    rec = record()
    rec["unit_k4096"] = np.asarray([4096], dtype=np.int16)
    with pytest.raises(RecordError, match="unit_k4096"):
        validate(rec)

    rec = record(sil_starts=(12, 4))
    with pytest.raises(RecordError, match="sil_start"):
        validate(rec)


def test_read_record_reports_missing_keys():
    import io

    buffer = io.BytesIO()
    np.savez(buffer, codes=np.zeros((32, 4), dtype=np.int16))
    with pytest.raises(RecordError, match="missing keys"):
        read_record(buffer.getvalue())


def test_write_record_converts_input_dtypes_instead_of_failing():
    rec = record()
    raw = write_record(**(rec | {"codes": rec["codes"].astype(np.int32)}))
    assert read_record(raw)["codes"].dtype == np.int16
