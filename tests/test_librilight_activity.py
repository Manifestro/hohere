from __future__ import annotations

import numpy as np

from hohere.librilight.activity import (
    FRAME_SAMPLES,
    SILENCE_CODES,
    activity,
    has_pause,
    is_silence,
    num_frames,
    pause_runs,
)

TONE = np.full(FRAME_SAMPLES, 0.5, np.float32)
QUIET = np.full(FRAME_SAMPLES, 10 ** (-50 / 20), np.float32)


def act(total: int, silent: tuple[int, int] = (0, 0)) -> np.ndarray:
    """`total` active frames with `[silent)` frames switched off."""
    out = np.ones(total, np.uint8)
    out[silent[0] : silent[1]] = 0
    return out


def test_activity_is_the_rms_of_each_zero_padded_frame() -> None:
    assert activity(np.concatenate([TONE, QUIET])).tolist() == [1, 0]
    assert activity(np.zeros(FRAME_SAMPLES, np.float32)).tolist() == [0]
    assert activity(TONE[:100]).tolist() == [1]  # the rest of the frame is zeros
    assert (num_frames(0), num_frames(FRAME_SAMPLES), num_frames(FRAME_SAMPLES + 1)) == (0, 1, 2)


def test_pause_runs_are_half_open_frame_ranges() -> None:
    assert pause_runs(act(10, (3, 6))) == [(3, 6)]
    assert pause_runs(np.zeros(4, np.uint8)) == [(0, 4)]
    assert pause_runs(np.zeros(0, np.uint8)) == []


def test_pause_must_start_at_frame_38() -> None:
    assert has_pause(act(100, (38, 41)))
    assert not has_pause(act(100, (37, 41)))
    # T - 25 <= 38 for T <= 63, so a short segment can never hold a pause.
    for total in (40, 62, 63):
        assert not has_pause(act(total, (38, total - 25)))


def test_pause_must_end_at_frame_t_minus_25() -> None:
    assert has_pause(act(100, (50, 75)))
    assert not has_pause(act(100, (50, 76)))


def test_pause_needs_three_frames_and_silence_needs_every_code() -> None:
    assert not has_pause(act(100, (50, 52)))
    assert has_pause(act(100, (50, 53)))
    codes = np.tile(np.asarray(SILENCE_CODES, np.int16)[:, None], (1, 5))
    assert is_silence(codes)
    codes[3, 2] = 0
    assert not is_silence(codes)
