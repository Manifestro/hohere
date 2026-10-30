#!/usr/bin/env python3
"""Batched WavLM must match one chunk at a time: same boundaries, same embeddings.

Takes the chunks of the first files of the plan and compares `Encoder.encode_batch` with
`Encoder.encode`. Needs the models; not part of pytest. Usage: check_batch.py STATE [N] [DTYPE]
"""

from __future__ import annotations

import sys

import numpy as np

from hohere import chunk, pipeline, vad, zerosyl
from hohere.state import State


def main(state_path: str, count: int = 16, dtype: str = "float32") -> int:
    from hohere.remote import RangeReader

    encoder = zerosyl.load_encoder(device="cuda", dtype=dtype)
    wavs: list[np.ndarray] = []
    import io

    import soundfile

    with State(state_path) as state:
        for file in state.ordered_files():
            raw = RangeReader(file.archive_url).read(file.flac_offset, file.flac_size)
            wav = soundfile.read(io.BytesIO(raw), dtype="float32")[0]
            for start, end in chunk.plan(file.vad, wav.size / 16000).chunks:
                wavs.append(pipeline._slice(wav, start, end))
            if len(wavs) >= count:
                break
    wavs = sorted(wavs[:count], key=lambda w: w.size)
    spans = [vad.speech(w) for w in wavs]
    batched = encoder.encode_batch(wavs, spans)
    bad = 0
    worst = 0.0
    for wav, s, got in zip(wavs, spans, batched, strict=True):
        want = encoder.encode(wav, s)
        same = want.boundaries == got.boundaries
        bad += not same
        if same and len(want):
            worst = max(worst, float(np.abs(want.embeddings - got.embeddings).max()))
        print(wav.size / 16000, "boundaries", len(want.boundaries), "same" if same else "DIFFER",
              f"max |diff| {worst:.2e}")
    print(f"chunks {len(wavs)}, differing boundaries {bad}, worst embedding diff {worst:.2e}")
    return 1 if bad else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args[0], int(args[1]) if len(args) > 1 else 16, *args[2:3]))
