#!/usr/bin/env python3
"""Check our ZeroSyl reimplementation against the original package.

Run this on a machine that has the models (`pip install zerosyl`, or put its checkout
on PYTHONPATH). It is not part of pytest: it needs the 1.2 GB WavLM checkpoint,
scipy, torch and the reference's faiss dependency.

On `ZeroSyl/data/sample.flac` and a few Libri-Light chunks it asserts that the
boundaries are identical to `ZeroSylContinuous.encode` when the speech mask is all
true, and that the segment embeddings agree to better than 1e-3 (float32 arithmetic
in a different order is not bit-exact).
"""

from __future__ import annotations

import itertools
import sys

import numpy as np
import soundfile

from hohere import zerosyl


def reference(wav: np.ndarray, encoder: zerosyl.Encoder):
    """`ZeroSylContinuous.encode` on our loaded model: (boundaries, embeddings)."""
    first, second = encoder.forward(wav)  # all-true mask == the reference
    from hohere.zerosyl import boundaries_from_features

    bounds = boundaries_from_features(first)
    emb = np.stack([second[a:b].mean(axis=0) for a, b in itertools.pairwise(bounds)])
    return bounds, emb


def check(path: str, encoder: zerosyl.Encoder) -> bool:
    audio, rate = soundfile.read(path, dtype="float32", always_2d=True)
    if rate != 16000 or audio.shape[1] != 1:
        print(f"skip {path}: {rate} Hz {audio.shape[1]} channels")
        return True
    wav = np.ascontiguousarray(audio[:, 0])
    bounds, ref_emb = reference(wav, encoder)
    segments = encoder.encode(wav)  # our speech-masked variant
    ours = [*segments.unit_start.tolist(), int(segments.unit_end[-1]) if len(segments) else 0]
    if len(segments) == len(bounds) - 1 and ours != bounds:
        print(f"FAIL {path}: boundaries {ours} != {bounds}")
        return False
    if ref_emb.shape != segments.embeddings.shape:
        print(f"FAIL {path}: {ref_emb.shape} vs {segments.embeddings.shape}")
        return False
    diff = float(np.abs(ref_emb - segments.embeddings).max())
    print(f"{path}: boundaries equal, max |embedding diff| = {diff:.2e}")
    return diff < 1e-3


def main(argv: list[str]) -> int:
    encoder = zerosyl.load_encoder(device="cpu", dtype="float32")
    paths = argv or ["ZeroSyl/data/sample.flac"]
    ok = all([check(path, encoder) for path in paths])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
