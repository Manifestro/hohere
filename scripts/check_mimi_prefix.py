#!/usr/bin/env python3
"""Are the first 8 of our 32 Mimi codebooks the 8 of the earlier encodings?

Takes chunks of the first files of the plan and compares the first 8 codebooks of a run
(`[32, T]`, batched) with:

  same_batch  8 codebooks, the very same input tensor: the residual quantizer alone
  one_by_one  8 codebooks, each chunk alone: what batching and padding change
  v1_ffmpeg   8 codebooks, audio resampled by ffmpeg as hohere v1 did: the older tokens

`same_batch` must match exactly, or the exit code is 1. The other two are reported, since
their input differs. Needs the models; not part of pytest.
Usage: check_mimi_prefix.py STATE [N]
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np

from hohere import chunk, mimi, pipeline
from hohere.state import State

PREFIX = 8
BATCH = 8
FFMPEG = ("ffmpeg", "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-ac", "1",
          "-ar", str(mimi.SAMPLE_RATE), "-f", "f32le", "pipe:1")


def load_chunks(state_path: str, count: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """`(16 kHz chunk, the same stretch at 24 kHz from ffmpeg)`, in the order of the plan."""
    import soundfile

    from hohere.remote import RangeReader

    out = []
    with State(state_path) as state:
        for file in state.select_until("full"):
            raw = RangeReader(file.archive_url).read(file.flac_offset, file.flac_size)
            wav = soundfile.read(io.BytesIO(raw), dtype="float32")[0]
            resampled = np.frombuffer(
                subprocess.run(FFMPEG, input=raw, capture_output=True, check=True).stdout,
                np.float32,
            )
            for start, end in chunk.chunks(file.vad, wav.size / 16000):
                piece = resampled[round(start * mimi.SAMPLE_RATE) : round(end * mimi.SAMPLE_RATE)]
                out.append((pipeline._slice(wav, start, end), piece))
            if len(out) >= count:
                return out[:count]
    return out


def encode8(encoder: mimi.MimiEncoder, tensor) -> np.ndarray:
    import torch

    with torch.inference_mode():
        return encoder.model.encode(tensor).cpu().numpy()


def agreement(reference: list[np.ndarray], other: list[np.ndarray]) -> dict:
    """Share of equal codes over the common frames, for codebook 0 and the first 8."""
    first = equal = zero = frames = 0
    for a, b in zip(reference, other, strict=True):
        width = min(a.shape[1], b.shape[1])
        same = a[:PREFIX, :width] == b[:PREFIX, :width]
        equal += int(same.sum())
        zero += int(same[0].sum())
        first += same.size
        frames += width
    return {"codebook_0": zero / frames, "first_8": equal / first, "frames": frames}


def main(state_path: str, count: int = 20) -> int:
    import torch

    encoder = mimi.load_encoder(device="cuda")
    pairs = load_chunks(state_path, count)
    pairs.sort(key=lambda pair: pair[0].size)
    wavs = [pair[0] for pair in pairs]

    full: list[np.ndarray] = []
    same_batch: list[np.ndarray] = []
    for start in range(0, len(wavs), BATCH):
        group = wavs[start : start + BATCH]
        full += encoder.encode_batch(group)
        tensor, lengths = encoder.batch_tensor(group)
        encoder.model.set_num_codebooks(PREFIX)
        codes = encode8(encoder, tensor)
        encoder.model.set_num_codebooks(mimi.CODEBOOKS)
        same_batch += [codes[row, :, :n] for row, n in enumerate(lengths)]

    encoder.model.set_num_codebooks(PREFIX)
    alone = []
    for wav in wavs:
        tensor, lengths = encoder.batch_tensor([wav])
        alone.append(encode8(encoder, tensor)[0, :, : lengths[0]])
    from_ffmpeg = []
    for _wav, piece in pairs:
        n = mimi.frames_for(piece.size)
        padded = np.zeros(n * mimi.HOP, dtype=np.float32)
        padded[: piece.size] = piece
        tensor = torch.from_numpy(padded).to(encoder.device).reshape(1, 1, -1)
        from_ffmpeg.append(encode8(encoder, tensor)[0])

    report = {
        "chunks": len(wavs),
        "same_batch": agreement(full, same_batch),
        "one_by_one": agreement(full, alone),
        "v1_ffmpeg": agreement(full, from_ffmpeg),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["same_batch"]["first_8"] == 1.0 else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args[0], int(args[1]) if len(args) > 1 else 20))
