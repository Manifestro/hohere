"""Mimi codes: 32 codebooks at 12.5 Hz for a 16 kHz chunk.

The audio is resampled to Mimi's 24 kHz, batched (batches are padded at the end to the
longest chunk and cut back to `T = ceil(n_24k / 1920)` frames), and the encoder is
asked for 32 codebooks explicitly. Codes are stored `int16`, `[32, T]`.

Moshi and torch are imported inside the functions: Part A installs neither.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SOURCE_RATE = 16000
HOP = 1920  # samples per 12.5 Hz frame at 24 kHz
CODEBOOKS = 32
MODEL_REPO = "kyutai/moshiko-pytorch-bf16"
MODEL_FILE = "tokenizer-e351c8d8-checkpoint125.safetensors"
MODEL_NAME = "mimi"
BATCH = 16


def frames_for(n_samples_24k: int) -> int:
    """`T = ceil(n_24k / 1920)` frames of a chunk."""
    return math.ceil(n_samples_24k / HOP) if n_samples_24k > 0 else 0


def checkpoint_path(models_dir: str | os.PathLike[str] = "work/models") -> str:
    """The tokenizer weights, fetched once into `work/models` (a local HF cache)."""
    from huggingface_hub import hf_hub_download

    target = Path(models_dir) / MODEL_FILE
    if target.exists():
        return str(target)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(models_dir)
    )


@dataclass
class MimiEncoder:
    """The loaded Mimi model plus the device/dtype it runs on."""

    model: Any
    device: Any
    sample_rate: int = SAMPLE_RATE

    def resample(self, wav16: np.ndarray) -> np.ndarray:
        import torch
        import torchaudio

        audio = torch.from_numpy(np.asarray(wav16, dtype=np.float32).reshape(1, -1))
        if int(audio.shape[1]) == 0:
            return np.zeros(0, dtype=np.float32)
        return (
            torchaudio.functional.resample(audio, SOURCE_RATE, SAMPLE_RATE)
            .squeeze(0)
            .numpy()
        )

    def encode_batch(self, chunks16: list[np.ndarray]) -> list[np.ndarray]:
        """Encode 16 kHz chunks; a batch too big for the device is split in two."""
        import torch

        if not chunks16:
            return []
        try:
            return self._encode_batch(chunks16)
        except torch.cuda.OutOfMemoryError:
            if len(chunks16) == 1:
                raise
        torch.cuda.empty_cache()
        middle = len(chunks16) // 2
        return [*self.encode_batch(chunks16[:middle]), *self.encode_batch(chunks16[middle:])]

    def batch_tensor(self, chunks16: list[np.ndarray]) -> tuple[Any, list[int]]:
        """The input of `model.encode` `[B, 1, samples]`, and the frames of each chunk."""
        import torch
        import torchaudio

        lengths = [frames_for(math.ceil(chunk.size * SAMPLE_RATE / SOURCE_RATE))
                   for chunk in chunks16]
        batch = np.zeros((len(chunks16), max(chunk.size for chunk in chunks16)), dtype=np.float32)
        for row, chunk in enumerate(chunks16):
            batch[row, : chunk.size] = chunk
        tensor = torch.from_numpy(batch).to(self.device)
        tensor = torchaudio.functional.resample(tensor, SOURCE_RATE, SAMPLE_RATE)
        width = max(length * HOP for length in lengths)  # whole frames for the longest chunk
        tensor = torch.nn.functional.pad(tensor, (0, max(0, width - tensor.shape[1])))
        return tensor[:, :width].unsqueeze(1), lengths

    def _encode_batch(self, chunks16: list[np.ndarray]) -> list[np.ndarray]:
        import torch

        tensor, lengths = self.batch_tensor(chunks16)
        with torch.inference_mode():
            codes = self.model.encode(tensor)
        if codes.dim() != 3 or codes.shape[1] != CODEBOOKS:
            raise RuntimeError(f"unexpected Mimi codes shape {tuple(codes.shape)}")
        out: list[np.ndarray] = []
        for row, frames in enumerate(lengths):
            piece = codes[row, :, :frames].to(torch.int16).cpu().numpy()
            out.append(np.ascontiguousarray(piece))
        return out


def load_encoder(path: str | None = None, device: str = "cuda", *, batch: int = BATCH,
                 models_dir: str = "work/models") -> MimiEncoder:
    """Load Mimi once; call it from the single model worker only."""
    import torch
    from moshi.models import loaders

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resolved = path or checkpoint_path(models_dir)
    model = loaders.get_mimi(resolved, device=device)
    model.set_num_codebooks(CODEBOOKS)
    model.eval()
    log.info("Mimi loaded from %s on %s (batch %d)", resolved, device, batch)
    return MimiEncoder(model=model, device=device)
