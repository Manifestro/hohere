"""Syllable units from a frozen WavLM-Large, following ZeroSyl exactly.

The reference (`zerosyl.ZeroSylContinuous.encode`) detects boundaries on the frame
norms of one layer and mean-pools another layer between them. We reproduce that
bit for bit and then make one change of our own: non-speech frames never reach the
k-means. Inside every detected segment only frames whose centre falls in a Silero
speech span are kept; a segment with no speech frame is dropped; the embedding is
the mean over the kept frames, and the unit's extent is the first and last kept
frame (the silent part of a "syllable" is not part of the unit).

`segment_features` is the whole of that post-processing and is a pure function of
two feature matrices, so it is tested without torch. `Encoder` owns the model and
the only real compute path.
"""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import vad

log = logging.getLogger(__name__)

CHECKPOINT_URL = "https://storage.googleapis.com/zerospeech-checkpoints/WavLM-Large.pt"
CHECKPOINT_NAME = "WavLM-Large.pt"
BOUNDARY_LAYER = 13
MEANPOOL_LAYER = 22
WINDOW = 3
PROMINENCE = 0.45
WIN = 400
HOP = 320
PAD = (WIN - HOP) // 2  # 40 samples either side, as the reference does
DTYPES = ("float32", "bfloat16", "float16")


@dataclass
class Segments:
    """Units of one chunk: 50 Hz frame extents and their embeddings."""

    unit_start: np.ndarray  # int32 [N], first speech frame
    unit_end: np.ndarray  # int32 [N], last speech frame + 1
    embeddings: np.ndarray  # float32 [N, D]
    boundaries: list[int]  # raw boundary frames, before the speech filter
    dropped: int = 0  # segments that had no speech frame at all

    def __len__(self) -> int:
        return int(self.unit_start.shape[0])


def layer_norm(wav: np.ndarray) -> np.ndarray:
    """`torch.nn.functional.layer_norm(wav, wav.shape)`, including its eps of 1e-5."""
    audio = np.asarray(wav, dtype=np.float32).reshape(-1)
    return (audio - audio.mean()) / np.sqrt(audio.var() + 1e-5)


def frame_norms(features: np.ndarray) -> np.ndarray:
    """L2 norm per frame, z-scored over the chunk: `ZeroSyl`'s boundary signal."""
    norms = np.linalg.norm(np.asarray(features, dtype=np.float64), axis=-1)
    std = float(norms.std())
    if std == 0.0:
        return norms - float(norms.mean())
    return (norms - float(norms.mean())) / std


def smooth(norms: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Moving average with edge padding (`np.convolve(..., mode="valid")`)."""
    if window <= 1 or norms.size == 0:
        return np.asarray(norms, dtype=np.float64)
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(norms, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def find_peaks(x: np.ndarray, prominence: float) -> np.ndarray:
    """`scipy.signal.find_peaks(x, prominence=...)`, with a numpy fallback.

    scipy is the reference implementation and is used whenever it is installed; the
    fallback reproduces the same rule (a strict local maximum, prominence measured
    against the higher of the two bases that delimit its basin) so that the segment
    post-processing can be tested on a machine without the models.
    """
    values = np.asarray(x, dtype=np.float64)
    try:
        from scipy.signal import find_peaks as scipy_find_peaks

        peaks, _ = scipy_find_peaks(values, prominence=prominence)
        return np.asarray(peaks, dtype=np.int64)
    except ImportError:
        pass
    if values.size < 3:
        return np.zeros(0, dtype=np.int64)
    candidates = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    ) + 1
    kept: list[int] = []
    for peak in candidates:
        height = values[peak]
        left = peak - 1
        while left > 0 and values[left] <= height:
            left -= 1
        right = peak + 1
        while right < values.size - 1 and values[right] <= height:
            right += 1
        base = max(values[left : peak + 1].min(), values[peak : right + 1].min())
        if height - base >= prominence:
            kept.append(int(peak))
    return np.asarray(kept, dtype=np.int64)


def boundaries_from_features(
    features: np.ndarray, *, window: int = WINDOW, prominence: float = PROMINENCE
) -> list[int]:
    """`[0] + peaks + [F]` from the layer-13 features of one chunk."""
    frames = int(np.asarray(features).shape[0])
    if frames == 0:
        return [0]
    smoothed = smooth(frame_norms(features), window)
    peaks = find_peaks(smoothed, prominence)
    return [0, *[int(p) for p in peaks], frames]


def segment_features(
    boundary_features: np.ndarray,
    semantic_features: np.ndarray,
    speech: np.ndarray,
    *,
    window: int = WINDOW,
    prominence: float = PROMINENCE,
) -> Segments:
    """Boundaries -> speech-only segments -> (unit extents, embeddings).

    `boundary_features` is layer 13, `semantic_features` layer 22, `speech` the
    `speech_mask50` of the chunk. Segments are half-open `[start, end)`.
    """
    boundary_features = np.asarray(boundary_features)
    semantic_features = np.asarray(semantic_features)
    speech = np.asarray(speech, dtype=bool)
    frames = int(boundary_features.shape[0])
    if semantic_features.shape[0] != frames or speech.shape[0] != frames:
        raise ValueError(
            f"feature frames disagree: boundary {boundary_features.shape[0]}, "
            f"semantic {semantic_features.shape[0]}, mask {speech.shape[0]}"
        )
    bounds = boundaries_from_features(
        boundary_features, window=window, prominence=prominence
    )
    starts: list[int] = []
    ends: list[int] = []
    vectors: list[np.ndarray] = []
    dropped = 0
    for start, end in itertools.pairwise(bounds):
        segment = speech[start:end]
        kept = np.flatnonzero(segment)
        if kept.size == 0:
            dropped += 1
            continue
        first, last = int(kept[0]), int(kept[-1])
        starts.append(start + first)
        ends.append(start + last + 1)
        vectors.append(np.asarray(semantic_features[start:end])[segment].mean(axis=0))
    dim = int(semantic_features.shape[-1]) if semantic_features.ndim == 2 else 0
    embeddings = (
        np.stack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, dim), dtype=np.float32)
    )
    return Segments(
        unit_start=np.asarray(starts, dtype=np.int32),
        unit_end=np.asarray(ends, dtype=np.int32),
        embeddings=embeddings,
        boundaries=bounds,
        dropped=dropped,
    )


def checkpoint_path(models_dir: str | os.PathLike[str] = "work/models") -> Path:
    """The local checkpoint, downloaded once (the file is ~1.2 GB, so cached)."""
    target = Path(models_dir) / CHECKPOINT_NAME
    if target.exists():
        return target
    import shutil
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    log.info("downloading %s to %s", CHECKPOINT_URL, target)
    with urllib.request.urlopen(CHECKPOINT_URL) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1 << 22)
    tmp.replace(target)
    return target


@dataclass
class Encoder:
    """Frozen WavLM-Large; one instance owns the device for the whole run."""

    model: Any
    device: Any
    dtype: str = "float32"
    sample_rate: int = vad.SAMPLE_RATE

    @property
    def frames_per_second(self) -> float:
        return 50.0

    def n_frames(self, n_samples: int) -> int:
        """WavLM frames of a chunk (the reference divides the padded length by 320)."""
        return int((int(n_samples) + 2 * PAD) // 320)

    def forward(self, wav16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Layer 13 and layer 22 features of one chunk, as numpy float32."""
        import torch

        audio = np.asarray(wav16, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return np.zeros((0, 1024), np.float32), np.zeros((0, 1024), np.float32)
        if self.model.cfg.normalize:
            audio = layer_norm(audio)
        padded = np.pad(audio, (PAD, PAD))
        tensor = torch.from_numpy(padded).to(self.device).unsqueeze(0)
        autocast = (
            torch.autocast(device_type="cuda", dtype=_torch_dtype(self.dtype))
            if self.dtype != "float32" and torch.cuda.is_available()
            else None
        )
        with torch.inference_mode():
            if autocast is not None:
                with autocast:
                    first, second = self.model.extract_dual_features(
                        tensor,
                        output_layer_1=BOUNDARY_LAYER,
                        output_layer_2=MEANPOOL_LAYER,
                    )
            else:
                first, second = self.model.extract_dual_features(
                    tensor,
                    output_layer_1=BOUNDARY_LAYER,
                    output_layer_2=MEANPOOL_LAYER,
                )
        return (
            first.squeeze(0).float().cpu().numpy(),
            second.squeeze(0).float().cpu().numpy(),
        )

    def forward_batch(self, wavs: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """`forward` for chunks of similar length, padded at the end of the batch.

        Every chunk is normalised alone. The mask covers each chunk's samples from its
        last real frame on, so frames past it are zeroed and never attended to, and a
        chunk's features are the ones it would get on its own.
        """
        import torch

        padded = [
            np.pad(layer_norm(w) if self.model.cfg.normalize else np.asarray(w, np.float32),
                   (PAD, PAD))
            for w in wavs
        ]
        width = max(p.size for p in padded)
        batch = np.zeros((len(padded), width), dtype=np.float32)
        mask = np.zeros((len(padded), width), dtype=bool)
        frames = [_frames_of(p.size) for p in padded]
        for row, (piece, n) in enumerate(zip(padded, frames, strict=True)):
            batch[row, : piece.size] = piece
            mask[row, n * HOP :] = True
        tensor = torch.from_numpy(batch).to(self.device)
        padding = torch.from_numpy(mask).to(self.device)
        autocast = torch.autocast(
            device_type="cuda",
            dtype=_torch_dtype(self.dtype),
            enabled=self.dtype != "float32" and torch.cuda.is_available(),
        )
        with torch.inference_mode(), autocast:
            first, second = self.model.extract_dual_features(
                tensor,
                padding_mask=padding,
                output_layer_1=BOUNDARY_LAYER,
                output_layer_2=MEANPOOL_LAYER,
            )
        first = first.float().cpu().numpy()
        second = second.float().cpu().numpy()
        return [(first[i, :n], second[i, :n]) for i, n in enumerate(frames)]

    def encode_batch(
        self, wavs: list[np.ndarray], speech_spans: list[list[vad.Span]]
    ) -> list[Segments]:
        """`encode` for several chunks at once."""
        return [
            segment_features(first, second, vad.speech_mask50(spans, first.shape[0]))
            for (first, second), spans in zip(self.forward_batch(wavs), speech_spans, strict=True)
        ]

    def encode(self, wav16: np.ndarray, speech_spans: list[vad.Span] | None = None) -> Segments:
        """Units of one chunk; a `None` mask means "every frame is speech"."""
        first, second = self.forward(wav16)
        frames = int(first.shape[0])
        mask = (
            np.ones(frames, dtype=bool)
            if speech_spans is None
            else vad.speech_mask50(speech_spans, frames)
        )
        return segment_features(first, second, mask)


def _frames_of(n_samples: int) -> int:
    """Frames the convolutional front end makes of `n_samples` (25 ms window, 20 ms hop)."""
    return (n_samples - WIN) // HOP + 1


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_encoder(
    device: str = "cuda", dtype: str = "float32", models_dir: str = "work/models"
) -> Encoder:
    """Load the frozen WavLM-Large checkpoint onto `device`."""
    import torch

    from .vendor.wavlm.WavLM import WavLM, WavLMConfig

    if dtype not in DTYPES:
        raise ValueError(f"unknown wavlm dtype {dtype!r}, expected one of {DTYPES}")
    path = checkpoint_path(models_dir)
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model = model.to(device)
    torch.backends.cuda.matmul.allow_tf32 = False  # the reference runs in fp32
    torch.backends.cudnn.allow_tf32 = False
    return Encoder(model=model, device=device, dtype=dtype)
