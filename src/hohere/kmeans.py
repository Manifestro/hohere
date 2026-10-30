"""Spherical k-means in torch: no faiss, no numpy matmul of a 4 M x 1024 matrix.

`fit` L2-normalises the points and clusters them by cosine similarity: k-means++
seeding on a seeded random subsample, then Lloyd iterations where the assignment is
an argmax over dot products (done in chunks so the distance matrix never has to fit
in memory) and the centroid is the normalised mean of its members. An empty cluster
is re-seeded with a random point, which is what keeps a 8192-way codebook from
collapsing on a tier with fewer distinct syllables than k.

The objective (mean cosine similarity of a point to its centroid, so higher is
better) is logged per iteration and stored next to the codebook: a run whose curve
is still climbing at the last iteration is a run whose `--iters` was too small.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

INIT_SAMPLE = 262144
ITERS = 100
ASSIGN_CHUNK = 16384  # rows per assignment block; bounds peak memory on the GPU too
SEED = 0


@dataclass
class FitResult:
    """A codebook and everything a later run needs to trust it."""

    centroids: Any  # torch.Tensor [K, D], unit norm
    objective: list[float] = field(default_factory=list)
    k: int = 0
    n_points: int = 0
    seed: int = 0
    empty_reseeds: int = 0

    def to_json(self, **extra: Any) -> dict[str, Any]:
        return {
            "k": int(self.k),
            "n_points": int(self.n_points),
            "dim": int(self.centroids.shape[1]),
            "iters": len(self.objective),
            "objective": [round(float(value), 6) for value in self.objective],
            "seed": int(self.seed),
            "empty_reseeds": int(self.empty_reseeds),
            **extra,
        }


def _normalize(x):
    norms = x.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return x / norms


def init_centroids(points, k: int, *, seed: int = SEED, sample: int = INIT_SAMPLE):
    """k-means++ on a seeded subsample, with cosine distance (unit-norm points)."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = int(points.shape[0])
    if total > sample:
        index = torch.randperm(total, generator=generator)[:sample].to(points.device)
        sub = points[index]
    else:
        sub = points
    n = int(sub.shape[0])
    if k > n:
        raise ValueError(f"k={k} but only {n} points to seed from")
    first = int(torch.randint(n, (1,), generator=generator).item())
    centroids = [sub[first]]
    closest = (1.0 - sub @ centroids[0]).clamp_min(0.0)  # cosine distance
    for _ in range(1, k):
        weights = closest / closest.sum().clamp_min(1e-12)
        pick = int(torch.multinomial(weights.cpu(), 1, generator=generator).item())
        centroids.append(sub[pick])
        distance = (1.0 - sub @ sub[pick]).clamp_min(0.0)
        closest = torch.minimum(closest, distance)
    return torch.stack(centroids)


def assign_chunked(points, centroids, chunk: int = ASSIGN_CHUNK):
    """`argmax(points @ centroids.T)` per block, plus the best similarity."""
    import torch

    best_ids: list[Any] = []
    best_scores: list[Any] = []
    for start in range(0, int(points.shape[0]), chunk):
        block = points[start : start + chunk]
        scores = block @ centroids.T
        values, index = scores.max(dim=1)
        best_ids.append(index)
        best_scores.append(values)
    if not best_ids:
        empty = torch.zeros(0, dtype=torch.long, device=points.device)
        return empty, empty.float()
    return torch.cat(best_ids), torch.cat(best_scores)


def fit(
    emb,
    k: int,
    *,
    iters: int = ITERS,
    init_sample: int = INIT_SAMPLE,
    seed: int = SEED,
    device: str = "cpu",
    log_every: int = 10,
):
    """Spherical k-means over `emb` [N, D]; returns a `FitResult`."""
    import torch

    if k <= 0:
        raise ValueError("k must be positive")
    points = torch.as_tensor(emb, dtype=torch.float32, device=device)
    if points.ndim != 2:
        raise ValueError(f"emb must be [N, D], got {tuple(points.shape)}")
    if points.shape[0] < k:
        raise ValueError(f"k={k} but only {points.shape[0]} points")
    points = _normalize(points)
    centroids = init_centroids(points, k, seed=seed, sample=init_sample)
    centroids = _normalize(centroids)
    result = FitResult(centroids=centroids, k=k, n_points=int(points.shape[0]), seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    for iteration in range(iters):
        ids, scores = assign_chunked(points, centroids)
        objective = float(scores.mean().item())
        result.objective.append(objective)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(k, dtype=torch.float32, device=centroids.device)
        sums.index_add_(0, ids, points)
        counts.index_add_(0, ids, torch.ones_like(scores))
        empty = torch.flatnonzero(counts == 0)
        sums[empty] = points[torch.randint(points.shape[0], (int(empty.numel()),),
                                           generator=generator).to(points.device)]
        counts[empty] = 1.0
        result.empty_reseeds += int(empty.numel() > 0)
        centroids = _normalize(sums / counts.unsqueeze(1))
        if log_every and iteration % log_every == 0:
            log.info("k-means k=%d iter %d objective %.6f", k, iteration, objective)
    result.centroids = centroids
    if result.objective:
        log.info("k-means k=%d done, objective %.6f", k, result.objective[-1])
    return result


def assign(emb, centroids, *, chunk: int = ASSIGN_CHUNK, device: str = "cpu") -> np.ndarray:
    """Nearest centroid per row, as `int16` ids (the record format's dtype)."""
    import torch

    points = torch.as_tensor(emb, dtype=torch.float32, device=device)
    if points.ndim != 2:
        raise ValueError(f"emb must be [N, D], got {tuple(points.shape)}")
    table = torch.as_tensor(centroids, dtype=torch.float32, device=device)
    table = _normalize(table)
    ids, _scores = assign_chunked(_normalize(points), table, chunk=chunk)
    return ids.to(torch.int16).cpu().numpy()


def save(
    result: FitResult,
    out_dir: str | os.PathLike[str],
    *,
    hours: float | None = None,
    **extra: Any,
) -> tuple[Path, Path]:
    """Write `k{K}.npy` (float32 [K, D]) and `k{K}.json` (the metadata)."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    npy = directory / f"k{result.k}.npy"
    meta = directory / f"k{result.k}.json"
    array = result.centroids.detach().to("cpu", dtype_key()).numpy()
    np.save(npy, array)
    payload = result.to_json(hours=hours, **extra)
    meta.write_text(json.dumps(payload, indent=2) + "\n")
    return npy, meta


def dtype_key():
    import torch

    return torch.float32


def load(out_dir: str | os.PathLike[str], k: int) -> np.ndarray:
    """The codebook `k{K}.npy`, checked to be [K, D] and unit norm."""
    path = Path(out_dir) / f"k{k}.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `hohere fit-units` first")
    table = np.load(path)
    if table.ndim != 2 or table.shape[0] != k:
        raise ValueError(f"{path} is {table.shape}, expected [{k}, D]")
    return table.astype(np.float32)
