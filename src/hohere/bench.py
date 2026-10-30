"""`--bench-minutes`: measure the real pipeline and estimate a full run.

The bench runs the *actual* pipeline (same threads, same stages, same shards) against
a throwaway copy of the state DB, with no upload and no `done` marks, so it can be
run on a machine that is still producing the real dataset. What it measures is what a
run costs; what it estimates is how long 100/1000/5000 hours and the full plan will
take on this machine.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

from .pipeline import Runner, Stages
from .state import State
from .status import Progress

log = logging.getLogger(__name__)

STAGE_KEYS = ("download", "decode", "vad", "wavlm", "assign", "mimi", "write", "upload")


def device_info() -> dict[str, Any]:
    """GPU name or CPU model, core count and RAM (all optional, never fatal)."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cores": os.cpu_count(),
        "ram_gb": None,
        "gpu": None,
    }
    try:  # macOS / Linux CPU model
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:  # pragma: no cover - defensive
        pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except Exception:  # torch is optional for the bench to exist at all
        info["torch"] = None
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        info["ram_gb"] = round(total / 1e9, 1)
    except (ValueError, OSError, AttributeError):  # pragma: no cover - platform specific
        pass
    return info


def _rate(stage: dict[str, float], audio_seconds: float, wall_seconds: float) -> float:
    """Audio seconds processed per second of this stage (x-realtime)."""
    total = float(stage.get("total_s") or 0.0)
    if total > 0:
        return audio_seconds / total
    return audio_seconds / wall_seconds if wall_seconds > 0 else 0.0


def build_payload(
    *,
    stats: Any,
    progress: Progress,
    device: dict[str, Any],
    dtype: str,
    plan_hours: float,
    hours: tuple[float, ...] = (100.0, 1000.0, 5000.0),
    target_bytes: int | None = None,
) -> dict[str, Any]:
    """The `work/bench.json` document."""
    wall = float(stats.wall_seconds) or 1e-9
    audio = float(stats.audio_seconds)
    stages = {
        key: {
            "n": int(entry[0]),
            "total_s": round(float(entry[1]), 3),
            "mean_s": round(float(entry[1]) / entry[0], 4) if entry[0] else 0.0,
            "x_realtime": round(_rate({"total_s": entry[1]}, audio, wall), 3),
        }
        for key, entry in progress.stage_totals.items()
    }
    stage_seconds = sum(entry["total_s"] for key, entry in stages.items() if key != "upload")
    overall = audio / wall
    queue_share = max(0.0, 1.0 - stage_seconds / wall) if wall > 0 else 0.0
    estimates = {
        str(int(limit)): round(limit * 3600.0 / overall, 1) if overall > 0 else None
        for limit in hours
    }
    estimates["full"] = round(plan_hours * 3600.0 / overall, 1) if overall > 0 else None
    agreement = None
    if stats.agreements:
        agreement = round(sum(stats.agreements) / len(stats.agreements), 4)
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "dtype": dtype,
        "audio_seconds": round(audio, 2),
        "wall_seconds": round(wall, 2),
        "chunks": int(stats.chunks),
        "files": int(stats.files),
        "errors": int(stats.errors),
        "x_realtime": round(overall, 3),
        "stages": stages,
        "queue_wait_share": round(queue_share, 3),
        "boundary_agreement": agreement,
        "plan_hours": round(plan_hours, 2),
        "estimate_wall_seconds": estimates,
    }


def run_bench(
    *,
    state: State,
    stages: Stages,
    files: list,
    minutes: float,
    out_dir: str | os.PathLike[str] = "work/bench-shards",
    dtype: str = "float32",
    tmp_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Run the pipeline for `minutes` of chunk audio on a copy of the state DB."""
    work = Path(tmp_dir) if tmp_dir is not None else Path(out_dir).parent
    work.mkdir(parents=True, exist_ok=True)
    copy = work / "bench-state.db"
    shutil.copyfile(state.path, copy)
    progress = Progress(device="bench", target=minutes / 60.0)
    with State(copy, out_dir=out_dir) as bench_state:
        bench_state.set_meta("bench", True)
        runner = Runner(
            state=bench_state,
            stages=stages,
            out_dir=out_dir,
            progress=progress,
            uploader=None,
            mark_done=False,
            audio_limit_seconds=float(minutes) * 60.0,
        )
        stats = runner.run(files)
        plan_hours = float(bench_state.get_meta("plan_hours", 0.0) or 0.0)
    payload = build_payload(
        stats=stats,
        progress=progress,
        device=device_info(),
        dtype=dtype,
        plan_hours=plan_hours,
    )
    path = Path(out_dir).parent / "bench.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("bench written to %s", path)
    return payload
