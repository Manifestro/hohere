"""The `hohere` command line.

`index` -> `plan` (chunk every file at index time) -> `fit-units` (the ZeroSyl
codebooks) -> `run` (stream -> chunk -> encode -> shard -> upload, with
`--status-port`/`--status-token` and `--bench-minutes`).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import typer

from . import bench as bench_mod
from . import (
    chunk as chunk_mod,
)
from . import (
    fitunits,
    kmeans,
    librilight,
    order,
    records,
    units,
)
from . import (
    pipeline as pipeline_mod,
)
from . import (
    shards as shards_mod,
)
from . import (
    status as status_mod,
)
from .librilight import SPLITS
from .state import State

app = typer.Typer(add_completion=False, no_args_is_help=True, help="hohere dataset pipeline")

DEFAULT_STATE = "work/state.db"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_splits(splits: str) -> list[str]:
    wanted = [part.strip() for part in splits.split(",") if part.strip()]
    unknown = [split for split in wanted if split not in SPLITS]
    if unknown:
        raise typer.BadParameter(f"unknown split(s) {unknown}; choose from {', '.join(SPLITS)}")
    if not wanted:
        raise typer.BadParameter("at least one split is required")
    return wanted


@app.command()
def index(
    splits: str = typer.Option("small,medium,large", "--splits", help="comma separated splits"),
    state_path: str = typer.Option(DEFAULT_STATE, "--state", help="SQLite state DB"),
    workers: int = typer.Option(64, "--workers", min=1, help="parallel range readers"),
    refresh: bool = typer.Option(False, "--refresh", help="rescan even if the DB is cached"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Index the Libri-Light archives into the state DB (cached across restarts)."""
    _setup_logging(verbose)
    wanted = _parse_splits(splits)
    started = time.time()
    with State(state_path) as state:
        files = librilight.index(wanted, workers=workers, state=state, use_cache=not refresh)
        state.assign_order()
        if state.get_meta("started_at") is None:
            state.set_meta("started_at", status_mod.iso(started))
        summary = state.summary()
    typer.echo(
        json.dumps(
            {
                "splits": wanted,
                "files": len(files),
                "source_hours": round(summary["source_hours"], 2),
                "tier_of_source_hours": _tier_csv(summary["source_hours"]),
                "seconds": round(time.time() - started, 1),
            },
            indent=2,
        )
    )


def _tier_csv(hours: float) -> str:
    from .order import tier_of

    return str(tier_of(hours))


@app.command()
def status(
    state_path: str = typer.Option(DEFAULT_STATE, "--state", help="SQLite state DB"),
    as_json: bool = typer.Option(True, "--json/--no-json", help="raw JSON (default) or a summary"),
) -> None:
    """Print the status payload of a state DB without starting a server."""
    with State(state_path) as state:
        progress = status_mod.Progress(
            device="cpu",
            target=state.get_meta("target_hours", "full"),
            state=None,  # durable numbers come from the DB, not from memory
        )
        payload = progress.snapshot(state)
    if not as_json:
        done, totals = payload["done"], payload["totals"]
        payload = {
            "files": f"{done['files']}/{totals['files']}",
            "source_hours": totals["source_hours"],
            "kept_hours": done["hours"],
            "tier": payload["tier"]["reached"],
            "eta_target_s": payload["eta_s"]["target"],
            "eta_full_s": payload["eta_s"]["full"],
        }
    typer.echo(json.dumps(payload, indent=2, sort_keys=False))


@app.command("m-stream")
def m_stream(
    npz: str = typer.Option(..., "--npz", help="one chunk .npz file"),
    variant: str = typer.Option("B", "--variant", help="A, B or B-sil"),
    k: int = typer.Option(4096, "--k", help="unit vocabulary: 4096 or 8192"),
    head: int = typer.Option(32, "--head", help="how many frames to print"),
) -> None:
    """Debug print of one record's m stream and its placement counters."""
    raw = Path(npz).read_bytes()
    record = records.read_record(raw)
    stream, stats = units.build_m_with_stats(record, variant, k)
    sil, pad = units.token_bounds(k)
    values, counts = np.unique(stream, return_counts=True)
    tokens = {
        ("SIL" if int(v) == sil else "PAD" if int(v) == pad else f"u{int(v)}"): int(c)
        for v, c in zip(values, counts, strict=True)
    }
    typer.echo(
        json.dumps(
            {
                "frames": int(stream.shape[0]),
                "variant": variant,
                "k": k,
                "placed": stats.placed,
                "shifted": stats.shifted,
                "dropped": stats.dropped,
                "tokens": {
                    "pad": tokens.get("PAD", 0),
                    "sil": tokens.get("SIL", 0),
                    "units": sum(v for key, v in tokens.items() if key.startswith("u")),
                },
                "head": stream[:head].tolist(),
                "dtype": str(stream.dtype),
            },
            indent=2,
        )
    )


# --- Part B -----------------------------------------------------------------------
def _require_plan(state: State, splits: str | None) -> None:
    """Refuse to start when a requested split was never indexed (ranks are final)."""
    wanted = [part.strip() for part in (splits or "").split(",") if part.strip()]
    indexed = sorted({row["split"] for row in state._all("SELECT DISTINCT split FROM files")})
    if not indexed:
        raise typer.BadParameter("no files are indexed; run `hohere index` first")
    missing = [split for split in wanted if split not in indexed]
    if missing:
        raise typer.BadParameter(
            f"split(s) {missing} are not indexed (indexed: {indexed}); "
            f"run `hohere index --splits {','.join(missing)}` first"
        )
    planned = state.get_meta("plan_splits")
    if planned is not None and sorted(planned) != indexed:
        raise typer.BadParameter(
            f"the plan covers splits {planned} but the DB indexes {indexed}; "
            "the order is final once encoding started"
        )


@app.command()
def plan(
    state_path: str = typer.Option(DEFAULT_STATE, "--state", help="SQLite state DB"),
    min_s: float = typer.Option(chunk_mod.MIN_S, "--min-s", help="minimum chunk length"),
    max_s: float = typer.Option(chunk_mod.MAX_S, "--max-s", help="maximum chunk length"),
) -> None:
    """Chunk every indexed file from its VAD spans and freeze the tier numbers."""
    with State(state_path) as state:
        report = state.plan_chunks(min_s=min_s, max_s=max_s)
        tiers = {
            str(tier): state.tier_hours(tier if tier == "full" else float(tier))
            for tier in order.TIERS
        }
    typer.echo(
        json.dumps(
            {
                "files": report["files"],
                "chunks": report["chunks"],
                "chunk_hours": round(report["chunk_hours"], 2),
                "tiers": {
                    key: {name: round(value, 2) for name, value in numbers.items()}
                    for key, numbers in tiers.items()
                },
            },
            indent=2,
        )
    )


@app.command("fit-units")
def fit_units(
    state_path: str = typer.Option(DEFAULT_STATE, "--state", help="SQLite state DB"),
    hours: float = typer.Option(100.0, "--hours", help="hours of tier to embed"),
    ks: str = typer.Option("4096,8192", "--k", help="comma separated codebook sizes"),
    device: str = typer.Option("cuda", "--device", help="cuda or cpu"),
    wavlm_dtype: str = typer.Option("float32", "--wavlm-dtype"),
    units_dir: str = typer.Option("work/units", "--units-dir"),
    models_dir: str = typer.Option("work/models", "--models-dir"),
    iters: int = typer.Option(kmeans.ITERS, "--iters", help="k-means iterations"),
    hf_repo: str | None = typer.Option(None, "--hf-repo", help="dataset repo to upload to"),
    upload_codebooks: bool = typer.Option(False, "--upload", help="upload the codebooks"),
    upload_embeddings: bool = typer.Option(False, "--upload-embeddings"),
    embed_only: bool = typer.Option(False, "--embed-only", help="skip the k-means fit"),
    splits: str | None = typer.Option(None, "--splits", help="splits the plan must cover"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Embed a tier with ZeroSyl and fit the unit codebooks (needs the `models` extra)."""
    _setup_logging(verbose)
    wanted = tuple(int(part) for part in ks.split(",") if part.strip())
    with State(state_path) as state:
        _require_plan(state, splits)
        files = state.select_until(hours)
        if not files:
            raise typer.BadParameter(f"no files within {hours} h; is the plan built?")
        stages = pipeline_mod.real_stages(
            device=device, models_dir=models_dir, wavlm_dtype=wavlm_dtype,
            units_dir=units_dir,
        )
        embedded = fitunits.embed(files, stages, out_dir=Path(units_dir) / "emb")
        typer.echo(json.dumps(embedded, indent=2))
        if embed_only:
            return
        embeddings = fitunits.load_embeddings(Path(units_dir) / "emb")
        report = fitunits.fit_all(
            embeddings, wanted, out_dir=units_dir, hours=hours, device=device, iters=iters
        )
    typer.echo(json.dumps(report, indent=2))
    if upload_codebooks or upload_embeddings:
        if not hf_repo:
            raise typer.BadParameter("--upload needs --hf-repo")
        uploaded = fitunits.upload(
            units_dir, hf_repo, ks=wanted, upload_embeddings=upload_embeddings
        )
        typer.echo(json.dumps({"uploaded": uploaded}, indent=2))


@app.command()
def run(
    state_path: str = typer.Option(DEFAULT_STATE, "--state", help="SQLite state DB"),
    until_hours: str = typer.Option("full", "--until-hours", help="hours of tier or `full`"),
    device: str = typer.Option("auto", "--device", help="auto, cuda or cpu"),
    hf_repo: str | None = typer.Option(None, "--hf-repo", help="dataset repo to upload to"),
    prefix: str = typer.Option("librilight", "--prefix", help="path prefix inside the repo"),
    no_upload: bool = typer.Option(False, "--no-upload", help="keep the shards locally"),
    units_dir: str = typer.Option("work/units", "--units-dir"),
    models_dir: str = typer.Option("work/models", "--models-dir"),
    out_dir: str = typer.Option("work/shards", "--out-dir"),
    status_port: int | None = typer.Option(
        None, "--status-port", help="serve the status page"
    ),
    status_token: str | None = typer.Option(None, "--status-token"),
    download_workers: int = typer.Option(8, "--download-workers"),
    decode_workers: int = typer.Option(4, "--decode-workers"),
    mimi_batch: int = typer.Option(16, "--mimi-batch"),
    wavlm_dtype: str = typer.Option("float32", "--wavlm-dtype"),
    bench_minutes: float | None = typer.Option(
        None, "--bench-minutes", help="measure, do not produce"
    ),
    splits: str | None = typer.Option(None, "--splits", help="splits the plan must cover"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Stream the tier through the models and upload full shards."""
    _setup_logging(verbose)
    target = until_hours if until_hours == "full" else float(until_hours)
    with State(state_path) as state:
        _require_plan(state, splits)
        _require_codebooks(units_dir)
        state.recover(out_dir)
        files = state.select_until(target)
        if not files:
            raise typer.BadParameter("the tier is empty; run `hohere index` and `hohere plan`")
        resolved = _resolve_device(device)
        if resolved == "cpu":
            import os as _os

            import torch

            torch.set_num_threads(_os.cpu_count() or 1)
        stages = pipeline_mod.real_stages(
            device=resolved, models_dir=models_dir, wavlm_dtype=wavlm_dtype,
            units_dir=units_dir, mimi_batch=mimi_batch,
        )
        if bench_minutes is not None:
            payload = bench_mod.run_bench(
                state=state,
                stages=stages,
                files=files,
                minutes=bench_minutes,
                out_dir=Path(out_dir).parent / "bench-shards",
                dtype=wavlm_dtype,
            )
            typer.echo(json.dumps(payload, indent=2))
            return
        progress = status_mod.Progress(device=resolved, target=target, state=state)
        server = (
            status_mod.serve(progress, state, port=status_port, token=status_token)
            if status_port is not None
            else None
        )
        uploader = None if no_upload or not hf_repo else shards_mod.Uploader(hf_repo, prefix)
        state.set_meta("target_hours", target)
        runner = pipeline_mod.Runner(
            state=state,
            stages=stages,
            out_dir=out_dir,
            progress=progress,
            uploader=uploader,
            download_workers=download_workers,
            decode_workers=decode_workers,
        )
        restore = pipeline_mod.install_stop_handlers(runner)
        try:
            stats = runner.run(files)
        finally:
            restore()
            if server is not None:
                server.stop()
    typer.echo(
        json.dumps(
            {
                "files": stats.files,
                "chunks": stats.chunks,
                "audio_hours": round(stats.audio_seconds / 3600.0, 3),
                "wall_seconds": round(stats.wall_seconds, 1),
                "errors": stats.errors,
            },
            indent=2,
        )
    )


def _require_codebooks(units_dir: str) -> None:
    missing = [
        str(Path(units_dir) / f"k{k}.npy") for k in (4096, 8192)
        if not (Path(units_dir) / f"k{k}.npy").exists()
    ]
    if missing:
        raise typer.BadParameter(
            f"missing {', '.join(missing)}; run `hohere fit-units` first"
        )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":  # pragma: no cover
    app()
