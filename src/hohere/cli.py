"""The `hohere` command line.

Part A ships `index`, `status` and `m-stream`. Part B adds `fit-units` (train the
ZeroSyl unit codebooks) and `run` (stream -> chunk -> encode -> shard -> upload, with
`--status-port`, `--status-token`, `--target`, `--no-upload`); the terminal block at
the bottom of this file is where they belong.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import typer

from . import librilight, records, units
from . import status as status_mod
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
# @app.command("fit-units")  ->  train the ZeroSyl codebooks (needs the `models` extra)
# @app.command()             ->  run: the streaming pipeline, the re-upload loop and
#                                the `--status-port` / `--status-token` server.


if __name__ == "__main__":  # pragma: no cover
    app()
