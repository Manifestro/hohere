"""Command-line interface: `uv run hohere <command>`."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from hohere import __version__
from hohere.config.schema import HohereConfig, dump_config, load_config

app = typer.Typer(
    name="hohere",
    help="Generate, synthesize, validate and publish two-channel conversational audio datasets.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config file.")]
LimitOpt = Annotated[
    int | None, typer.Option("--limit", "-n", help="Process at most N conversations.")
]
DatasetOpt = Annotated[
    Path | None, typer.Option("--dataset", "-d", help="Dataset output dir (overrides config).")
]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _setup_logging(output_dir: Path, verbose: bool) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    console_handler = RichHandler(console=console, show_path=False, rich_tracebacks=True)
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root.addHandler(console_handler)
    log_dir = output_dir / "reports" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "hohere.jsonl")
    file_handler.setFormatter(_JsonFormatter())
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "filelock", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(config: Path, dataset: Path | None = None, verbose: bool = False) -> HohereConfig:
    if not config.exists():
        console.print(f"[red]config not found:[/] {config}. Run `hohere init` first.")
        raise typer.Exit(2)
    cfg = load_config(config)
    if dataset is not None:
        cfg.dataset.output_dir = dataset
    _setup_logging(Path(cfg.dataset.output_dir), verbose)
    return cfg


def _pipeline(cfg: HohereConfig):
    import os

    from hohere.pipeline import Pipeline

    # Parallel workers set HOHERE_PROGRESS=0 so their bars do not overwrite each other.
    return Pipeline(cfg, show_progress=os.environ.get("HOHERE_PROGRESS", "1") != "0")


def _print_counts(title: str, counts: dict) -> None:
    table = Table(title=title, show_header=False)
    for k, v in counts.items():
        table.add_row(str(k), json.dumps(v) if isinstance(v, dict) else str(v))
    console.print(table)


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        console.print(f"hohere {__version__}")
        raise typer.Exit()


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the config.")] = Path(
        "configs/default.yaml"
    ),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing file.")] = False,
) -> None:
    """Write a default configuration file."""
    if path.exists() and not force:
        console.print(f"[yellow]{path} exists; use --force to overwrite[/]")
        raise typer.Exit(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_config(HohereConfig()))
    console.print(f"[green]wrote[/] {path}")


@app.command()
def plan(config: ConfigOpt = Path("configs/default.yaml"), dataset: DatasetOpt = None) -> None:
    """Create (or show) the dataset plan with fixed splits."""
    from hohere.planning import plan_summary

    specs = _pipeline(_load(config, dataset)).plan()
    _print_counts("plan", plan_summary(specs))


@app.command()
def generate(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    limit: LimitOpt = None,
    verbose: bool = False,
) -> None:
    """Generate and validate structured dialogues with Qwen."""
    report = _pipeline(_load(config, dataset, verbose)).generate(limit)
    _print_counts("generate", report["counts"] | {"rejections": report["rejection_reasons"]})


@app.command()
def synthesize(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    limit: LimitOpt = None,
    shard: Annotated[
        str | None,
        typer.Option("--shard", help="i/n: run worker i of n in parallel (e.g. 0/3, 1/3, 2/3)."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Run N worker processes on this machine (user voices are cached first).",
        ),
    ] = 1,
    warm: Annotated[
        str | None,
        typer.Option("--warm", help="Only fill the TTS cache for one speaker (A or B)."),
    ] = None,
    verbose: bool = False,
) -> None:
    """Synthesize every accepted dialogue (user voices + model voice)."""
    if workers > 1:
        _synthesize_workers(config, dataset, workers, verbose)
        return
    parsed = None
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        if not 0 <= i < n:
            raise typer.BadParameter("shard must be i/n with 0 <= i < n")
        parsed = (i, n)
    if warm is not None and warm not in ("A", "B"):
        raise typer.BadParameter("--warm must be A or B")
    pipe = _pipeline(_load(config, dataset, verbose))
    report = pipe.synthesize(limit, shard=parsed, warm_speaker=warm)
    _print_counts(f"warm {warm}" if warm else "synthesize", report["counts"])


def _synthesize_workers(config: Path, dataset: Path | None, workers: int, verbose: bool) -> None:
    """Run `workers` shard processes in parallel.

    With a separate model-voice TTS (e.g. Chatterbox) the user voices are cached first in
    one process, so the shard workers only load the model-voice TTS and fit a small GPU.
    When one TTS serves both speakers that pre-pass would only serialize work: skipped.
    """
    import subprocess

    base = [sys.executable, "-m", "hohere", "synthesize", "--config", str(config)]
    if dataset is not None:
        base += ["--dataset", str(dataset)]
    if verbose:
        base.append("--verbose")
    if load_config(config).speech.model_provider != "same":
        console.print("[bold]warming user-voice cache (speaker A)[/]")
        if subprocess.run([*base, "--warm", "A"]).returncode != 0:
            console.print("[red]warm-up failed[/]")
            raise typer.Exit(1)
    console.print(f"[bold]starting {workers} synthesis workers[/]")
    import os

    quiet = {**os.environ, "HOHERE_PROGRESS": "0"}
    procs = [
        subprocess.Popen([*base, "--shard", f"{i}/{workers}"], env=None if i == 0 else quiet)
        for i in range(workers)
    ]
    codes = [p.wait() for p in procs]
    if any(codes):
        console.print(f"[red]workers exited with {codes}; re-run to resume[/]")
        raise typer.Exit(1)
    console.print("[green]all workers finished[/]")


@app.command()
def compose(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    limit: LimitOpt = None,
    verbose: bool = False,
) -> None:
    """Compose synchronized two-channel conversations."""
    report = _pipeline(_load(config, dataset, verbose)).compose(limit)
    _print_counts("compose", report["counts"] | {"failures": report["failure_reasons"]})


@app.command()
def validate(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    no_audio: Annotated[bool, typer.Option("--no-audio", help="Structural checks only.")] = False,
) -> None:
    """Validate the dataset on disk and write reports/validation.json."""
    report = _pipeline(_load(config, dataset)).validate(audio=not no_audio)
    _print_counts("validation", report["summary"])
    if report["summary"]["failed"]:
        raise typer.Exit(1)


@app.command()
def manifest(config: ConfigOpt = Path("configs/default.yaml"), dataset: DatasetOpt = None) -> None:
    """Build manifest.json for the passed conversations."""
    m = _pipeline(_load(config, dataset)).manifest()
    _print_counts("manifest", m["totals"])


@app.command()
def export(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    out: Annotated[Path, typer.Option("--out", "-o", help="Export directory.")] = Path(
        "exports/latest"
    ),
    no_text: Annotated[
        bool, typer.Option("--no-text", help="Training view: drop utterance text from events.")
    ] = False,
) -> None:
    """Export a Hugging Face Datasets-compatible copy (parquet + card + manifests)."""
    m = _pipeline(_load(config, dataset)).export(out, include_text=False if no_text else None)
    _print_counts(f"export -> {out}", m["totals"])


@app.command()
def publish(
    repo_id: Annotated[str | None, typer.Option("--repo-id", help="namespace/dataset-name")] = None,
    export_dir: Annotated[Path, typer.Option("--export-dir", "-e")] = Path("exports/latest"),
    config: ConfigOpt = Path("configs/default.yaml"),
    private: Annotated[bool, typer.Option("--private/--public")] = True,
    tag: Annotated[str | None, typer.Option("--tag", help="Version tag, e.g. v0.1.0")] = None,
    yes_public: Annotated[
        bool, typer.Option("--yes-really-public", help="Required together with --public.")
    ] = False,
) -> None:
    """Upload an export to a (private by default) Hugging Face dataset repository."""
    from hohere.publishing.huggingface import PublishError, publish_dataset

    cfg = _load(config)
    repo = repo_id or cfg.publishing.repository_id
    if not repo:
        console.print("[red]no repository: pass --repo-id or set publishing.repository_id[/]")
        raise typer.Exit(2)
    try:
        result = publish_dataset(
            export_dir,
            repo,
            private=private,
            tag=tag,
            token_env=cfg.publishing.token_env,
            allow_public=yes_public,
        )
    except PublishError as e:
        console.print(f"[red]publish failed:[/] {e}")
        raise typer.Exit(1) from e
    _print_counts("published", result)


dialogues_app = typer.Typer(
    help="Move generated dialogues between machines (text on a GPU server, audio at home).",
    no_args_is_help=True,
)
app.add_typer(dialogues_app, name="dialogues")
RepoOpt = Annotated[str, typer.Option("--repo-id", help="Private HF dataset repo for dialogues.")]
PrefixOpt = Annotated[
    str | None, typer.Option("--prefix", help="Folder in the repo (default: v<dataset_version>).")
]


@dialogues_app.command("push")
def dialogues_push(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    repo_id: RepoOpt = "manifestro/hohere-dialogues",
    prefix: PrefixOpt = None,
) -> None:
    """Upload plan, splits and dialogues (no audio) to a private HF repo."""
    from hohere.publishing.huggingface import PublishError, push_dialogues
    from hohere.storage.local import DatasetStore

    cfg = _load(config, dataset)
    try:
        result = push_dialogues(
            DatasetStore(cfg.dataset.output_dir),
            repo_id,
            prefix or f"v{cfg.project.dataset_version}",
            cfg.publishing.token_env,
        )
    except PublishError as e:
        console.print(f"[red]push failed:[/] {e}")
        raise typer.Exit(1) from e
    _print_counts("dialogues pushed", result)


@dialogues_app.command("pull")
def dialogues_pull(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    repo_id: RepoOpt = "manifestro/hohere-dialogues",
    prefix: PrefixOpt = None,
) -> None:
    """Download plan, splits and dialogues into the local output dir, ready to synthesize."""
    from hohere.publishing.huggingface import PublishError, pull_dialogues
    from hohere.storage.local import DatasetStore

    cfg = _load(config, dataset)
    try:
        result = pull_dialogues(
            DatasetStore(cfg.dataset.output_dir),
            repo_id,
            prefix or f"v{cfg.project.dataset_version}",
            cfg.publishing.token_env,
        )
    except PublishError as e:
        console.print(f"[red]pull failed:[/] {e}")
        raise typer.Exit(1) from e
    _print_counts("dialogues pulled", result)


@dialogues_app.command("texts")
def dialogues_texts(
    repo_id: Annotated[str, typer.Option("--repo-id", help="Published dataset repo.")],
    out: Annotated[Path, typer.Option("--out", help="JSONL file to write.")],
    token_env: str = "HF_TOKEN",
) -> None:
    """Extract dialogue texts of a published dataset, for `dialogue.dedup_against`."""
    from hohere.publishing.huggingface import PublishError, hub_dialogue_texts

    try:
        result = hub_dialogue_texts(repo_id, out, token_env)
    except PublishError as e:
        console.print(f"[red]texts failed:[/] {e}")
        raise typer.Exit(1) from e
    _print_counts("dialogue texts", result)


librilight_app = typer.Typer(
    help="Turn Libri-Light medium into Mimi token files (index, encode, finalize).",
    no_args_is_help=True,
)
app.add_typer(librilight_app, name="librilight")


def _json_list(path: Path | None) -> list:
    return json.loads(path.read_text()) if path is not None else []


@librilight_app.command("index")
def librilight_index(
    out: Annotated[Path, typer.Option("--out", help="Work directory for the index and plan.")],
    url: Annotated[
        str, typer.Option("--url", help="Archive URL (HTTPS with Range).")
    ] = "https://dl.fbaipublicfiles.com/librilight/data/medium.tar",
    parts: Annotated[int, typer.Option("--parts", min=1, help="Slices walked in parallel.")] = 512,
    workers: Annotated[int, typer.Option("--workers", min=1, help="HTTP threads.")] = 64,
    exclude_speakers: Annotated[
        Path | None, typer.Option("--exclude-speakers", help="JSON list of speaker ids.")
    ] = None,
    exclude_recordings: Annotated[
        Path | None,
        typer.Option("--exclude-recordings", help="JSON list of \\[speaker, stem] pairs."),
    ] = None,
) -> None:
    """Walk the tar over HTTP, pair files, segment them and write WORK/plan.jsonl."""
    from hohere.librilight.index import run_index

    summary = run_index(
        out,
        url,
        parts,
        workers,
        _json_list(exclude_speakers),
        _json_list(exclude_recordings),
    )
    _print_counts("libri-light index", summary)


@librilight_app.command("encode")
def librilight_encode(
    plan: Annotated[Path, typer.Option("--plan", help="WORK/plan.jsonl.")],
    part: Annotated[int, typer.Option("--part", min=0, help="Shard of this process.")],
    parts: Annotated[int, typer.Option("--parts", min=1, help="Number of shards.")],
    out: Annotated[Path, typer.Option("--out", help="WORK/part-K.")],
    checkpoint: Annotated[
        str, typer.Option("--checkpoint", help="Mimi safetensors: local path or repo/file.")
    ] = "kyutai/moshiko-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors",
    batch_items: Annotated[int, typer.Option("--batch-items", min=1)] = 16,
    max_padded_frames: Annotated[int, typer.Option("--max-padded-frames", min=1)] = 3000,
    download_workers: Annotated[int, typer.Option("--download-workers", min=1)] = 4,
    decode_workers: Annotated[int, typer.Option("--decode-workers", min=1)] = 12,
    url: Annotated[
        str, typer.Option("--url", help="Archive URL (HTTPS with Range).")
    ] = "https://dl.fbaipublicfiles.com/librilight/data/medium.tar",
) -> None:
    """Encode this part's segments to Mimi tokens, one process per GPU."""
    from hohere.librilight.encode import run_encode

    result = run_encode(
        plan=plan,
        part=part,
        parts=parts,
        out=out,
        checkpoint=checkpoint,
        batch_items=batch_items,
        max_padded_frames=max_padded_frames,
        download_workers=download_workers,
        decode_workers=decode_workers,
        url=url,
    )
    _print_counts("libri-light encode", result)


@librilight_app.command("finalize")
def librilight_finalize(
    work: Annotated[Path, typer.Option("--work", help="Work directory with part-*/done.")],
    out: Annotated[Path | None, typer.Option("--out", help="Release dir (WORK/release).")] = None,
    shard_bytes: Annotated[int, typer.Option("--shard-bytes", min=1)] = 1_000_000_000,
    repo: Annotated[str | None, typer.Option("--repo", help="Dataset repo to upload to.")] = None,
    split: Annotated[
        str, typer.Option("--split", help="Folder and split name.")
    ] = "librilight.medium",
    private: Annotated[bool, typer.Option("--private/--public")] = True,
    token_env: Annotated[
        str, typer.Option("--token-env", help="Env var with the HF token.")
    ] = "HF_TOKEN",
) -> None:
    """Merge, check and pack the release (optionally upload it)."""
    from hohere.librilight.finalize import run_finalize

    result = run_finalize(
        work,
        out,
        shard_bytes=shard_bytes,
        split=split,
        repo=repo,
        private=private,
        token_env=token_env,
    )
    _print_counts("libri-light finalize", result["report"] | {"shards": result["shards"]})


@app.command()
def pipeline(
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
    limit: LimitOpt = None,
    verbose: bool = False,
) -> None:
    """Run generate -> synthesize -> compose -> validate. Safe to re-run after interruption."""
    results = _pipeline(_load(config, dataset, verbose)).run(limit)
    for stage in ("generate", "synthesize", "compose"):
        _print_counts(stage, results[stage]["counts"])
    _print_counts("validation", results["validate"])


@app.command()
def labels(
    out: Annotated[Path, typer.Option("--out", help="Directory for labels parquet + stats.")],
    repo_id: Annotated[
        str | None, typer.Option("--repo-id", help="Published dataset (reads events only).")
    ] = None,
    revision: Annotated[str | None, typer.Option(help="Dataset revision (default: main).")] = None,
    config: ConfigOpt = Path("configs/default.yaml"),
    dataset: DatasetOpt = None,
) -> None:
    """Frame activity (12.5 Hz) + turn-event table + transition stats, for codec models."""
    if repo_id:
        from hohere.publishing.huggingface import PublishError, hub_labels

        try:
            result = hub_labels(repo_id, out, revision)
        except PublishError as e:
            console.print(f"[red]labels failed:[/] {e}")
            raise typer.Exit(1) from e
        stats = result.pop("stats")
    else:
        from hohere.labels import store_labels, write_labels
        from hohere.storage.local import DatasetStore

        cfg = _load(config, dataset)
        rows = store_labels(DatasetStore(cfg.dataset.output_dir))
        stats = write_labels(rows, out, {"output_dir": str(cfg.dataset.output_dir)})
        result = {"conversations": len(rows), "out": str(out)}
    _print_counts("labels", result)
    console.print_json(
        data={
            k: stats[k]
            for k in (
                "counts",
                "backchannel_share_assistant",
                "interruption_share_user_interrupts",
                "gap_ms",
                "yield_ms",
            )
        }
    )


@app.command()
def status(config: ConfigOpt = Path("configs/default.yaml"), dataset: DatasetOpt = None) -> None:
    """Show progress of each stage."""
    cfg = _load(config, dataset)
    from hohere.storage.local import DatasetStore

    store = DatasetStore(cfg.dataset.output_dir)
    planned = len(store.load_plan()) if store.plan_path.exists() else 0
    accepted = store.accepted_dialogue_ids()
    convs = store.conversation_ids()
    passed = [c for c in convs if store.load_metadata(c).quality_status == "passed"]
    metas = [store.load_metadata(c) for c in passed]
    hours = sum(m.duration_seconds for m in metas) / 3600
    from hohere.validation.dialogue import dialogue_words

    text_words = sum(sum(dialogue_words(store.load_dialogue(c)).values()) for c in accepted)
    passed_words = sum((m.words_a or 0) + (m.words_b or 0) for m in metas)
    target = cfg.dataset.target_words
    _print_counts(
        "status",
        {
            "planned": planned,
            "dialogues accepted": len(accepted),
            "accepted words": f"{text_words:,}" + (f" / {target:,}" if target else ""),
            "synthesized": sum(store.has_segments(c) for c in accepted),
            "composed": len(convs),
            "passed": len(passed),
            "passed words": f"{passed_words:,}",
            "passed hours": f"{hours:.2f}" + ("" if target else f" / {cfg.dataset.target_hours}"),
            "rejections": len(store.all_rejections()),
        },
    )


def main() -> None:  # pragma: no cover
    sys.exit(app())
