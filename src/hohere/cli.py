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
    from hohere.pipeline import Pipeline

    return Pipeline(cfg)


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
    verbose: bool = False,
) -> None:
    """Synthesize every accepted dialogue with Kokoro."""
    report = _pipeline(_load(config, dataset, verbose)).synthesize(limit)
    _print_counts("synthesize", report["counts"])


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
def status(config: ConfigOpt = Path("configs/default.yaml"), dataset: DatasetOpt = None) -> None:
    """Show progress of each stage."""
    cfg = _load(config, dataset)
    from hohere.storage.local import DatasetStore

    store = DatasetStore(cfg.dataset.output_dir)
    planned = len(store.load_plan()) if store.plan_path.exists() else 0
    accepted = store.accepted_dialogue_ids()
    convs = store.conversation_ids()
    passed = [c for c in convs if store.load_metadata(c).quality_status == "passed"]
    hours = sum(store.load_metadata(c).duration_seconds for c in passed) / 3600
    _print_counts(
        "status",
        {
            "planned": planned,
            "dialogues accepted": len(accepted),
            "synthesized": sum(store.has_segments(c) for c in accepted),
            "composed": len(convs),
            "passed": len(passed),
            "passed hours": f"{hours:.2f} / {cfg.dataset.target_hours}",
            "rejections": len(store.all_rejections()),
        },
    )


def main() -> None:  # pragma: no cover
    sys.exit(app())
