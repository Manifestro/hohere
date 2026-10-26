"""`python -m hohere.bench build|publish`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from hohere.bench.build import SEEN_VOICES, UNSEEN_VOICES, build_rows, write_bench
from hohere.bench.probes import build_probes

app = typer.Typer(help="hohere-bench: minimal-pair probes (evaluation only).", no_args_is_help=True)
console = Console()
DEFAULT_REPO = "manifestro/hohere-bench"


@app.command()
def build(
    out: Annotated[Path, typer.Option(help="Export directory.")] = Path("data/bench"),
    pairs: Annotated[int, typer.Option(help="Echo pairs per category.")] = 70,
    provider: Annotated[str, typer.Option(help="kokoro or mock.")] = "kokoro",
    repo_id: str = DEFAULT_REPO,
) -> None:
    """Synthesize every probe pair in every user voice and write parquet + card."""
    from hohere.config.schema import SpeechConfig

    probes = build_probes(pairs)
    if provider == "mock":
        from hohere.synthesis.mock import MockSynthesizer

        tts = MockSynthesizer(24000)
    else:
        from hohere.synthesis.kokoro import KokoroSynthesizer

        tts = KokoroSynthesizer(SpeechConfig())
    rows = build_rows(probes, tts, SEEN_VOICES + UNSEEN_VOICES)
    console.print(write_bench(rows, out, repo_id))


@app.command()
def publish(
    out: Annotated[Path, typer.Option(help="Export directory.")] = Path("data/bench"),
    repo_id: str = DEFAULT_REPO,
    tag: str | None = None,
    token_env: str = "HF_TOKEN",
) -> None:
    """Upload to a PRIVATE dataset repo (refuses a public one)."""
    from hohere.publishing.huggingface import _api

    api = _api(token_env)
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="dataset").private:
        console.print(f"[red]{repo_id} is public; refusing to upload[/]")
        raise typer.Exit(1)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="dataset",
                      commit_message="hohere-bench release")
    if tag:
        api.create_tag(repo_id, tag=tag, repo_type="dataset", tag_message=f"Release {tag}")
    console.print(f"published {repo_id} (private){f', tag {tag}' if tag else ''}")


if __name__ == "__main__":
    app()
