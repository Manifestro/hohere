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


def _tts(provider: str):
    from hohere.config.schema import SpeechConfig

    if provider == "mock":
        from hohere.synthesis.mock import MockSynthesizer

        return MockSynthesizer(24000)
    from hohere.synthesis.kokoro import KokoroSynthesizer

    return KokoroSynthesizer(SpeechConfig())


StoreOpt = Annotated[Path, typer.Option(help="E1 store (output_dir of configs/bench-e1.yaml).")]
OutOpt = Annotated[Path, typer.Option(help="Bench export directory.")]


@app.command()
def e2(store: StoreOpt = Path("data/bench-e1"), out: OutOpt = Path("data/bench")) -> None:
    """E2 single turn, derived from the E1 store."""
    from hohere.bench.suites import single_turn_rows, write_suite
    from hohere.storage.local import DatasetStore

    console.print(write_suite(single_turn_rows(DatasetStore(store)), "e2_single_turn", out))


@app.command()
def e4(out: OutOpt = Path("data/bench"), per_cell: int = 25, provider: str = "kokoro") -> None:
    """E4 memory: templated scripts, one seen and one unseen user voice per item."""
    from hohere.bench.suites import memory_items, memory_rows, write_suite

    rows = memory_rows(memory_items(per_cell), _tts(provider))
    console.print(write_suite(rows, "e4_memory", out))


@app.command()
def e5(
    store: StoreOpt = Path("data/bench-e1"),
    out: OutOpt = Path("data/bench"),
    provider: str = "kokoro",
) -> None:
    """E5 interruptions (and backchannel controls), derived from the E1 store."""
    from hohere.bench.suites import interruption_rows, write_suite
    from hohere.storage.local import DatasetStore

    rows = interruption_rows(DatasetStore(store), _tts(provider))
    console.print(write_suite(rows, "e5_interruptions", out))


@app.command()
def e6(
    out: OutOpt = Path("data/bench"),
    config: Annotated[Path, typer.Option(help="Topics, personas, LLM endpoint.")] = Path(
        "configs/bench-e1.yaml"
    ),
    n: int = 160,
    provider: str = "kokoro",
) -> None:
    """E6 long user turns: texts from the configured LLM endpoint, then TTS."""
    from hohere.bench.suites import EVAL_VOICES, long_turn_rows, long_turn_texts, write_suite
    from hohere.config.schema import load_config
    from hohere.generators.prompts import USER_PERSONAS

    cfg = load_config(config)
    items = long_turn_texts(
        cfg.dialogue, cfg.dataset.topics, cfg.dialogue.user_personas or USER_PERSONAS, n
    )
    console.print(write_suite(long_turn_rows(items, _tts(provider), EVAL_VOICES),
                              "e6_long_turns", out))


@app.command()
def card(out: OutOpt = Path("data/bench"), repo_id: str = DEFAULT_REPO) -> None:
    """Root README covering every suite present in `out` (E3 stays under data/)."""
    from hohere.bench.suites import bench_card

    (out / "README.md").write_text(bench_card(out, repo_id))
    console.print(f"wrote {out / 'README.md'}")


if __name__ == "__main__":
    app()
