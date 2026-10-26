"""Text-on-server / audio-at-home flow: warm cache, parallel workers, dialogue hand-off."""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hohere.config.schema import dump_config
from hohere.pipeline import Pipeline
from hohere.storage.local import DatasetStore


def test_warm_pass_caches_only_user_turns(cfg):
    pipe = Pipeline(cfg, show_progress=False)
    pipe.generate()
    ids = pipe.store.accepted_dialogue_ids()
    report = pipe.synthesize(warm_speaker="A")
    a_turns = sum(t.speaker == "A" for cid in ids for t in pipe.store.load_dialogue(cid).turns)
    assert report["counts"]["warmed"] == len(ids)
    assert report["counts"]["cache_misses"] + report["counts"]["cache_hits"] == a_turns
    assert not any(pipe.store.has_segments(c) for c in ids)
    full = Pipeline(cfg, show_progress=False).synthesize()
    assert full["counts"]["cache_hits"] >= a_turns
    assert all(pipe.store.has_segments(c) for c in ids)


def test_cli_workers_synthesize_everything(cfg, tmp_path):
    Pipeline(cfg, show_progress=False).generate()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(dump_config(cfg))
    proc = subprocess.run(
        [sys.executable, "-m", "hohere", "synthesize", "-c", str(cfg_path), "--workers", "2"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    store = DatasetStore(cfg.dataset.output_dir)
    assert all(store.has_segments(c) for c in store.accepted_dialogue_ids())


class _FakeHub:
    """Stands in for HfApi + snapshot_download, backed by a local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def create_repo(self, *a, **k):
        self.root.mkdir(parents=True, exist_ok=True)

    def repo_info(self, *a, **k):
        return type("Info", (), {"private": True})

    def upload_folder(self, folder_path, path_in_repo, allow_patterns, **k):
        src = Path(folder_path)
        for f in src.rglob("*"):
            rel = f.relative_to(src).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                dst = self.root / path_in_repo / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)

    def snapshot_download(self, repo_id, local_dir, allow_patterns, **k):
        for f in self.root.rglob("*"):
            rel = f.relative_to(self.root).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                dst = Path(local_dir) / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)


def test_dialogue_push_pull_roundtrip(cfg, tmp_path, monkeypatch):
    hub_mod = pytest.importorskip("huggingface_hub")
    import hohere.publishing.huggingface as hf

    fake = _FakeHub(tmp_path / "remote")
    monkeypatch.setattr(hf, "_api", lambda token_env: fake)
    monkeypatch.setattr(hub_mod, "snapshot_download", fake.snapshot_download)

    server = Pipeline(cfg, show_progress=False)
    server.generate()
    pushed = hf.push_dialogues(server.store, "org/dialogues", "v0.3")
    assert pushed["accepted_dialogues"] == len(server.store.accepted_dialogue_ids())
    assert not list((tmp_path / "remote").rglob("*.wav"))  # text only

    home = DatasetStore(tmp_path / "home")
    pulled = hf.pull_dialogues(home, "org/dialogues", "v0.3")
    assert pulled["accepted_dialogues"] == pushed["accepted_dialogues"]
    assert home.accepted_dialogue_ids() == server.store.accepted_dialogue_ids()
    assert home.load_plan() == server.store.load_plan()

    # splits are immutable: a home store with different splits must refuse the merge
    other = DatasetStore(tmp_path / "other")
    other.root.mkdir()
    (other.root / "splits.json").write_text('{"families": {}, "conversations": {}}')
    with pytest.raises(hf.PublishError, match="differs"):
        hf.pull_dialogues(other, "org/dialogues", "v0.3")
