"""The run pipeline end to end with fake models: real shards, no torch, no network."""

from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from hohere import bench, kmeans, pipeline, records, vad
from hohere.librilight import SourceFile
from hohere.state import State
from hohere.status import Progress

SR = vad.SAMPLE_RATE


def source(file_id: str, seconds: float = 70.0, rank: int = 0) -> SourceFile:
    """A file whose VAD is a train of 10 s spans with 1 s gaps (so it chunks)."""
    spans = [
        (index * 11.0, min(index * 11.0 + 10.0, seconds))
        for index in range(max(1, int(seconds // 11)))
    ]
    return SourceFile(
        file_id=file_id,
        split="small",
        speaker="1",
        book="b",
        archive_url="http://example/small.tar",
        flac_offset=rank * 1000,
        flac_size=1000,
        json_offset=rank * 1000 + 10,
        json_size=10,
        duration_s=seconds,
        vad=spans,
    )


class FakeModels:
    """Deterministic stand-ins for Silero + WavLM + Mimi + k-means."""

    def __init__(self, dim: int = 8, fail: tuple[str, ...] = ()) -> None:
        self.dim = dim
        self.fail = set(fail)
        self.downloaded: list[str] = []
        self.decoded: list[str] = []
        self.chunks = 0

    def stages(self, mimi_batch: int = 4) -> pipeline.Stages:
        rng = np.random.default_rng(0)
        centroids = {k: kmeans_init(k, self.dim, rng) for k in (4096, 8192)}

        def download(file: SourceFile) -> bytes:
            if file.file_id in self.fail:
                raise ConnectionError("archive is down")
            self.downloaded.append(file.file_id)
            return b"flac" + file.file_id.encode()

        def decode(raw: bytes, file: SourceFile) -> np.ndarray:
            if raw[4:].decode() != file.file_id:
                raise ValueError("wrong payload")
            self.decoded.append(file.file_id)
            return np.linspace(-0.5, 0.5, int(file.duration_s * SR), dtype=np.float32)

        def speech(wav: np.ndarray) -> list[tuple[int, int]]:
            n = wav.shape[0]
            return [(0, n // 2), (n // 2 + SR // 2, n)]

        def encode_units(wav: np.ndarray, spans: list[tuple[int, int]]):
            frames = (wav.shape[0] + 80) // 320
            mask = vad.speech_mask50(spans, frames)
            kept = np.flatnonzero(mask)
            if kept.size == 0:
                return EmptySegments(self.dim)
            starts = kept[:: max(1, kept.size // 4)].astype(np.int32)
            ends = (starts + 1).astype(np.int32)
            emb = rng.normal(size=(starts.size, self.dim)).astype(np.float32)
            return SimpleSegments(starts, ends, emb)

        def encode_mimi(chunks: list[np.ndarray]) -> list[np.ndarray]:
            self.chunks += len(chunks)
            out = []
            for chunk in chunks:
                frames = int(np.ceil((chunk.shape[0] * 24 / 16) / 1920))
                out.append(np.zeros((32, frames), dtype=np.int16))
            return out

        def assign(emb: np.ndarray, k: int) -> np.ndarray:
            if emb.shape[0] == 0:
                return np.zeros(0, dtype=np.int16)
            return kmeans.assign(emb, centroids[k]) if torch_ok() else np.zeros(
                emb.shape[0], dtype=np.int16
            )

        return pipeline.Stages(
            download=download,
            decode=decode,
            speech=speech,
            encode_units=encode_units,
            encode_mimi=encode_mimi,
            assign=assign,
            mimi_batch=mimi_batch,
        )


class SimpleSegments:
    def __init__(self, unit_start, unit_end, embeddings) -> None:
        self.unit_start = unit_start
        self.unit_end = unit_end
        self.embeddings = embeddings

    def __len__(self) -> int:
        return int(self.unit_start.shape[0])


class EmptySegments(SimpleSegments):
    def __init__(self, dim: int) -> None:
        super().__init__(
            np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, dim), np.float32)
        )


def torch_ok() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def kmeans_init(k: int, dim: int, rng) -> np.ndarray:
    table = rng.normal(size=(k, dim)).astype(np.float32)
    return table / np.linalg.norm(table, axis=1, keepdims=True)


def prepared_state(tmp_path, files: list[SourceFile]) -> State:
    state = State(tmp_path / "state.db")
    state.add_files(files)
    state.assign_order()
    state.plan_chunks()
    return state


def test_the_pipeline_writes_real_shards_in_order(tmp_path):
    files = [source(f"small-1-b-f{i}") for i in range(4)]
    state = prepared_state(tmp_path, files)
    ordered = state.select_until("full")
    models = FakeModels()
    progress = Progress()
    runner = pipeline.Runner(
        state=state,
        stages=models.stages(),
        out_dir=tmp_path / "shards",
        progress=progress,
        uploader=None,
        mark_done=False,
    )
    stats = runner.run(ordered)
    assert stats.files == 4
    assert stats.chunks >= 4
    assert stats.errors == 0
    tars = sorted((tmp_path / "shards").glob("shard-*.tar"))
    assert tars
    ids: list[str] = []
    for tar in tars:
        with tarfile.open(tar) as handle:
            for member in handle:
                ids.append(member.name)
                payload = records.read_record(handle.extractfile(member).read())
                assert payload["codes"].shape[0] == 32
                assert payload["unit_k4096"].dtype == np.int16
    # every file is complete and its own chunks stay in order (files interleave
    # because the stages run in parallel; the shard order within a file is fixed)
    per_file: dict[str, list[int]] = {}
    for name in ids:
        stem, index = name[: -len(".npz")].rsplit("-", 1)
        per_file.setdefault(stem, []).append(int(index))
    assert len(per_file) == len(files)
    for indexes in per_file.values():
        assert indexes == sorted(indexes)
    lines = []
    for jsonl in sorted((tmp_path / "shards").glob("shard-*.jsonl")):
        lines.extend(json.loads(line) for line in jsonl.read_text().splitlines())
    assert sorted(line["id"] + ".npz" for line in lines) == sorted(ids)
    ranks = [line["order_rank"] for line in lines]
    assert ranks and set(ranks) <= {0, 1, 2, 3}
    stages_done = {key: value[0] for key, value in progress.stage_totals.items()}
    for stage in ("download", "decode", "vad", "wavlm", "assign", "mimi", "write"):
        assert stages_done[stage] > 0, stage


def test_the_manifest_names_the_centroids(tmp_path):
    state = prepared_state(tmp_path, [source("small-1-b-f0")])
    stages = FakeModels().stages()
    stages.units_version = {"units_k4096": "x", "units_k8192": "y"}
    pipeline.Runner(
        state=state, stages=stages, out_dir=tmp_path / "shards", mark_done=False
    ).run(state.select_until("full"))
    lines = next((tmp_path / "shards").glob("shard-*.jsonl")).read_text().splitlines()
    assert lines
    for line in map(json.loads, lines):
        assert line["units_k4096"] == "x"
        assert line["units_k8192"] == "y"


def test_until_hours_selects_a_prefix_of_the_order(tmp_path):
    files = [source(f"small-1-b-f{i}", seconds=3600.0) for i in range(5)]
    state = prepared_state(tmp_path, files)
    ordered = state.select_until("full")
    # a tier of 0 h holds the files with no hours yet, plus the first file crossing 0
    assert state.select_until(0.0) == ordered[:1]
    assert state.select_until("full") == ordered
    assert state.tier_hours(0.0)["files"] == 1
    # every tier is a prefix of the order, never a reordering
    for hours in (0.0, 0.01, 1.0):
        tier = state.select_until(hours)
        assert [f.file_id for f in tier] == [f.file_id for f in ordered][: len(tier)]


def test_a_failing_file_is_recorded_and_the_run_continues(tmp_path):
    files = [source(f"small-1-b-f{i}") for i in range(3)]
    state = prepared_state(tmp_path, files)
    models = FakeModels(fail=(files[1].file_id,))
    progress = Progress()
    runner = pipeline.Runner(
        state=state,
        stages=models.stages(),
        out_dir=tmp_path / "shards",
        progress=progress,
        mark_done=True,
    )
    stats = runner.run(state.select_until("full"))
    assert stats.errors == 1
    assert state.count_files(status="error") == 1
    assert stats.files == 2
    assert progress.snapshot()["errors"]["count"] == 1
    with tarfile.open(next(iter(sorted((tmp_path / "shards").glob("shard-*.tar"))))) as handle:
        assert len(handle.getmembers()) >= 2


def test_an_interrupted_run_resumes_from_the_recovered_state(tmp_path):
    files = [source(f"small-1-b-f{i}") for i in range(6)]
    state = prepared_state(tmp_path, files)
    models = FakeModels()
    first = pipeline.Runner(
        state=state,
        stages=models.stages(),
        out_dir=tmp_path / "shards",
        uploader=None,
        mark_done=True,
        crash_after_files=2,
    )
    first.run(state.select_until("full"))
    done_before = state.count_files(status="done")
    assert done_before >= 2
    # a restart discards whatever shard was still open
    state.recover(tmp_path / "shards")
    assert not state.shards(status="open")
    second = pipeline.Runner(
        state=state,
        stages=models.stages(),
        out_dir=tmp_path / "shards",
        uploader=None,
        mark_done=True,
    )
    todo = [
        file for file in state.select_until("full")
        if state.file_status(file.file_id) != "done"
    ]
    stats = second.run(todo)
    assert stats.files == len(todo)
    assert state.count_files(status="done") == len(files)


def test_upload_is_what_marks_a_file_done(tmp_path):
    files = [source("small-1-b-f0")]
    state = prepared_state(tmp_path, files)
    uploaded: list[str] = []

    class Uploader:
        def upload(self, shard) -> None:
            uploaded.append(shard.name)
            for path in shard.paths():
                path.unlink(missing_ok=True)

    runner = pipeline.Runner(
        state=state,
        stages=FakeModels().stages(),
        out_dir=tmp_path / "shards",
        uploader=Uploader(),
        mark_done=True,
    )
    runner.run(state.select_until("full"))
    assert uploaded
    assert state.count_files(status="done") == 1
    assert state.shards(status="uploaded")
    assert not list((tmp_path / "shards").glob("shard-*.tar"))


def test_progress_reports_every_stage_and_the_chunk_count(tmp_path):
    files = [source("small-1-b-f0")]
    state = prepared_state(tmp_path, files)
    progress = Progress()
    runner = pipeline.Runner(
        state=state, stages=FakeModels().stages(), out_dir=tmp_path / "shards",
        progress=progress, mark_done=False,
    )
    runner.run(state.select_until("full"))
    payload = progress.snapshot()
    assert payload["done"]["chunks"] == runner.stats.chunks
    assert payload["stage_times"]["wavlm"]["n"] == runner.stats.chunks


def test_bench_writes_the_json_and_leaves_the_state_alone(tmp_path):
    files = [source(f"small-1-b-f{i}") for i in range(3)]
    state = prepared_state(tmp_path, files)
    models = FakeModels()
    stages = models.stages()
    payload = bench.run_bench(
        state=state,
        stages=stages,
        files=state.select_until("full"),
        minutes=0.02,
        out_dir=tmp_path / "bench-shards",
        tmp_dir=tmp_path,
    )
    assert (tmp_path / "bench.json").exists()
    assert payload["audio_seconds"] > 0
    assert payload["x_realtime"] > 0
    assert payload["wall_seconds"] > 0
    assert set(payload["stages"]) >= {"download", "decode", "mimi", "write"}
    assert payload["queue_wait_share"] >= 0
    assert set(payload["estimate_wall_seconds"]) == {"100", "1000", "5000", "full"}
    assert payload["dtype"] == "float32"
    assert state.count_files(status="done") == 0  # the real DB is untouched


def test_bench_estimates_scale_with_the_plan(tmp_path):
    progress = Progress()
    progress.add("mimi", 10.0)
    stats = pipeline.Stats(audio_seconds=100.0, wall_seconds=20.0, chunks=4)
    payload = bench.build_payload(
        stats=stats, progress=progress, device={"cpu": "x"}, dtype="bfloat16", plan_hours=10.0
    )
    assert payload["x_realtime"] == pytest.approx(5.0)
    assert payload["stages"]["mimi"]["x_realtime"] == pytest.approx(10.0)
    assert payload["estimate_wall_seconds"]["100"] == pytest.approx(72000.0)
    assert payload["estimate_wall_seconds"]["full"] == pytest.approx(7200.0)


def test_real_stages_lazily_requires_the_models(tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.real_stages(units_dir=tmp_path / "units")


def test_kmeans_fits_and_assigns_when_torch_is_available():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 16)).astype(np.float32)
    result = kmeans.fit(emb, 8, iters=10, device="cpu")
    assert result.centroids.shape == (8, 16)
    assert len(result.objective) == 10
    ids = kmeans.assign(emb, result.centroids)
    assert ids.dtype == np.int16
    assert int(ids.max()) < 8
    assert len(set(ids.tolist())) > 1
