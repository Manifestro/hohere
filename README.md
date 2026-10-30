# hohere

A restartable dataset pipeline that streams all of [Libri-Light](https://github.com/facebookresearch/libri-light)
straight out of the remote `.tar` archives (no downloads), cuts every recording into
30–60 s chunks on the Libri-Light VAD, and stores per chunk:

* Mimi codes — 32 codebooks at 12.5 Hz,
* ZeroSyl syllable-unit events (frozen WavLM-Large, k-means codebooks),
* internal pause (`SIL`) events from Silero.

One chunk is one `.npz` member of a ~1 GB `shard-NNNNNN.tar`; its metadata is a line
in the matching `shard-NNNNNN.jsonl`. A full shard is uploaded to a Hugging Face
dataset repo and deleted locally, so nothing accumulates on disk. Nothing is rescanned
after a restart: the archive index lives in a SQLite state DB, shards move
`open → closed → uploaded`, and a web page shows progress and ETA.

## Install

```sh
uv sync                         # code + CLI, no torch (index/plan/m-stream)
uv run pytest
uv run ruff check
```

Models live behind the `models` extra (`torch`, `torchaudio`, `moshi`, `silero-vad`,
`scipy`). They are never needed for indexing, planning or the tests. For a real run use
one of the two setup scripts:

```sh
scripts/setup_pod.sh    # runpod/pytorch (cu128, torch 2.8): models on top of the image torch
scripts/setup_cpu.sh    # plain Ubuntu 24.04: uv venv + CPU torch wheels + the models
```

Both scripts install `hohere -e .` first and then the model stack **without replacing
the image's torch**: `moshi` goes in with `--no-deps` plus its light dependencies, and
`torchaudio` is only installed when it does not already match the installed torch.

## Commands

```sh
# 1. index the archives (one pass, cached in the state DB forever)
uv run hohere index --splits small,medium,large --state work/state.db --workers 64

# 2. chunk every file from its VAD spans and freeze the tier numbers
uv run hohere plan --state work/state.db

# 3. embed a tier and fit the unit codebooks (needs the models)
uv run hohere fit-units --state work/state.db --hours 100 --k 4096,8192 --device cuda \
    --hf-repo bagzhan/hohere-units --upload

# 4. the run itself: download -> decode -> models -> shard -> upload
uv run hohere run --state work/state.db --until-hours 100 --device auto \
    --hf-repo bagzhan/hohere --prefix librilight \
    --status-port 8080 --status-token s3cret

# the same JSON as the status page, without a server
uv run hohere status --state work/state.db

# measure instead of produce: no upload, no state changes, writes work/bench.json
uv run hohere run --state work/state.db --until-hours 100 --bench-minutes 10

# debug one record's m stream
uv run hohere m-stream --npz work/chunk.npz --variant B --k 4096
```

`run` refuses to start when `work/units/k4096.npy`/`k8192.npy` are missing
(`fit-units` first) and when a requested split was never indexed — the order is final
once encoding started, so a split cannot appear later and reshuffle a finished tier.

## Status page

`--status-port 8080` serves `GET /api/status` (JSON) and `GET /` (a small page with a
progress bar and a human ETA, no external assets, works on a phone). With
`--status-token T` every request needs `?token=T`. The same payload backs
`hohere status`, so the offline and online views cannot drift apart. Stage timings
per chunk (download, decode, vad, wavlm, assign, mimi, write, upload) are what the
benchmark turns into x-realtime numbers.

## Benchmark

`--bench-minutes M` runs the real pipeline (same threads, same models, real shards)
against a throwaway copy of the state DB for M minutes of chunk audio, then writes
`work/bench.json`: device (GPU name or CPU model, cores, RAM), torch version, dtype,
audio seconds, wall seconds, x-realtime overall and per stage, the share of wall time
spent waiting in queues, boundary agreement between float32 and the chosen
`--wavlm-dtype`, and the estimated wall time for 100/1000/5000 h and the full plan.

## Order and tiers

Files are ordered by `sha256(file_id)` and the order is frozen when `plan` runs: the
first N hours of *kept chunk audio* form tier N (`100, 200, 500, 1000, 5000, full`).
Chunking happens at index time, so a tier is exact before a single chunk is encoded
and `--until-hours N` is just `cum_hours <= N` (the file crossing N is included).
Every chunk records its `order_rank` and `cum_hours` in the manifest.

## Layout

```
src/hohere/
  remote.py      HTTP Range reader: retries, stall guard, keep-alive connections
  tarindex.py    ustar/GNU/PAX walking, parallel slices, chain check
  librilight.py  archive entries -> SourceFile (duration, VAD spans, file_id)
  order.py       sha256 order, tiers, hours -> files
  chunk.py       30-60 s chunks on the VAD (never over 60 s)
  units.py       the m-stream builder (units, SIL, PAD) and training pairs
  records.py     one chunk's npz: write, read, validate
  shards.py      ShardWriter + Uploader
  state.py       SQLite resume state, plan (n_chunks, cum_hours), tiers
  status.py      Progress, /api/status, the HTML page
  vad.py         Silero spans, WavLM speech mask, internal pauses
  zerosyl.py     frozen WavLM-Large boundaries + speech-only unit embeddings
  vendor/wavlm/  WavLM from nicolvisser/ZeroSyl (MIT), unmodified
  kmeans.py      spherical k-means in torch
  mimi.py        Mimi codes, 32 codebooks at 12.5 Hz
  fitunits.py    hohere fit-units: embedding parts + codebook fitting
  pipeline.py    the threaded run, real stages, stop handlers
  bench.py       --bench-minutes measurement and estimates
  cli.py         typer app
```

## License

MIT © Bagzhan. `src/hohere/vendor/wavlm/` is MIT code from
[nicolvisser/ZeroSyl](https://github.com/nicolvisser/ZeroSyl) (its `LICENSE` sits
next to it); see `scripts/check_zerosyl.py` for the parity check against the original.
