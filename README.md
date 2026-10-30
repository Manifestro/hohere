# hohere

A restartable dataset pipeline that streams all of [Libri-Light](https://github.com/facebookresearch/libri-light)
straight out of the remote `.tar` archives (no downloads), cuts every recording into
30–60 s chunks on the Libri-Light VAD, and stores per chunk:

* Mimi codes — 32 codebooks at 12.5 Hz,
* ZeroSyl syllable-unit events,
* internal pause (`SIL`) events.

One chunk is one `.npz` member of a ~1 GB `shard-NNNNNN.tar`; the metadata of every
chunk is a line in the matching `shard-NNNNNN.jsonl`. A full shard is uploaded to a
Hugging Face dataset repo and deleted locally, so nothing accumulates on disk.

Nothing is rescanned after a restart: the archive index lives in a SQLite state DB,
shards are tracked as `open → closed → uploaded`, and a `--status-port` web page shows
progress and ETA.

Part A (this tree) is everything except the ML models: remote tar indexing, chunking,
the m-stream builder, records, shards, resume state, status page and CLI. Part B adds
`fit-units` and `run` behind the `models` extra (`torch`, `torchaudio`, `moshi`,
`silero-vad`, `scipy`).

## Install

```sh
uv sync                 # Part A only; no torch, no models
uv run pytest
uv run ruff check
```

## Commands

```sh
# index the archives into the state DB (one pass, cached forever)
uv run hohere index --splits small,medium,large --state work/state.db --workers 64

# same JSON as the web status page, without a server
uv run hohere status --state work/state.db

# print one record's m stream and the placement counts
uv run hohere m-stream --npz work/chunk.npz --variant B --k 4096
```

The `run` command (stream → chunk → encode → shard → upload, with `--status-port`,
`--status-token`, `--target` and `--no-upload`) arrives with Part B.

## Order and tiers

Files are ordered by `sha256(file_id)`; the order is fixed forever, so the first N
hours of kept chunk audio form tier N. Tiers are `100, 200, 500, 1000, 5000, full`
and every chunk carries its `order_rank` and cumulative `cum_hours` in the manifest,
which makes any tier selectable later with a simple filter.

## License

MIT © Bagzhan
