# Libri-Light medium → Mimi tokens

`medium.tar` (321 GB, `https://dl.fbaipublicfiles.com/librilight/data/medium.tar`,
md5 `c75e7ac62471bfbf2db77528d62a9b74`) is read over HTTPS Range requests and turned into
Mimi token files, one `.pt` per utterance segment. Nothing is downloaded whole: the index
walks the tar by range, `encode` fetches only the FLACs of its own part.

Everything below runs on the pod image with `ffmpeg` installed and torch 2.8 preinstalled:

```bash
uv sync --extra librilight
```

## 1. Index (no GPU)

```bash
uv run hohere librilight index --out /workspace/ll \
    --exclude-speakers /workspace/ll/exclude_speakers.json \
    --exclude-recordings /workspace/ll/exclude_recordings.json
```

`index` HEADs the archive, walks it in `--parts` slices with `--workers` threads, checks
that the regular entries form one contiguous chain, pairs each FLAC with its JSON and
writes `WORK/tar_index.jsonl`, `WORK/plan.jsonl` and `WORK/index_summary.json`. Every
HTTP read has a 20 s low-speed guard and 5 retries, so a stalled stream only costs time.

Re-running it is cheap: an existing `tar_index.jsonl` is reused and the JSON/STREAMINFO
of each file is cached in `WORK/meta/` (`<speaker>/<chapter>.json`).

## 2. Encode (one process per GPU)

```bash
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k uv run hohere librilight encode \
      --plan /workspace/ll/plan.jsonl --part $k --parts 4 \
      --out /workspace/ll/part-$k 2>&1 | tee /workspace/ll/part-$k.log &
done
wait
```

`assign_parts` gives a whole speaker to exactly one part, so nobody downloads a FLAC
twice. Downloads, `ffmpeg` and Mimi form a bounded pipeline (at most 8 files in memory),
segments are batched across files by length (16 items, 3000 padded frames by default),
and a finished file is handed to a pool of 8 writer threads, so the GPU never waits for a
`torch.save`. Token files are renamed into place without an fsync;
`WORK/part-K/done/<chapter>.json` is written — and fsynced — only once every token file
of that chapter is on disk. It is the resume marker: restart the same command after a
crash and finished files are skipped. One JSON progress line goes to stdout every 30 s.

Mimi is checked at start-up: digital silence must encode to
`[1049, 1700, 1626, 546, 306, 1443, 1871, 2008]`, otherwise the run refuses to start,
because the silent channel of every token file is built from those codes.

## 3. Finalize (CPU)

```bash
uv run hohere librilight finalize --work /workspace/ll --repo <namespace>/librilight-medium
```

`finalize` merges the per-part manifests, runs the acceptance checks (ids, chapter
numbering, exclusions, checksums, tensor shapes and the silent channel) and writes
`release/checks.json` either way. Only a passing release is packed:
`release/<split>/shard-0000.tar, …` of about 1 GB, `manifest.jsonl` and
`silence_codes.json`. With `--repo` the folder is uploaded in commits of at most 20
files, retrying on 429; the token comes from `$HF_TOKEN`.
