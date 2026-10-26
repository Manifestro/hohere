# Dataset format (schema 1.0)

## Per conversation

```
conversations/<conversation_id>/
├── channel_a.wav      speaker A only, mono PCM16, 24 kHz
├── channel_b.wav      speaker B only, mono PCM16, 24 kHz, same sample count as A
├── events.json        sample-accurate annotations
├── metadata.json      conversation metadata
└── quality.json       QC result and reasons
```

Overlapping speech appears at the same sample positions in both files; voices are never mixed
into one channel. Outside its own utterances a channel is exact digital silence.

## metadata.json

```json
{
  "id": "conv_000001",
  "schema_version": "1.0",
  "language": "en",
  "sample_rate": 24000,
  "channels": 2,
  "duration_samples": 1920000,
  "duration_seconds": 80.0,
  "speakers": {"A": "af_heart", "B": "am_michael"},
  "speaker_speeds": {"A": 0.982, "B": 1.031},
  "scenario": "backchannel/planning_a_weekend_trip",
  "skill": "backchannel",
  "topic": "planning a weekend trip",
  "seed": 1234567,
  "generator": {"dialogue": "qwen", "dialogue_model": "...", "speech": "kokoro", "speech_model": "hexgrad/Kokoro-82M"},
  "split": "train",
  "quality_status": "passed",
  "config_fingerprint": "40e721c50eae1471"
}
```

## events.json

`start_sample` is inclusive, `end_sample` exclusive. Events are sorted by `start_sample`.

| `event_type` | speaker | meaning |
|---|---|---|
| `normal_turn` | A/B | utterance that holds the floor |
| `self_correction` | A/B | utterance revising an earlier one; `related_event_id` → corrected utterance |
| `backchannel` | A/B | short reaction; `related_event_id` → host utterance it overlaps |
| `overlap` | A/B | starts before `related_event_id` ends; both complete |
| `interruption` | A/B | starts inside `related_event_id`, which is then cut off |
| `cutoff` | A/B | fade-out where an utterance stops early; `related_event_id` → that utterance, `cause_event_id` → the interruption (if any) |
| `short_pause` / `long_pause` | null | silence on **both** channels |

Speech events (`normal_turn`, `self_correction`, `backchannel`, `overlap`, `interruption`)
cover exactly the audio placed in the channel. `text` is the source text of the utterance
(the audio of a cut utterance contains only a prefix of it); exports made with `--no-text`
set it to null.

## Working directory

```
plan.json, splits.json           plan and immutable split manifest (by scenario family)
dialogues/accepted|rejected/     validated dialogues and every rejected attempt with reasons
segments/<id>/                   raw synthesized utterances (float32 WAV) + segments.json
rejected/<id>.<stage>.json       synthesis / compose / audio rejections
reports/runs/*.json              per-run report (counts, config fingerprint, versions)
reports/validation.json          dataset-level QC report
reports/logs/hohere.jsonl        structured logs
manifest.json                    release manifest with per-file SHA-256
```

## Hugging Face export

`data/<split>-NNNNN-of-MMMMM.parquet` with columns: `id`, `channel_a` (Audio), `channel_b`
(Audio), `duration_samples`, `duration_seconds`, `sample_rate`, `language`, `skill`,
`scenario`, `topic`, `voice_a`, `voice_b`, `seed`, `events` (list of event structs),
`metadata_json`. The export also contains `README.md` (dataset card), `manifest.json`,
`splits.json`, `generation_config.yaml` and `reports/validation.json`.
