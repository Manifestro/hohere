# Frame labels and hohere-bench: how to use them

This guide is for model training and evaluation (e.g. Dauys). It covers:

1. what exists and where;
2. frame labels for training corpora;
3. the evaluation suites and how to score each one;
4. the recommended next steps.

Conventions everywhere: channel **A = user**, channel **B = assistant** (`af_heart`), 24 kHz
mono. **Frames** follow Mimi: 12.5 frames/s, **1 frame = 1920 samples = 80 ms**. Frame `k`
covers samples `[k·1920, (k+1)·1920)`.

---

## 1. Artifacts

| What | Where | Pin |
|---|---|---|
| Training corpus | `manifestro/hohere-1m` (public) | revision `2e3880f30e8725ed242d9d2f71e0794d62ed1795` |
| Frame labels for it | `manifestro/hohere-dialogues` (private), `labels/hohere-1m/` | latest commit of that folder |
| Evaluation suites | `manifestro/hohere-bench` (private) | tag `v0.2.0` |

**Never train on `hohere-bench`.** Its topics (`configs/bench-e1.yaml`) and the user voices
`af_river`, `am_adam`, `af_sky`, `bf_lily` are absent from all training corpora. Keep it
that way in future corpora.

Rebuild commands (on a GPU pod, see `docs/runpod.md`):

```bash
uv run hohere labels --repo-id manifestro/hohere-1m --revision <sha> --out data/labels/hohere-1m
uv run hohere pipeline -c configs/bench-e1.yaml          # E1 store
uv run hohere export   -c configs/bench-e1.yaml -o data/bench/e1
uv run hohere labels   -c configs/bench-e1.yaml --out data/bench/e1/labels
uv run python -m hohere.bench e2 | e4 | e5 | e6 | card | publish --tag vX.Y.Z
```

---

## 2. Frame labels (training corpora)

File: `labels-00000-of-00001.parquet`, one row per conversation. Join on `id` with the
corpus rows.

| Column | Type | Meaning |
|---|---|---|
| `id` | string | conversation id (same as the corpus) |
| `skill` | string | primary skill |
| `num_frames` | int | `ceil(duration_samples / 1920)` |
| `activity_a`, `activity_b` | list\<uint8\> | 1 if ≥ half of the frame is covered by that speaker's speech (backchannels included) |
| `events` | list\<struct\> | turn events, see below |

**Alignment check.** `num_frames` must equal the Mimi code length when the audio is
zero-padded to a multiple of 1920 samples. If your token cache trims instead of padding,
drop the last label frame. Assert the lengths match before training.

### Event rows

All rows have `type`, `speaker`, `start_frame`, `end_frame` (exclusive), `start_sample`,
`end_sample` and `event_id`. Other fields are null unless listed.

| `type` | Meaning | Extra fields |
|---|---|---|
| `turn_end` | a floor-holding turn ends | `next_speaker` (who holds the floor next, or null), `gap_ms` |
| `response_onset` | B takes the floor right after A (A→B transition) | `other` = "A", `gap_ms` (negative = overlapping start) |
| `user_onset` | A takes the floor right after B | `other` = "B", `gap_ms` |
| `interruption` | `speaker` cuts into `other`'s turn | `other`, `yield_ms` = how long `other` kept talking after the interruption started |
| `backchannel` | short reaction by `speaker` | `other` (host speaker), `inside_host` (true = during the host's turn, false = in a gap) |
| `cutoff` | `speaker` stops mid-utterance (fade-out region) | `cause_speaker` (the interrupter, or null for self-abandoned) |

Floor-holding speech means normal turns, overlaps, interruptions and self-corrections.
Backchannels never hold the floor.

### Using the labels in training (spec C3, C8)

```python
import numpy as np
import pyarrow.parquet as pq

labels = {r["id"]: r for r in pq.read_table("labels-00000-of-00001.parquet").to_pylist()}

def transition_mask(row, radius=6):
    """C8a: frames to up-weight (±0.5 s around A turn ends and A interruptions,
    B backchannels plus 3 frames before)."""
    m = np.zeros(row["num_frames"], bool)
    for e in row["events"]:
        if e["type"] == "turn_end" and e["speaker"] == "A" and e["next_speaker"] == "B":
            c = e["end_frame"]
            m[max(0, c - radius) : c + radius] = True
        elif e["type"] == "interruption" and e["speaker"] == "A":
            c = e["start_frame"]
            m[max(0, c - radius) : c + radius] = True
        elif e["type"] == "backchannel" and e["speaker"] == "B":
            m[max(0, e["start_frame"] - 3) : e["end_frame"]] = True
    return m

def onset_windows(row, length=25):
    """C3 / metrics: the first 2 s of each assistant response."""
    return [(e["start_frame"], min(e["start_frame"] + length, e["end_frame"]))
            for e in row["events"] if e["type"] == "response_onset"]
```

### What hohere-1m contains (`stats.json`)

| Measure | Value |
|---|---|
| Backchannels | 1,089, **all by the assistant** |
| Interruptions | 784 user→assistant, 21 assistant→user |
| Assistant keeps talking after being interrupted | median 336 ms (p95 486 ms) |
| A→B response gap | median 491 ms, p95 1.1 s, 8% overlapping starts |

The target behaviours are present but rare in frame terms (well under 1% of frames). That
is why C8 up-weights them. The response gap is slower than human conversation (~200 ms),
so a model trained on this data alone will answer late.

---

## 3. Evaluation suites (`hohere-bench` v0.2.0)

Load any suite with `datasets`:

```python
from datasets import load_dataset
ds = load_dataset("manifestro/hohere-bench", "e5_interruptions", split="test",
                  revision="v0.2.0", token=True)
```

Common columns:
- `id`, `suite`;
- `voice_a`, `voice_a_seen_in_training` (**report seen and unseen separately**);
- `channel_a`, `channel_b` (full length, equal sizes), `total_samples`, `num_frames`;
- `info_json`: texts and provenance, for analysis and judges only; the model never sees it.

### Two scoring modes

**Teacher-forced NLL.** Feed both channels in full and read the model's NLL of channel B
on a frame window. Report CB0 (semantic) separately from CB1–7 (acoustic).

**Teacher-forced up to `t0`, free-running after it.** Feed channel A for all frames. Feed
the reference channel B for frames `< t0`. From `t0` on, the model generates its own B,
and its outputs become its history. Use the deployed decoder (spec C6: depth sampling,
activity hysteresis). "Active" below means the decoder's activity state, not audio
energy. Fix the seed and run at least 3 seeds per row for free-running metrics.

### Per-suite protocol

| Suite (config) | `t0` / window | Metrics |
|---|---|---|
| **E1** `e1_conversations` (+ `e1_labels`) | NLL over all valid frames; windows from labels | CB0 and CB1–7 NLL; **CB0 gap** vs. a same-size train subset; NLL on onset windows; teacher-forced activity F1 |
| **E2** `e2_single_turn` | `t0 = user_start_frame` | **onset latency** = first frame ≥ `user_end_frame` that starts ≥ 2 active frames, minus `user_end_frame`, × 80 ms. Compare with `gap_ms` of the reference. **False start**: ≥ 6 consecutive active frames before `user_end_frame`. **Miss**: no onset within 37 frames (3 s). Intelligibility: ASR WER of the generated answer against `answer_text` (loose, the answer may legitimately differ). Relevance: an LLM judge scores 0–3 given `user_text`. Also teacher-forced NLL on `[answer_start_frame, +25)` with A original vs A silent vs A from another E2 row in the same voice. |
| **E3** `default` | answer frames from `answer_start_sample` | crossed forced choice: correct when NLL(ax\|qx) + NLL(ay\|qy) < NLL(ax\|qy) + NLL(ay\|qx). Chance 50%. By `kind` and `category`. See the dataset card for `compose`. |
| **E4** `e4_memory` | answer frames from `answer_start_frame` | Place each `candidate_i_audio` into channel B at `answer_start_sample` (4 inputs). Predicted = lowest NLL over the candidate's frames. **Accuracy** (chance 25%) by `distance` × `update`. For `update` rows, also the **stale rate** (= `stale_index` chosen). |
| **E5** `e5_interruptions` | `t0 = interrupt_start_frame` | `interrupt` rows: **yield latency** = first frame ≥ `t0` that starts ≥ 3 inactive frames, minus `t0`, × 80 ms. Success when ≤ 500 ms; target ≥ 80% success. `backchannel_control` rows: **false yield** = ≥ 6 consecutive inactive frames within `[t0, min(t0 + 25, assistant_end_frame))`. The reference B after `t0` is the uninterrupted original: never teacher-force past `t0`. |
| **E6** `e6_long_turns` | `t0 = turn_start_frame` (reference B is silent) | **Backchannels**: active B segments of 2–15 frames starting before `turn_end_frame`. Report the rate per minute and the share starting within ±2 frames of a pause window (`pause_start_frames[i]`..`pause_end_frames[i]`). **Takeover**: any active segment > 15 frames before `turn_end_frame`. Optionally, the onset latency after `turn_end_frame`. |

Statistics: bootstrap 95% confidence intervals over rows (E3: over pairs, E1: over
conversations). Use one evaluation config for all compared checkpoints: same decoder, same
seeds, same revisions.

### What the bench does not cover

Static rows cannot react to what the model says. Conversational behaviour across turns
(answering what was actually asked, being interrupted for a reason, continuing a topic)
needs the closed-loop simulator (`hohere sim`, spec §7, not built yet). Treat E2/E5/E6 as
controlled single-event probes.

---

## 4. Next steps

1. **Baseline R0.** Score the v0.1 checkpoint (step 5,306) on all suites with the C6
   decoder. This is the table every later change is compared against.
2. **Track FT.** Fine-tune from R0 with C8 (transition weighting from `transition_mask`,
   activity look-ahead head) on hohere-1m. Compare E2 onset, E5 yield and E6 backchannels
   against R0, plus E1 CB0 NLL as a regression guard.
3. **Part 2 corpus.** Generate after FT shows which timing problems remain. Change the
   gap distribution toward human timing; keep the event directions as they are (the
   stats above). Never use bench voices or topics.
4. **Simulator** (`hohere sim`) for closed-loop acceptance and Stage 3 post-training.
