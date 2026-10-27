"""Synthesize the probe set and write it as a Hugging Face dataset (evaluation only).

One row = one probe pair in one user voice. Every row stores four mono clips (question x/y
in the user's voice, answer x/y in the assistant's voice) and one shared timing, so the
four conditions (qx+ax, qx+ay, qy+ax, qy+ay) differ only in content, never in timing:

    channel A: [lead][question ...........][                  silence ]
    channel B: [                 silence  ][gap][answer ........][tail]
                                            ^ answer_start_sample (same for x and y)
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from hohere.bench.probes import Probe
from hohere.synthesis.base import SpeechSynthesizer, trim_silence

BENCH_VERSION = "0.1.0"
SAMPLE_RATE = 24000
MODEL_VOICE = "af_heart"
# Two user voices that appear in hohere training data and two that never do.
SEEN_VOICES = ["af_sarah", "am_michael"]
UNSEEN_VOICES = ["af_river", "am_adam"]
LEAD_S, GAP_S, TAIL_S = 0.5, 0.6, 0.5


def compose(row: dict, question: str, answer: str) -> tuple[np.ndarray, np.ndarray]:
    """Two-channel input for one condition, e.g. compose(row, "x", "y") = qx + ay.

    `row` holds decoded float32 arrays under question_x_audio ... answer_y_audio.
    """
    n = row["total_samples"]
    a, b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    q = np.asarray(row[f"question_{question}_audio"], np.float32)
    ans = np.asarray(row[f"answer_{answer}_audio"], np.float32)
    s = row["question_start_sample"]
    a[s : s + len(q)] = q
    s = row["answer_start_sample"]
    b[s : s + len(ans)] = ans
    return a, b


def _layout(qx: int, qy: int, ax: int, ay: int) -> dict:
    lead, gap, tail = (round(t * SAMPLE_RATE) for t in (LEAD_S, GAP_S, TAIL_S))
    answer_start = lead + max(qx, qy) + gap
    return {
        "question_start_sample": lead,
        "answer_start_sample": answer_start,
        "total_samples": answer_start + max(ax, ay) + tail,
    }


def _wav(audio: np.ndarray, name: str) -> dict:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return {"bytes": buf.getvalue(), "path": name}


def _features():
    from datasets import Audio, Features, Value

    audio = Audio(sampling_rate=SAMPLE_RATE)
    return Features(
        {
            "id": Value("string"),
            "pair_id": Value("string"),
            "kind": Value("string"),
            "category": Value("string"),
            "voice_a": Value("string"),
            "voice_a_seen_in_training": Value("bool"),
            "voice_b": Value("string"),
            "question_x": Value("string"),
            "question_y": Value("string"),
            "answer_x": Value("string"),
            "answer_y": Value("string"),
            "slot_x": Value("string"),
            "slot_y": Value("string"),
            "question_x_audio": audio,
            "question_y_audio": audio,
            "answer_x_audio": audio,
            "answer_y_audio": audio,
            "sample_rate": Value("int32"),
            "question_start_sample": Value("int64"),
            "answer_start_sample": Value("int64"),
            "total_samples": Value("int64"),
        }
    )


class Speaker:
    """Synthesize + trim, memoized (each answer is shared by all user voices)."""

    def __init__(self, tts: SpeechSynthesizer) -> None:
        self.tts = tts
        self._memo: dict[tuple[str, str], np.ndarray] = {}

    def __call__(self, text: str, voice: str) -> np.ndarray:
        key = (text, voice)
        if key not in self._memo:
            raw = np.asarray(self.tts.synthesize(text, voice, 1.0), np.float32)
            clip = trim_silence(raw, SAMPLE_RATE, -45.0, 20)
            peak = float(np.max(np.abs(clip))) if clip.size else 0.0
            if peak < 1e-3:
                raise RuntimeError(f"silent synthesis for {text!r} ({voice})")
            self._memo[key] = clip * min(1.0, 0.89 / peak)
        return self._memo[key]


def build_rows(probes: list[Probe], tts: SpeechSynthesizer, voices: list[str]) -> list[dict]:
    speak = Speaker(tts)
    rows = []
    for p in probes:
        ax, ay = speak(p.answer_x, MODEL_VOICE), speak(p.answer_y, MODEL_VOICE)
        for voice in voices:
            qx, qy = speak(p.question_x, voice), speak(p.question_y, voice)
            rid = f"{p.pair_id}-{voice}"
            rows.append(
                {
                    "id": rid,
                    **p.to_dict(),
                    "voice_a": voice,
                    "voice_a_seen_in_training": voice in SEEN_VOICES,
                    "voice_b": MODEL_VOICE,
                    "question_x_audio": _wav(qx, f"{rid}_qx.wav"),
                    "question_y_audio": _wav(qy, f"{rid}_qy.wav"),
                    "answer_x_audio": _wav(ax, f"{rid}_ax.wav"),
                    "answer_y_audio": _wav(ay, f"{rid}_ay.wav"),
                    "sample_rate": SAMPLE_RATE,
                    **_layout(len(qx), len(qy), len(ax), len(ay)),
                }
            )
    return rows


def write_bench(rows: list[dict], out_dir: Path, repo_id: str) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_features().arrow_schema)
    pq.write_table(table, out_dir / "data" / "test-00000-of-00001.parquet")
    first_voice = [r for r in rows if r["voice_a"] == rows[0]["voice_a"]]
    summary = {
        "version": BENCH_VERSION,
        "rows": len(rows),
        "pairs": len({r["pair_id"] for r in rows}),
        "voices_a": sorted({r["voice_a"] for r in rows}),
        "pairs_by_kind": dict(Counter(r["kind"] for r in first_voice)),
        "pairs_by_category": dict(Counter(r["category"] for r in first_voice)),
        "seconds": round(sum(r["total_samples"] for r in rows) / SAMPLE_RATE, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(card(summary, repo_id))
    return summary


def card(s: dict, repo_id: str) -> str:
    cats = "\n".join(f"| {c} | {n} |" for c, n in s["pairs_by_category"].items())
    return f"""---
license: cc-by-4.0
language: [en]
pretty_name: hohere-bench
task_categories: [audio-to-audio]
tags: [benchmark, evaluation, full-duplex, spoken-dialogue, minimal-pairs, synthetic]
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test-*.parquet
---

# hohere-bench v{s['version']}

**Evaluation only. Never train on this set.** It checks whether a full-duplex speech model
answers *what the user said* or just produces assistant-sounding speech.

Each pair has two user questions that differ in one key word or fact, and the two matching
assistant answers. A model that listens finds answer **x** more likely after question **x**
than after question **y**.

| | |
|---|---:|
| Pairs | {s['pairs']} |
| Rows (pair × user voice) | {s['rows']} |
| User voices (channel A) | {', '.join(s['voices_a'])} |
| Assistant voice (channel B) | {MODEL_VOICE} |
| Audio | 24 kHz mono clips, PCM16 |

- **echo** ({s['pairs_by_kind'].get('echo', 0)} pairs): the answer repeats the differing word
  ("Can you find the red mug?" → "The red mug is on the top shelf.").
- **infer** ({s['pairs_by_kind'].get('infer', 0)} pairs): the answer follows from the meaning
  without repeating it ("It's pouring outside…" → "Yes, take the umbrella…").
- `voice_a_seen_in_training`: {', '.join(SEEN_VOICES)} are hohere training voices,
  {', '.join(UNSEEN_VOICES)} never appear in hohere data.

| Category | Pairs |
|---|---:|
{cats}

## Conditions

All four conditions of a row share one timing (`question_start_sample`,
`answer_start_sample`, `total_samples`), so only the content differs:

```python
import numpy as np
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="test")

def compose(row, q, a):  # q, a in {{"x", "y"}}
    n = row["total_samples"]
    ch_a, ch_b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    qa = row[f"question_{{q}}_audio"]["array"]
    an = row[f"answer_{{a}}_audio"]["array"]
    s = row["question_start_sample"]; ch_a[s:s + len(qa)] = qa
    s = row["answer_start_sample"];  ch_b[s:s + len(an)] = an
    return ch_a, ch_b  # channel A = user, channel B = assistant
```

## Scoring

Teacher-force channel B and sum the NLL over the answer frames only
(from `answer_start_sample` on), ideally per codebook, with the semantic codebook reported
separately. A pair counts as correct when

```
NLL(ax | qx) + NLL(ay | qy)  <  NLL(ax | qy) + NLL(ay | qx)
```

Chance is 50%. Report accuracy overall, per `kind`, per `category`, and for seen vs unseen
user voices. Useful controls on the same rows: channel A silent, and channel A time-reversed.

## License

CC-BY-4.0, © Manifestro. Fully synthetic (Kokoro-82M, Apache-2.0); no real voices.
"""
