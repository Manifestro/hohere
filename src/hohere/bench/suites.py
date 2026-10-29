"""Evaluation suites beyond minimal pairs (evaluation only, never trained on).

    E1  held-out conversations       hohere pipeline with configs/bench-e1.yaml (+ labels)
    E2  single turn                  first user turn + first assistant answer of each E1 talk
    E4  memory                       fact -> distractors -> optional update -> question;
                                     four answer candidates for forced choice
    E5  interruptions                a user interjection (or a backchannel control) inserted
                                     into an assistant turn of an E1 talk
    E6  long user turns              one long user turn with natural pauses, no assistant
                                     speech: backchannel opportunities and no-takeover check

Every row holds two full-length channels (A = user, B = assistant reference) and frame
indices at 12.5 Hz (1920 samples per frame), so a model can be teacher-forced up to an
event and run freely after it.
"""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from hohere.bench.build import MODEL_VOICE, SAMPLE_RATE, Speaker
from hohere.labels import FLOOR_EVENTS, FRAME_SAMPLES, num_frames
from hohere.schemas.dataset import Event, EventType
from hohere.synthesis.base import SpeechSynthesizer

SUITES_VERSION = "0.2.0"
# User voices for E1/E2/E4/E5/E6: the first four occur in hohere training corpora,
# the last four in none. configs/bench-e1.yaml uses the same eight.
EVAL_SEEN = ["af_sarah", "am_michael", "bf_emma", "bm_george"]
EVAL_UNSEEN = ["af_river", "am_adam", "af_sky", "bf_lily"]
EVAL_VOICES = EVAL_SEEN + EVAL_UNSEEN


def _frame(sample: int) -> int:
    return sample // FRAME_SAMPLES


def _frame_end(sample: int) -> int:
    return -(-sample // FRAME_SAMPLES)


def _wav(audio: np.ndarray, name: str) -> dict:
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio, np.float32), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return {"bytes": buf.getvalue(), "path": name}


def _base_row(rid: str, suite: str, voice: str, a: np.ndarray, b: np.ndarray, info: dict) -> dict:
    n = len(a)
    assert len(b) == n
    return {
        "id": rid,
        "suite": suite,
        "voice_a": voice,
        "voice_a_seen_in_training": voice in EVAL_SEEN,
        "voice_b": MODEL_VOICE,
        "channel_a": _wav(a, f"{rid}_a.wav"),
        "channel_b": _wav(b, f"{rid}_b.wav"),
        "sample_rate": SAMPLE_RATE,
        "total_samples": n,
        "num_frames": num_frames(n),
        "info_json": json.dumps(info, ensure_ascii=False),
    }


def _floor(events: list[Event]) -> list[Event]:
    return sorted(
        (e for e in events if e.event_type in FLOOR_EVENTS and e.speaker),
        key=lambda e: (e.start_sample, e.event_id),
    )


# -- E2: single turn -------------------------------------------------------------------

E2_TAIL = round(0.5 * SAMPLE_RATE)


def single_turn_rows(store) -> list[dict]:
    """The opening exchange of every E1 conversation that starts with the user.

    Channel A keeps only the first user turn (everything after it is silenced); channel B
    is the reference up to the end of the first answer. Answers that start by
    interrupting are skipped."""
    rows = []
    for cid in store.conversation_ids():
        meta = store.load_metadata(cid)
        events = store.load_events(cid).events
        floor = _floor(events)
        if len(floor) < 2 or floor[0].speaker != "A" or floor[1].speaker != "B":
            continue
        user, answer = floor[0], floor[1]
        if answer.event_type is EventType.INTERRUPTION:
            continue
        a, b, _ = store.load_channels(cid)
        end = min(len(a), answer.end_sample + E2_TAIL)
        a, b = a[:end].copy(), b[:end].copy()
        a[user.end_sample :] = 0.0
        rows.append(
            {
                **_base_row(
                    f"e2-{cid}",
                    "e2_single_turn",
                    meta.speakers["A"],
                    a,
                    b,
                    {
                        "source_id": cid,
                        "topic": meta.topic,
                        "skill": meta.skill.value,
                        "user_text": user.text,
                        "answer_text": answer.text,
                    },
                ),
                "user_start_frame": _frame(user.start_sample),
                "user_end_frame": _frame_end(user.end_sample),
                "answer_start_frame": _frame(answer.start_sample),
                "answer_end_frame": _frame_end(answer.end_sample),
                "gap_ms": round((answer.start_sample - user.end_sample) * 1000 / SAMPLE_RATE),
            }
        )
    return rows


# -- E5: interruptions -----------------------------------------------------------------

INTERJECTIONS = [
    "Wait, sorry, can I ask something?",
    "Hold on a second.",
    "Sorry to cut in, but that's not what I meant.",
    "Wait, wait, go back.",
    "Actually, hang on.",
    "Sorry, can you repeat that?",
    "No, no, that's wrong.",
    "Hang on, I have a question.",
    "Wait, how much did you say?",
    "Stop, stop, I got it.",
    "Excuse me, one thing first.",
    "Hold on, that doesn't sound right.",
]
# Control condition: the user only backchannels; the assistant should keep talking.
CONTROL_BACKCHANNELS = ["Mm-hm.", "Yeah.", "Right.", "Uh-huh.", "Okay."]
E5_MIN_TURN_S = 3.0
E5_POSITION = (0.3, 0.7)
E5_AFTER_S = 2.0  # audio kept after the original end of the assistant turn


def interruption_rows(
    store, tts: SpeechSynthesizer, seed: int = 2026, control_share: float = 0.3
) -> list[dict]:
    """One assistant turn per E1 conversation, with a user interjection (or a control
    backchannel) inserted where the user channel is silent.

    Channel B is the original, uninterrupted assistant audio: teacher-force it only up to
    `interrupt_start_frame`. Channel A is silenced after the inserted phrase."""
    speak = Speaker(tts)
    rng = np.random.default_rng(seed)
    rows = []
    for cid in store.conversation_ids():
        meta = store.load_metadata(cid)
        voice = meta.speakers["A"]
        a, b, _ = store.load_channels(cid)
        turns = [
            e
            for e in _floor(store.load_events(cid).events)
            if e.speaker == "B"
            and (e.end_sample - e.start_sample) >= E5_MIN_TURN_S * SAMPLE_RATE
            and not np.any(a[e.start_sample : e.end_sample])
        ]
        if not turns:
            continue
        turn = turns[int(rng.integers(len(turns)))]
        control = bool(rng.random() < control_share)
        pool = CONTROL_BACKCHANNELS if control else INTERJECTIONS
        text = pool[int(rng.integers(len(pool)))]
        clip = speak(text, voice)
        pos = rng.uniform(*E5_POSITION)
        start = turn.start_sample + int(pos * (turn.end_sample - turn.start_sample))
        end = min(len(a), turn.end_sample + round(E5_AFTER_S * SAMPLE_RATE))
        if start + len(clip) > end:
            continue
        a, b = a[:end].copy(), b[:end].copy()
        a[turn.start_sample :] = 0.0
        a[start : start + len(clip)] = clip
        rows.append(
            {
                **_base_row(
                    f"e5-{cid}",
                    "e5_interruptions",
                    voice,
                    a,
                    b,
                    {
                        "source_id": cid,
                        "topic": meta.topic,
                        "user_text": text,
                        "assistant_text": turn.text,
                    },
                ),
                "condition": "backchannel_control" if control else "interrupt",
                "assistant_start_frame": _frame(turn.start_sample),
                "interrupt_start_frame": _frame(start),
                "interrupt_end_frame": _frame_end(start + len(clip)),
                "assistant_end_frame": _frame_end(turn.end_sample),
            }
        )
    return rows


# -- E4: memory ------------------------------------------------------------------------

OBJECTS = [
    "keys",
    "wallet",
    "passport",
    "glasses",
    "charger",
    "headphones",
    "notebook",
    "watch",
    "umbrella",
    "ticket",
    "remote",
    "scarf",
]
LOCATIONS = [
    "blue box",
    "kitchen drawer",
    "red bag",
    "top shelf",
    "car",
    "coat pocket",
    "bedroom closet",
    "garden shed",
    "desk drawer",
    "laundry basket",
    "green basket",
    "hallway cabinet",
]
FACT = [
    "I put my {o} in the {l}.",
    "Just so you know, my {o} {be} in the {l}.",
    "Remember, I left my {o} in the {l}.",
]
UPDATE = [
    "Oh, I moved my {o} to the {l}.",
    "Actually, my {o} {be} in the {l} now.",
    "Change of plan, I put my {o} in the {l}.",
]
ACK = ["Okay, noted.", "Got it.", "Sure, I'll remember that.", "Alright."]
QUESTION = ["Where did I put my {o}?", "Where {be} my {o}?", "Do you remember where my {o} {be}?"]
ANSWER = "{Pron} in the {l}."
# Small talk that never mentions objects or places from the lists above.
DISTRACTORS = [
    ("What's the weather like today?", "Mild and sunny, around twenty degrees."),
    ("Can you set a timer for ten minutes?", "Timer set for ten minutes."),
    ("What time is it?", "It's a quarter past four."),
    ("Tell me a fun fact.", "Octopuses have three hearts."),
    ("How do you spell necessary?", "N, e, c, e, s, s, a, r, y."),
    ("Play some quiet music.", "Playing something calm now."),
    ("What's seven times eight?", "Seven times eight is fifty-six."),
    ("Remind me to call my mom tonight.", "I'll remind you this evening."),
    ("Is it going to rain tomorrow?", "Probably not, it should stay dry."),
    ("How long does it take to boil an egg?", "About nine minutes for a hard-boiled egg."),
    ("What's a good name for a cat?", "How about Pepper or Miso?"),
    ("Turn the volume down a little.", "Turning it down."),
    ("How many days until Friday?", "Three more days."),
    ("Give me a quick joke.", "Why did the scarecrow win an award? He was outstanding."),
]
_PLURAL = {"keys", "glasses", "headphones"}
E4_DISTANCES = [0, 1, 2, 4, 6, 8]
E4_GAP_S, E4_LEAD_S, E4_TAIL_S = 0.5, 0.5, 0.5


def _fill(template: str, obj: str, loc: str = "") -> str:
    plural = obj in _PLURAL
    return template.format(
        o=obj, l=loc, be="are" if plural else "is", Pron="They're" if plural else "It's"
    )


def memory_items(per_cell: int = 25, seed: int = 2026) -> list[dict]:
    """Deterministic E4 scripts: distance x update cells, `per_cell` items each."""
    rng = np.random.default_rng(seed)
    items = []
    for distance in E4_DISTANCES:
        for update in (False, True):
            for k in range(per_cell):
                obj = OBJECTS[int(rng.integers(len(OBJECTS)))]
                locs = [LOCATIONS[i] for i in rng.choice(len(LOCATIONS), 4, replace=False)]
                first, final = locs[0], (locs[1] if update else locs[0])
                turns = [
                    ("A", _fill(FACT[int(rng.integers(len(FACT)))], obj, first)),
                    ("B", ACK[int(rng.integers(len(ACK)))]),
                ]
                for i in rng.choice(len(DISTRACTORS), distance, replace=False):
                    turns += [("A", DISTRACTORS[i][0]), ("B", DISTRACTORS[i][1])]
                if update:
                    # the update comes after half of the distractors (right away if none)
                    at = 2 + 2 * (distance // 2)
                    turns[at:at] = [
                        ("A", _fill(UPDATE[int(rng.integers(len(UPDATE)))], obj, final)),
                        ("B", ACK[int(rng.integers(len(ACK)))]),
                    ]
                turns.append(("A", _fill(QUESTION[int(rng.integers(len(QUESTION)))], obj)))
                # candidates: correct, the stale location (if updated), then distinct others
                others = [loc for loc in locs if loc not in (final, first)]
                cand = [final, first if update else others.pop(0), *others][:4]
                order = rng.permutation(4)
                candidates = [cand[i] for i in order]
                items.append(
                    {
                        "item_id": f"e4-d{distance}-{'u' if update else 'n'}-{k:03d}",
                        "distance": distance,
                        "update": update,
                        "object": obj,
                        "turns": turns,
                        "update_distance": distance - distance // 2 if update else None,
                        "candidates": [_fill(ANSWER, obj, c) for c in candidates],
                        "candidate_locations": candidates,
                        "correct_index": int(candidates.index(final)),
                        "stale_index": int(candidates.index(first)) if update else None,
                    }
                )
    return items


def memory_rows(
    items: list[dict],
    tts: SpeechSynthesizer,
    seen: list[str] = EVAL_SEEN,
    unseen: list[str] = EVAL_UNSEEN,
) -> list[dict]:
    """Each item in one seen and one unseen user voice (rotating). Sequential turns with
    fixed gaps; channel B ends before the answer slot. The four candidate answers
    (assistant voice) are separate clips placed at `answer_start_sample`."""
    speak = Speaker(tts)
    gap, lead, tail = (round(t * SAMPLE_RATE) for t in (E4_GAP_S, E4_LEAD_S, E4_TAIL_S))
    rows = []
    for k, it in enumerate(items):
        cands = [speak(c, MODEL_VOICE) for c in it["candidates"]]
        for voice in (seen[k % len(seen)], unseen[k % len(unseen)]):
            clips = [(s, speak(t, voice if s == "A" else MODEL_VOICE)) for s, t in it["turns"]]
            pos, placed = lead, []
            for s, c in clips:
                placed.append((s, pos, c))
                pos += len(c) + gap
            answer_start = pos
            total = answer_start + max(len(c) for c in cands) + tail
            a, b = np.zeros(total, np.float32), np.zeros(total, np.float32)
            for s, p, c in placed:
                (a if s == "A" else b)[p : p + len(c)] = c
            rid = f"{it['item_id']}-{voice}"
            row = _base_row(
                rid,
                "e4_memory",
                voice,
                a,
                b,
                {
                    "turns": it["turns"],
                    "candidates": it["candidates"],
                    "candidate_locations": it["candidate_locations"],
                    "object": it["object"],
                    "update_distance": it["update_distance"],
                },
            )
            row.update(
                {
                    "distance": it["distance"],
                    "update": it["update"],
                    "correct_index": it["correct_index"],
                    "stale_index": it["stale_index"],
                    "question_end_frame": _frame_end(placed[-1][1] + len(placed[-1][2])),
                    "answer_start_sample": answer_start,
                    "answer_start_frame": _frame(answer_start),
                    **{
                        f"candidate_{i}_audio": _wav(c, f"{rid}_c{i}.wav")
                        for i, c in enumerate(cands)
                    },
                }
            )
            rows.append(row)
    return rows


# -- E6: long user turns ---------------------------------------------------------------

E6_PAUSE_MS = (300, 700)
E6_MIN_S = 15.0

LONG_TURN_PROMPT = """\
Write what a person says to their voice assistant in ONE long spoken turn: they are
telling a story or explaining a situation about the topic, before asking for help at the
very end. 6 to 9 short spoken sentences, 80 to 120 words in total, plain everyday English,
no lists, no stage directions, no quotes. Persona: {persona}. Topic: {topic}.
Answer with ONLY a JSON object: {{"sentences": ["...", "..."]}}"""


def long_turn_texts(
    dialogue_cfg, topics: list[str], personas: list[str], n: int, seed: int = 2026
) -> list[dict]:
    """Ask the configured OpenAI-compatible endpoint (vLLM) for `n` long user turns."""
    from hohere.generators.qwen import QwenGenerator, extract_json

    gen = QwenGenerator(dialogue_cfg)
    rng = np.random.default_rng(seed)
    out = []
    attempts = 0
    while len(out) < n and attempts < 3 * n:
        attempts += 1
        topic = topics[int(rng.integers(len(topics)))]
        persona = personas[int(rng.integers(len(personas)))]
        prompt = LONG_TURN_PROMPT.format(persona=persona, topic=topic)
        try:
            content, model = gen._chat([{"role": "user", "content": prompt}], seed=seed + attempts)
            sentences = [s.strip() for s in extract_json(content)["sentences"] if s.strip()]
        except Exception:
            continue
        words = sum(len(s.split()) for s in sentences)
        if 6 <= len(sentences) <= 10 and 70 <= words <= 140:
            out.append(
                {
                    "item_id": f"e6-{len(out):03d}",
                    "topic": topic,
                    "persona": persona,
                    "sentences": sentences,
                    "model": model,
                }
            )
    return out


def long_turn_rows(
    items: list[dict], tts: SpeechSynthesizer, voices: list[str], seed: int = 2026
) -> list[dict]:
    """One user voice per item (round-robin). Pauses between sentences are the
    backchannel opportunities; channel B is silent throughout."""
    speak = Speaker(tts)
    rng = np.random.default_rng(seed)
    lead = round(0.5 * SAMPLE_RATE)
    rows = []
    for k, it in enumerate(items):
        voice = voices[k % len(voices)]
        clips = [speak(s, voice) for s in it["sentences"]]
        pos, pauses = lead, []
        segs = []
        for i, c in enumerate(clips):
            segs.append((pos, c))
            pos += len(c)
            if i < len(clips) - 1:
                gap = round(rng.integers(*E6_PAUSE_MS) * SAMPLE_RATE / 1000)
                pauses.append((pos, pos + gap))
                pos += gap
        turn_end = pos
        total = turn_end + round(2.0 * SAMPLE_RATE)
        if turn_end - lead < E6_MIN_S * SAMPLE_RATE:
            continue
        a, b = np.zeros(total, np.float32), np.zeros(total, np.float32)
        for p, c in segs:
            a[p : p + len(c)] = c
        rid = f"{it['item_id']}-{voice}"
        rows.append(
            {
                **_base_row(
                    rid,
                    "e6_long_turns",
                    voice,
                    a,
                    b,
                    {
                        "topic": it["topic"],
                        "persona": it["persona"],
                        "sentences": it["sentences"],
                        "text_model": it["model"],
                    },
                ),
                "turn_start_frame": _frame(lead),
                "turn_end_frame": _frame_end(turn_end),
                "pause_start_frames": [_frame(s) for s, _ in pauses],
                "pause_end_frames": [_frame_end(e) for _, e in pauses],
            }
        )
    return rows


# -- writing ---------------------------------------------------------------------------

_INT_FIELDS = {
    "e2_single_turn": [
        "user_start_frame",
        "user_end_frame",
        "answer_start_frame",
        "answer_end_frame",
        "gap_ms",
    ],
    "e4_memory": [
        "distance",
        "correct_index",
        "stale_index",
        "question_end_frame",
        "answer_start_sample",
        "answer_start_frame",
    ],
    "e5_interruptions": [
        "assistant_start_frame",
        "interrupt_start_frame",
        "interrupt_end_frame",
        "assistant_end_frame",
    ],
    "e6_long_turns": ["turn_start_frame", "turn_end_frame"],
}


def _features(suite: str):
    from datasets import Audio, Features, Sequence, Value

    audio = Audio(sampling_rate=SAMPLE_RATE)
    f = {
        "id": Value("string"),
        "suite": Value("string"),
        "voice_a": Value("string"),
        "voice_a_seen_in_training": Value("bool"),
        "voice_b": Value("string"),
        "channel_a": audio,
        "channel_b": audio,
        "sample_rate": Value("int32"),
        "total_samples": Value("int64"),
        "num_frames": Value("int32"),
        "info_json": Value("string"),
        **{k: Value("int64") for k in _INT_FIELDS[suite]},
    }
    if suite == "e4_memory":
        f["update"] = Value("bool")
        f.update({f"candidate_{i}_audio": audio for i in range(4)})
    if suite == "e5_interruptions":
        f["condition"] = Value("string")
    if suite == "e6_long_turns":
        f["pause_start_frames"] = Sequence(Value("int64"))
        f["pause_end_frames"] = Sequence(Value("int64"))
    return Features(f)


def write_suite(rows: list[dict], suite: str, out_dir: Path, shard_rows: int = 100) -> dict:
    """<out>/<suite>/data/test-NNNNN-of-MMMMM.parquet + summary; returns the summary."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        raise ValueError(f"{suite}: no rows")
    root = Path(out_dir) / suite
    (root / "data").mkdir(parents=True, exist_ok=True)
    for old in (root / "data").glob("*.parquet"):
        old.unlink()
    rows = sorted(rows, key=lambda r: r["id"])
    schema = _features(suite).arrow_schema
    n = (len(rows) + shard_rows - 1) // shard_rows
    for k in range(n):
        chunk = rows[k * shard_rows : (k + 1) * shard_rows]
        pq.write_table(
            pa.Table.from_pylist(chunk, schema=schema),
            root / "data" / f"test-{k:05d}-of-{n:05d}.parquet",
        )
    summary = {
        "suite": suite,
        "version": SUITES_VERSION,
        "rows": len(rows),
        "voices_a": dict(Counter(r["voice_a"] for r in rows)),
        "seconds": round(sum(r["total_samples"] for r in rows) / SAMPLE_RATE, 1),
    }
    if suite == "e4_memory":
        summary["by_distance_update"] = dict(
            Counter(f"d{r['distance']}-{'u' if r['update'] else 'n'}" for r in rows)
        )
    if suite == "e5_interruptions":
        summary["by_condition"] = dict(Counter(r["condition"] for r in rows))
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


# -- card ------------------------------------------------------------------------------

SUITE_DOCS = {
    "e1_conversations": (
        "e1/data/test-*.parquet",
        "Held-out conversations in the hohere format (topics and half of the user voices "
        "appear in no training corpus). Frame labels in config `e1_labels`. Use for NLL "
        "(CB0 vs acoustic codebooks), the CB0 generalization gap, and early stopping.",
    ),
    "e1_labels": (
        "e1/labels/*.parquet",
        "Per-frame activity (12.5 Hz) and turn-event table for E1, joined on `id`.",
    ),
    "e2_single_turn": (
        "e2_single_turn/data/test-*.parquet",
        "Opening user turn + reference answer of each E1 conversation. Teacher-force up to "
        "`user_end_frame`, run freely after: onset latency, intelligibility, relevance; "
        "conditional NLL of the reference answer on the onset window.",
    ),
    "e4_memory": (
        "e4_memory/data/test-*.parquet",
        "Fact, 0-8 distractor exchanges, optional update, question. Four answer candidates "
        "in the assistant voice start at `answer_start_sample`; correct = lowest NLL. "
        "Chance 25%. `stale_index` marks the pre-update answer.",
    ),
    "e5_interruptions": (
        "e5_interruptions/data/test-*.parquet",
        "A user interjection (`interrupt`) or a backchannel (`backchannel_control`) inserted "
        "into an assistant turn. Teacher-force channel B up to `interrupt_start_frame`, run "
        "freely after: yield latency for interrupts, false yields for controls.",
    ),
    "e6_long_turns": (
        "e6_long_turns/data/test-*.parquet",
        "One long user turn (>= 15 s) with pauses between sentences "
        "(`pause_start_frames`/`pause_end_frames`); channel B is silent. Measures assistant "
        "backchannels at pauses and premature takeovers before `turn_end_frame`.",
    ),
}


def bench_card(out_dir: Path, repo_id: str) -> str:
    """Root README for the whole bench: E3 (minimal pairs) under `data/`, other suites in
    their own folders. Lists only suites present in `out_dir`."""
    out = Path(out_dir)
    present = {
        "e1_conversations": (out / "e1" / "data").exists(),
        "e1_labels": (out / "e1" / "labels").exists(),
        **{
            s: (out / s / "data").exists()
            for s in ("e2_single_turn", "e4_memory", "e5_interruptions", "e6_long_turns")
        },
    }
    configs = [
        "  - config_name: default\n    data_files:\n      - split: test\n"
        "        path: data/test-*.parquet"
    ]
    for name, (path, _) in SUITE_DOCS.items():
        if present[name]:
            configs.append(
                f"  - config_name: {name}\n    data_files:\n      - split: test\n"
                f"        path: {path}"
            )
    rows = [
        "| `default` (E3 minimal pairs) | Two questions differing in one word or fact, two "
        "matching answers; crossed forced choice. Chance 50%. |"
    ]
    for name, (_, doc) in SUITE_DOCS.items():
        if present[name]:
            rows.append(f"| `{name}` | {doc} |")
    sizes = []
    for s in ("e2_single_turn", "e4_memory", "e5_interruptions", "e6_long_turns"):
        f = out / s / "summary.json"
        if f.exists():
            d = json.loads(f.read_text())
            sizes.append(f"| `{s}` | {d['rows']} | {d['seconds'] / 3600:.2f} h |")
    configs_yaml = "\n".join(configs)
    rows_md = "\n".join(rows)
    sizes_md = "\n".join(sizes)
    return f"""---
license: cc-by-4.0
language: [en]
pretty_name: hohere-bench
task_categories: [audio-to-audio]
tags: [benchmark, evaluation, full-duplex, spoken-dialogue, synthetic]
configs:
{configs_yaml}
---

# hohere-bench v{SUITES_VERSION}

**Evaluation only. Never train on any part of this repository.** Suites for full-duplex
speech models trained on hohere corpora: channel A = user, channel B = assistant
(`{MODEL_VOICE}`), 24 kHz mono, frame indices at 12.5 Hz (1920 samples per frame).

User voices: {", ".join(EVAL_SEEN)} occur in hohere training corpora;
{", ".join(EVAL_UNSEEN)} occur in none (`voice_a_seen_in_training`). Report results
separately for both groups.

| Config | What it measures |
|---|---|
{rows_md}

| Suite | Rows | Audio |
|---|---:|---:|
{sizes_md}

## Protocol

- **Teacher-forced up to the event, free-running after it** (E2, E5, E6): feed both
  channels up to the event frame, then let the model generate channel B while channel A
  continues from the row.
- **Forced choice** (E3, E4): score each candidate answer by NLL over the answer frames
  only, per codebook, with the semantic codebook reported separately.
- Conversational behaviour in live interaction (reacting to what the model actually said)
  is outside this static bench; use a closed-loop simulator for it.

`info_json` holds texts and provenance for analysis only; models must not see text.

## License

CC-BY-4.0, © Manifestro. Fully synthetic (Kokoro-82M, Apache-2.0; texts from
Qwen3-30B-A3B, Apache-2.0, and templates); no real voices.
"""
