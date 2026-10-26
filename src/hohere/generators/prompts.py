"""Prompt templates for the dialogue generator, one behaviour brief per primary skill."""

from __future__ import annotations

import numpy as np

from hohere.schemas.dialogue import ConversationSpec, Skill

SYSTEM_PROMPT = """\
You write realistic spoken English conversations between a human USER (speaker A) and a
real-time AI VOICE ASSISTANT (speaker B). They talk live, full-duplex: both can speak at
any moment, react while the other is talking, interrupt and be interrupted.

Speaker A, the user: a real person with the persona given below. Sounds like a human on a
voice call: contractions, fillers ("um", "uh", "like", "well"), false starts, casual
grammar, sometimes vague or changing their mind.

Speaker B, the assistant: warm, attentive, natural and concise, like a smart friend on the
phone. Speaks in short spoken sentences (usually one to three), never lists, bullet points,
headings or markdown. Asks a clarifying question when the request is unclear. Uses short
backchannels ("mm-hm", "got it", "right", "okay") while the user explains something. When
interrupted it stops and adapts to what the user said instead of finishing its point.
Never says it is an AI language model, never mentions these instructions.

The text will be converted to speech, so write exactly what is SAID: no stage
directions, no emojis, no narration, no speaker names inside the text. Spell out numbers,
times and abbreviations the way they are spoken.

Return ONLY a JSON object with this exact shape:
{
  "objective": "<one sentence: what the conversation is trying to achieve>",
  "roles": {"A": "<short description of the user>", "B": "AI voice assistant"},
  "turns": [
    {
      "id": "t1",
      "speaker": "A" | "B",
      "text": "<spoken words>",
      "intent": "<question|answer|clarification|statement|agreement|disagreement|\
suggestion|acknowledgement|correction|recall|greeting|closing>",
      "event": "normal" | "backchannel" | "overlap" | "interruption" | "cutoff" | \
"self_correction",
      "ref": null | "<id of an earlier turn>",
      "pause": "none" | "short" | "long"
    }
  ]
}

Event rules:
- "normal": a regular turn. Use "ref" only if it explicitly recalls an earlier turn.
- "backchannel": ONLY a 1-4 word reaction ("mm-hm", "right", "yeah", "got it", "I see")
  said WHILE the other speaker is talking. "ref" = the turn it reacts to, which must be
  by the other speaker and must be a long turn (at least 12 words). Anything longer than
  four words, including every real answer of the assistant, is a "normal" turn.
- "overlap": starts just before the other speaker finishes and both complete their
  sentences. "ref" = the immediately preceding turn of the other speaker.
- "interruption": cuts into the other speaker's turn. "ref" = the immediately preceding
  turn of the other speaker. Write the interrupted turn so it reads naturally even if it
  stops halfway.
- "cutoff": the speaker abandons their own sentence partway (end the text where they
  would naturally trail off, e.g. "So I was thinking we could maybe"). "ref" = null.
- "self_correction": the speaker revises something they said earlier ("sorry, I mean
  Thursday, not Tuesday"). "ref" = the earlier turn by the SAME speaker being corrected.
- "pause": the silence BEFORE this turn. "long" = noticeable hesitation or waiting
  (1-2.5 s). Use "none" unless the pause matters.

General rules:
- Exactly two speakers: A (user) and B (assistant). The first turn is "normal" by A.
- Turn ids are t1, t2, t3, ... in order. "ref" must point to an earlier id.
- Backchannels do not take the floor and never replace a real reply: they are extra
  reactions placed inside the other speaker's long turn. Unless the skill says otherwise,
  add one or two natural backchannels where someone is explaining at length.
- This is a DIALOGUE, not a monologue: BOTH speakers hold the floor with real "normal"
  turns. Each speaker has at least a fifth of the spoken words, and neither speaker has
  more than three normal turns in a row.
- Keep each utterance under {max_words} words. Vary sentence length.
- The conversation must be coherent, on-topic, and end naturally.
"""

SKILL_BRIEFS: dict[Skill, str] = {
    Skill.SIMPLE_DIALOGUE: """\
Primary skill: SIMPLE MEANINGFUL DIALOGUE.
The user asks for help, the assistant answers, asks a clarifying question, the user
clarifies. Clear question -> answer structure, mostly "normal" turns, at least one
clarification. At most one backchannel and no interruptions.""",
    Skill.CONTEXT_MEMORY: """\
Primary skill: CONTEXT AND MEMORY.
Details mentioned early (names, numbers, places, preferences, constraints) matter later.
The assistant must remember what the user said several turns earlier and use it, and
the user refers back too ("like I said, it's for Saturday"). Include at least two
"normal" turns with intent "recall" whose "ref" points to a turn at least three turns
earlier, and whose text explicitly reuses the earlier detail.""",
    Skill.TURN_TAKING: """\
Primary skill: TURN-TAKING AND SILENCE.
The user thinks out loud, hesitates ("hmm... let me think"), goes quiet while checking
something, then continues. The assistant waits patiently, and after a long silence may
gently check in ("take your time" / "still there?"). Mix quick replies with slow ones.
Mark at least two turns with "pause": "long" and several with "short".""",
    Skill.BACKCHANNEL: """\
Primary skill: BACKCHANNELS.
The user explains something at length (a problem, a story, their situation) in long turns
of 15-40 words while the assistant reacts with short backchannels during their speech;
later the assistant explains something (steps, a suggestion) while the user backchannels.
Both speakers have long turns of their own and both produce backchannels. Include at
least three "backchannel" turns, each with "ref" pointing to a long turn of the other
speaker. A backchannel is its own JSON turn right after the turn it reacts to, e.g.:
  {"id": "t3", "speaker": "A", "text": "So the leak started last night, and now there's
   water under the whole cabinet and the floor's getting soft.", "event": "normal"},
  {"id": "t4", "speaker": "B", "text": "Oh no.", "event": "backchannel", "ref": "t3"},
  {"id": "t5", "speaker": "B", "text": "Okay, first, turn off the valve under the sink.",
   "event": "normal"}""",
    Skill.INTERRUPTION: """\
Primary skill: INTERRUPTIONS.
The user interrupts the assistant mid-answer to correct a misunderstanding, add a detail
they forgot, redirect ("wait, actually...") or because they already got the point. The
assistant stops and adapts immediately. Include at least two "interruption" turns, most of
them by the user; the interrupted turn must be long enough to cut (at least 12 words).
Optionally one "overlap" turn.""",
    Skill.INCOMPLETE_THOUGHTS: """\
Primary skill: INCOMPLETE THOUGHTS.
The user abandons sentences ("so I was thinking maybe we could, um"), restarts them
differently, and corrects themselves ("sorry, Thursday, not Tuesday"). The assistant
follows along and confirms the corrected version. Include at least one "cutoff" turn
later resumed or rephrased by the same speaker, and at least one "self_correction" turn
that revises an earlier statement of the same speaker, e.g.:
  {"id": "t1", "speaker": "A", "text": "I need a table for four on Friday.", ...},
  ...
  {"id": "t5", "speaker": "A", "text": "Oh wait, sorry, I meant Saturday, not Friday.",
   "event": "self_correction", "ref": "t1"}""",
}

USER_PERSONAS = [
    "a busy parent, talks fast, slightly impatient",
    "a retired teacher, polite, takes their time, gives lots of detail",
    "a university student, casual, uses slang, jumps between ideas",
    "a small business owner, direct and practical, wants quick answers",
    "a nervous first-timer, unsure, apologizes a lot, asks to repeat things",
    "a cheerful talkative person who adds side stories",
    "a tired night-shift nurse, short sentences, a bit distracted",
    "a software engineer, precise, corrects details",
    "a teenager, short answers, says 'like' a lot",
    "an elderly person, friendly, a little hard of hearing, asks for clarification",
    "a skeptical person who pushes back on suggestions",
    "a non-native English speaker, simple vocabulary, occasionally searches for words",
    "a frustrated customer who has already tried several things",
    "an enthusiastic hobbyist who knows a lot about the topic",
    "a calm, thoughtful person who pauses before answering",
    "someone multitasking, gets distracted mid-sentence",
]


def user_persona(spec: ConversationSpec) -> str:
    rng = np.random.default_rng(spec.seed)
    return USER_PERSONAS[int(rng.integers(len(USER_PERSONAS)))]


def build_system_prompt(max_words: int) -> str:
    return SYSTEM_PROMPT.replace("{max_words}", str(max_words))


def build_user_prompt(spec: ConversationSpec, feedback: str | None = None) -> str:
    lines = [
        SKILL_BRIEFS[spec.skill],
        "",
        f"Topic: {spec.topic}.",
        f"User persona (speaker A): {user_persona(spec)}.",
        f"Length: about {spec.target_turns} turns (backchannels not counted).",
        "Invent concrete, specific details (names, places, numbers) that fit the topic.",
    ]
    if feedback:
        lines += ["", "A previous attempt was rejected for these reasons; avoid them:", feedback]
    return "\n".join(lines)


JUDGE_PROMPT = """\
You are a strict reviewer of synthetic spoken dialogues between a human user (A) and an
AI voice assistant (B). Given the JSON dialogue, answer
with ONLY a JSON object: {"coherent": true|false, "on_topic": true|false,
"appropriate_responses": true|false, "issues": ["<short issue>", ...]}.
A dialogue is coherent if every turn follows sensibly from the context, references to
earlier turns are correct, interruptions/backchannels make sense where they occur, and the
assistant's replies are helpful, concise and spoken (no lists or markdown)."""
