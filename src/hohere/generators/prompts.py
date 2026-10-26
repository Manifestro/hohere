"""Prompt templates for the dialogue generator, one behaviour brief per primary skill."""

from __future__ import annotations

from hohere.schemas.dialogue import ConversationSpec, Skill

SYSTEM_PROMPT = """\
You write realistic, natural spoken English conversations between two people, A and B.
The text will be converted to speech, so write exactly what people would SAY:
contractions, fillers ("um", "uh", "well") where natural, no stage directions,
no emojis, no markdown, no narration, no speaker names inside the text.
Spell out numbers, times and abbreviations the way they are spoken.

Return ONLY a JSON object with this exact shape:
{
  "objective": "<one sentence: what the conversation is trying to achieve>",
  "roles": {"A": "<short role of A>", "B": "<short role of B>"},
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
- "backchannel": a 1-4 word reaction ("mm-hm", "right", "yeah", "oh wow", "I see")
  said WHILE the other speaker is talking. "ref" = the turn it reacts to, which must be
  by the other speaker and must be a long turn (at least 12 words).
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
- Exactly two speakers, A and B. The first turn is "normal" by A.
- Turn ids are t1, t2, t3, ... in order. "ref" must point to an earlier id.
- Backchannels do not take the floor and never replace a real reply: they are extra
  reactions placed inside the other speaker's long turn.
- This is a DIALOGUE, not a monologue: BOTH speakers hold the floor with real "normal"
  turns (questions, opinions, stories, answers). Each speaker has at least a fifth of the
  spoken words, and neither speaker has more than three normal turns in a row.
- Keep each utterance under {max_words} words. Vary sentence length.
- The conversation must be coherent, on-topic, and end naturally.
"""

SKILL_BRIEFS: dict[Skill, str] = {
    Skill.SIMPLE_DIALOGUE: """\
Primary skill: SIMPLE MEANINGFUL DIALOGUE.
Questions and answers, clarifications, short exchanges. Mostly "normal" turns with
clear question -> answer structure and at least one clarification question.
At most one backchannel and no interruptions.""",
    Skill.CONTEXT_MEMORY: """\
Primary skill: CONTEXT AND MEMORY.
Speakers must refer back to specific details mentioned several turns earlier
(names, numbers, places, decisions). Include at least two "normal" turns with intent
"recall" whose "ref" points to a turn at least three turns earlier, and whose text
explicitly reuses the earlier detail.""",
    Skill.TURN_TAKING: """\
Primary skill: TURN-TAKING AND SILENCE.
Vary the rhythm: quick replies, thoughtful hesitations ("hmm... let me think"),
waiting for an answer. Mark at least two turns with "pause": "long" and several with
"short". Use fillers and hesitant starts where a person is thinking.""",
    Skill.BACKCHANNEL: """\
Primary skill: BACKCHANNELS.
Both speakers take turns telling things at length (15-40 words per long turn), asking
each other questions and replying properly. While one speaks, the other reacts with short
backchannels during their speech, and the roles swap: BOTH speakers produce backchannels
and BOTH have long turns of their own. Include at least three "backchannel" turns, each
with "ref" pointing to a long turn of the other speaker.""",
    Skill.INTERRUPTION: """\
Primary skill: INTERRUPTIONS.
Include at least two "interruption" turns: a speaker stops the other's answer to
clarify, correct a misunderstanding, or change direction. The interrupted turn must be
long enough to be interrupted (at least 12 words). Optionally one "overlap" turn.""",
    Skill.INCOMPLETE_THOUGHTS: """\
Primary skill: INCOMPLETE THOUGHTS.
Include at least one "cutoff" turn (a sentence abandoned partway) followed later by the
same speaker resuming or rephrasing it, and at least one "self_correction" turn that
revises an earlier statement of the same speaker.""",
}


def build_system_prompt(max_words: int) -> str:
    return SYSTEM_PROMPT.replace("{max_words}", str(max_words))


def build_user_prompt(spec: ConversationSpec, feedback: str | None = None) -> str:
    lines = [
        SKILL_BRIEFS[spec.skill],
        "",
        f"Topic: {spec.topic}.",
        f"Length: about {spec.target_turns} turns (backchannels not counted).",
        "Invent concrete, specific details (names, places, numbers) that fit the topic.",
    ]
    if feedback:
        lines += ["", "A previous attempt was rejected for these reasons; avoid them:", feedback]
    return "\n".join(lines)


JUDGE_PROMPT = """\
You are a strict reviewer of synthetic spoken dialogues. Given the JSON dialogue, answer
with ONLY a JSON object: {"coherent": true|false, "on_topic": true|false,
"appropriate_responses": true|false, "issues": ["<short issue>", ...]}.
A dialogue is coherent if every turn follows sensibly from the context, references to
earlier turns are correct, and interruptions/backchannels make sense where they occur."""
