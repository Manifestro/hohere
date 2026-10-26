"""Structured dialogue schema: what the dialogue generator produces and synthesis consumes."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

Speaker = Literal["A", "B"]


class Skill(StrEnum):
    """Primary conversational skill of a dialogue. Drives the dataset mix."""

    SIMPLE_DIALOGUE = "simple_dialogue"
    CONTEXT_MEMORY = "context_memory"
    TURN_TAKING = "turn_taking"
    BACKCHANNEL = "backchannel"
    INTERRUPTION = "interruption"
    INCOMPLETE_THOUGHTS = "incomplete_thoughts"


class TurnEvent(StrEnum):
    """How a turn relates to the conversation flow, as written by the dialogue generator."""

    NORMAL = "normal"
    BACKCHANNEL = "backchannel"  # short reaction while `ref` is speaking
    OVERLAP = "overlap"  # starts before `ref` ends, both finish
    INTERRUPTION = "interruption"  # starts mid-`ref`, `ref` gets cut off
    CUTOFF = "cutoff"  # speaker abandons this utterance before its natural end
    SELF_CORRECTION = "self_correction"  # revises an earlier statement (`ref`, same speaker)


class PauseHint(StrEnum):
    NONE = "none"
    SHORT = "short"
    LONG = "long"


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    speaker: Speaker
    text: str = Field(min_length=1)
    intent: str = "statement"
    event: TurnEvent = TurnEvent.NORMAL
    ref: str | None = None  # turn this one reacts to / overlaps / corrects / recalls
    pause: PauseHint = PauseHint.NONE  # pause *before* this turn


class Dialogue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    conversation_id: str
    language: str = "en"
    skill: Skill
    scenario_family: str
    topic: str
    objective: str
    roles: dict[Speaker, str]
    turns: list[Turn]
    seed: int
    split: Literal["train", "validation", "test"]
    generator: dict[str, str] = Field(default_factory=dict)

    def turn_map(self) -> dict[str, Turn]:
        return {t.id: t for t in self.turns}


class ConversationSpec(BaseModel):
    """One planned conversation. Fixed before generation so splits are set before synthesis."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    skill: Skill
    topic: str
    scenario_family: str
    split: Literal["train", "validation", "test"]
    seed: int
    target_turns: int


class RejectionReason(BaseModel):
    code: str
    message: str
    turn_id: str | None = None


class Rejection(BaseModel):
    """Machine-readable record of a rejected item. Rejections are always persisted."""

    conversation_id: str
    stage: Literal["dialogue", "synthesis", "compose", "audio"]
    attempt: int = 0
    reasons: list[RejectionReason]
    payload: dict | None = None
