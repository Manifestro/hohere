"""On-disk dataset schema: events.json and metadata.json for each conversation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hohere.schemas.dialogue import SCHEMA_VERSION, Skill, Speaker


class EventType(StrEnum):
    NORMAL_TURN = "normal_turn"
    SHORT_PAUSE = "short_pause"
    LONG_PAUSE = "long_pause"
    BACKCHANNEL = "backchannel"
    OVERLAP = "overlap"
    INTERRUPTION = "interruption"
    CUTOFF = "cutoff"
    SELF_CORRECTION = "self_correction"


SPEECH_EVENTS = {
    EventType.NORMAL_TURN,
    EventType.BACKCHANNEL,
    EventType.OVERLAP,
    EventType.INTERRUPTION,
    EventType.SELF_CORRECTION,
}
PAUSE_EVENTS = {EventType.SHORT_PAUSE, EventType.LONG_PAUSE}


class Event(BaseModel):
    """A sample-accurate event. Speech events cover exactly the audio placed in a channel.

    `cutoff` marks the fade-out region where an utterance was stopped early; its
    `related_event_id` points to the speech event that was cut (and, for interruptions,
    `cause_event_id` to the interrupting event). Pause events have no speaker.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    speaker: Speaker | None
    utterance_id: str | None
    event_type: EventType
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    related_event_id: str | None = None
    cause_event_id: str | None = None
    text: str | None = None


class EventsFile(BaseModel):
    schema_version: str = SCHEMA_VERSION
    conversation_id: str
    sample_rate: int
    events: list[Event]


class ConversationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    schema_version: str = SCHEMA_VERSION
    language: str
    sample_rate: int
    channels: Literal[2] = 2
    duration_samples: int
    duration_seconds: float
    speakers: dict[Speaker, str]
    speaker_speeds: dict[Speaker, float]
    scenario: str
    skill: Skill
    topic: str
    seed: int
    generator: dict[str, str]
    split: Literal["train", "validation", "test"]
    quality_status: Literal["passed", "failed", "pending"] = "pending"
    config_fingerprint: str | None = None
