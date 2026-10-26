"""Dialogue generator interface and shared normalization of raw generator output."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from hohere.schemas.dialogue import ConversationSpec, Dialogue, Turn


class DialogueGenerationError(RuntimeError):
    """The provider failed to return a parseable dialogue."""


@runtime_checkable
class DialogueGenerator(Protocol):
    name: str

    def generate(
        self, spec: ConversationSpec, attempt: int = 0, feedback: str | None = None
    ) -> Dialogue: ...


def normalize_raw_dialogue(
    raw: dict[str, Any],
    spec: ConversationSpec,
    generator_info: dict[str, str],
    language: str = "en",
) -> Dialogue:
    """Convert raw generator JSON into a Dialogue with stable utterance ids.

    Raw ids (t1, t2, ...) become `<conversation_id>_u001`, ...; refs are remapped.
    Unknown refs are kept verbatim so the validator rejects them explicitly.
    Raises pydantic.ValidationError / KeyError / TypeError on malformed input.
    """
    raw_turns = raw["turns"]
    if not isinstance(raw_turns, list):
        raise TypeError("'turns' must be a list")
    id_map: dict[str, str] = {}
    for i, t in enumerate(raw_turns, start=1):
        id_map[str(t.get("id", f"t{i}"))] = f"{spec.conversation_id}_u{i:03d}"
    turns = []
    for i, t in enumerate(raw_turns, start=1):
        ref = t.get("ref")
        ref = id_map.get(str(ref), str(ref)) if ref not in (None, "", "null") else None
        turns.append(
            Turn(
                id=f"{spec.conversation_id}_u{i:03d}",
                speaker=t["speaker"],
                text=str(t["text"]).strip(),
                intent=str(t.get("intent") or "statement"),
                event=t.get("event") or "normal",
                ref=ref,
                pause=t.get("pause") or "none",
            )
        )
    return Dialogue(
        conversation_id=spec.conversation_id,
        language=language,
        skill=spec.skill,
        scenario_family=spec.scenario_family,
        topic=spec.topic,
        objective=str(raw.get("objective") or ""),
        roles=raw.get("roles") or {"A": "speaker A", "B": "speaker B"},
        turns=turns,
        seed=spec.seed,
        split=spec.split,
        generator=generator_info,
    )
