"""Structured dialogue validation, run before any audio is synthesized."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

from hohere.config.schema import HohereConfig
from hohere.schemas.dialogue import Dialogue, PauseHint, RejectionReason, Skill, TurnEvent

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_NON_SPOKEN_RE = re.compile(r"[*_#<>\[\]{}|~`]|\(.*?\)")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def estimate_seconds(dialogue: Dialogue, cfg: HohereConfig) -> float:
    """Rough spoken duration before synthesis: floor-holding speech plus pauses."""
    wps = cfg.dialogue.words_per_second
    tl = cfg.timeline
    total = 0.0
    for t in dialogue.turns:
        if t.event is TurnEvent.BACKCHANNEL:
            continue
        total += len(words(t.text)) / wps
        lo, hi = tl.long_pause_ms if t.pause is PauseHint.LONG else tl.normal_pause_ms
        total += (lo + hi) / 2000
    return total


def _main_turns(dialogue: Dialogue):
    return [t for t in dialogue.turns if t.event is not TurnEvent.BACKCHANNEL]


def validate_dialogue(dialogue: Dialogue, cfg: HohereConfig) -> list[RejectionReason]:
    dc, tl = cfg.dialogue, cfg.timeline
    reasons: list[RejectionReason] = []

    def bad(code: str, message: str, turn_id: str | None = None) -> None:
        reasons.append(RejectionReason(code=code, message=message, turn_id=turn_id))

    turns = dialogue.turns
    if not turns:
        bad("empty_dialogue", "dialogue has no turns")
        return reasons

    speakers = {t.speaker for t in turns}
    if speakers != {"A", "B"}:
        bad("speaker_count", f"expected exactly speakers A and B, got {sorted(speakers)}")
    if set(dialogue.roles) != {"A", "B"}:
        bad("roles", "roles must define exactly A and B")

    ids = [t.id for t in turns]
    if len(ids) != len(set(ids)):
        bad("duplicate_turn_id", "turn ids are not unique")

    if turns[0].event is not TurnEvent.NORMAL:
        bad("first_turn", "first turn must be a normal turn", turns[0].id)

    main = _main_turns(dialogue)
    if not dc.min_turns <= len(main) <= dc.max_turns:
        bad("turn_count", f"{len(main)} main turns, allowed [{dc.min_turns}, {dc.max_turns}]")

    index = {t.id: i for i, t in enumerate(turns)}
    by_id = dialogue.turn_map()
    seen_text: dict[str, str] = {}

    for i, t in enumerate(turns):
        n = len(words(t.text))
        if _NON_SPOKEN_RE.search(t.text):
            bad("non_spoken_text", "text contains markup or stage directions", t.id)
        if n < dc.min_words_per_utterance:
            bad("utterance_too_short", f"{n} words", t.id)
        if n > dc.max_words_per_utterance:
            bad("utterance_too_long", f"{n} words > {dc.max_words_per_utterance}", t.id)

        if t.event is not TurnEvent.BACKCHANNEL:
            key = " ".join(words(t.text))
            if key and key in seen_text and n > 3:
                bad("repeated_utterance", f"same text as {seen_text[key]}", t.id)
            seen_text.setdefault(key, t.id)

        ref = by_id.get(t.ref) if t.ref else None
        if t.ref and ref is None:
            bad("unknown_ref", f"ref {t.ref!r} does not exist", t.id)
            continue
        if ref is not None and index[ref.id] >= i:
            bad("forward_ref", "ref must point to an earlier turn", t.id)
            continue

        if t.event is TurnEvent.BACKCHANNEL:
            if n > dc.max_backchannel_words:
                bad("backchannel_too_long", f"{n} words > {dc.max_backchannel_words}", t.id)
            if ref is None:
                bad("backchannel_without_ref", "backchannel needs a host turn", t.id)
            elif ref.speaker == t.speaker:
                bad("backchannel_same_speaker", "backchannel must react to the other speaker", t.id)
            elif len(words(ref.text)) < 8:
                bad("backchannel_host_too_short", "host turn too short to backchannel", t.id)

        if t.event in (TurnEvent.INTERRUPTION, TurnEvent.OVERLAP):
            prev_main = next(
                (p for p in reversed(turns[:i]) if p.event is not TurnEvent.BACKCHANNEL), None
            )
            if ref is None:
                bad(f"{t.event.value}_without_ref", "needs the turn it collides with", t.id)
            elif ref.speaker == t.speaker:
                bad(f"{t.event.value}_same_speaker", "must collide with the other speaker", t.id)
            elif prev_main is None or prev_main.id != ref.id:
                bad(f"{t.event.value}_not_adjacent", "ref must be the previous main turn", t.id)
            elif t.event is TurnEvent.INTERRUPTION and len(words(ref.text)) < 6:
                bad("interrupted_turn_too_short", "interrupted turn is too short", t.id)

        if t.event is TurnEvent.SELF_CORRECTION:
            if ref is None:
                bad("self_correction_without_ref", "must reference the corrected turn", t.id)
            elif ref.speaker != t.speaker:
                bad("self_correction_other_speaker", "must correct the same speaker", t.id)

    est = estimate_seconds(dialogue, cfg)
    if not dc.min_duration_seconds <= est <= dc.max_duration_seconds:
        bad(
            "estimated_duration",
            f"~{est:.0f}s outside [{dc.min_duration_seconds}, {dc.max_duration_seconds}]",
        )

    reasons += _skill_requirements(dialogue, cfg)
    if not tl.enable_backchannels and any(t.event is TurnEvent.BACKCHANNEL for t in turns):
        bad("event_disabled", "backchannels are disabled in the timeline config")
    if not tl.enable_interruptions and any(t.event is TurnEvent.INTERRUPTION for t in turns):
        bad("event_disabled", "interruptions are disabled in the timeline config")
    if not tl.enable_overlap and any(t.event is TurnEvent.OVERLAP for t in turns):
        bad("event_disabled", "overlap is disabled in the timeline config")
    return reasons


def _skill_requirements(dialogue: Dialogue, cfg: HohereConfig) -> list[RejectionReason]:
    turns = dialogue.turns
    counts = Counter(t.event for t in turns)
    index = {t.id: i for i, t in enumerate(turns)}
    out: list[RejectionReason] = []

    def need(ok: bool, message: str) -> None:
        if not ok:
            out.append(RejectionReason(code="skill_requirement", message=message))

    match dialogue.skill:
        case Skill.SIMPLE_DIALOGUE:
            need(counts[TurnEvent.INTERRUPTION] == 0, "simple dialogue must not interrupt")
            need(counts[TurnEvent.BACKCHANNEL] <= 1, "simple dialogue: at most one backchannel")
        case Skill.CONTEXT_MEMORY:
            recalls = [
                t
                for t in turns
                if t.ref
                and t.event is TurnEvent.NORMAL
                and t.ref in index
                and index[t.id] - index[t.ref] >= 3
            ]
            need(len(recalls) >= 1, "context_memory needs a turn recalling >= 3 turns back")
        case Skill.TURN_TAKING:
            need(
                sum(t.pause is PauseHint.LONG for t in turns) >= 1,
                "turn_taking needs at least one long pause",
            )
        case Skill.BACKCHANNEL:
            need(counts[TurnEvent.BACKCHANNEL] >= 2, "backchannel skill needs >= 2 backchannels")
        case Skill.INTERRUPTION:
            need(counts[TurnEvent.INTERRUPTION] >= 1, "interruption skill needs an interruption")
        case Skill.INCOMPLETE_THOUGHTS:
            need(
                counts[TurnEvent.CUTOFF] + counts[TurnEvent.SELF_CORRECTION] >= 1,
                "incomplete_thoughts needs a cutoff or self-correction",
            )
    return out


def _shingles(text: str, n: int = 3) -> set[str]:
    w = words(text)
    if len(w) < n:
        return {" ".join(w)} if w else set()
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def dialogue_text(dialogue: Dialogue) -> str:
    return " ".join(t.text for t in dialogue.turns if t.event is not TurnEvent.BACKCHANNEL)


class DuplicateIndex:
    """Exact and near-duplicate detection over accepted dialogues (word 3-gram Jaccard)."""

    def __init__(self, threshold: float = 0.6) -> None:
        self.threshold = threshold
        self._exact: dict[str, str] = {}
        self._shingles: dict[str, set[str]] = {}
        self._postings: dict[str, list[str]] = defaultdict(list)

    def __len__(self) -> int:
        return len(self._shingles)

    def check(self, dialogue: Dialogue) -> RejectionReason | None:
        text = dialogue_text(dialogue)
        digest = hashlib.sha256(" ".join(words(text)).encode()).hexdigest()
        if digest in self._exact:
            return RejectionReason(
                code="exact_duplicate", message=f"identical to {self._exact[digest]}"
            )
        sh = _shingles(text)
        if not sh:
            return None
        overlap: Counter[str] = Counter()
        for s in sh:
            overlap.update(self._postings.get(s, ()))
        for other, inter in overlap.items():
            jac = inter / (len(sh) + len(self._shingles[other]) - inter)
            if jac >= self.threshold:
                return RejectionReason(
                    code="near_duplicate", message=f"jaccard {jac:.2f} with {other}"
                )
        return None

    def add(self, dialogue: Dialogue) -> None:
        text = dialogue_text(dialogue)
        cid = dialogue.conversation_id
        self._exact[hashlib.sha256(" ".join(words(text)).encode()).hexdigest()] = cid
        sh = _shingles(text)
        self._shingles[cid] = sh
        for s in sh:
            self._postings[s].append(cid)
