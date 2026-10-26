"""Offline, deterministic dialogue generator for tests and dry runs. No model needed."""

from __future__ import annotations

import numpy as np

from hohere.generators.base import normalize_raw_dialogue
from hohere.schemas.dialogue import ConversationSpec, Dialogue, Skill

_WORDS = """
about after again almost already always another answer around because before better
between bring busy call careful change check choose clear close cold come could
different dinner doing early easy enough evening every exactly family famous feel
finally find first follow forget friend garden give going good great happen heard help
honest house idea important instead kitchen know later leave letter light little long
maybe meeting minute money morning movie never next nothing number office often open
order outside paper people perfect phone place plan point pretty probably problem quick
quiet ready really reason remember right road same school second seems should simple
small something sometimes soon sorry start still story street sure table talk team thing
think through ticket today together tomorrow train travel trying usually village visit
wait walk want weather week weekend window without wonder work worried write yesterday
young
""".split()  # noqa: SIM905
_BACKCHANNELS = ["mm-hm", "right", "yeah", "uh-huh", "I see", "oh wow", "okay", "sure"]


class MockGenerator:
    name = "mock"

    def __init__(self, language: str = "en", max_words: int = 40) -> None:
        self.language = language
        self.max_words = max_words

    def _sentence(self, rng: np.random.Generator, lo: int, hi: int) -> str:
        n = int(rng.integers(lo, hi + 1))
        words = [str(w) for w in rng.choice(_WORDS, size=n)]
        words[0] = words[0].capitalize()
        return " ".join(words) + rng.choice([".", "?", "."])

    def generate(
        self, spec: ConversationSpec, attempt: int = 0, feedback: str | None = None
    ) -> Dialogue:
        rng = np.random.default_rng(spec.seed + attempt)
        hi = min(self.max_words, 28)
        turns: list[dict] = []

        def add(speaker: str, text: str, event: str = "normal", ref=None, pause="none"):
            turns.append(
                {
                    "id": f"t{len(turns) + 1}",
                    "speaker": speaker,
                    "text": text,
                    "event": event,
                    "ref": ref,
                    "pause": pause,
                }
            )
            return turns[-1]["id"]

        speaker = "A"
        last_main: str | None = None
        for k in range(spec.target_turns):
            other = "B" if speaker == "A" else "A"
            pause = "none"
            if spec.skill is Skill.TURN_TAKING and k in (2, 5):
                pause = "long"
            if spec.skill is Skill.INTERRUPTION and k in (2, 4) and last_main:
                last_main = add(speaker, self._sentence(rng, 6, hi), "interruption", last_main)
            elif spec.skill is Skill.INCOMPLETE_THOUGHTS and k == 2:
                last_main = add(speaker, self._sentence(rng, 8, 14), "cutoff")
            elif spec.skill is Skill.INCOMPLETE_THOUGHTS and k == 4:
                prev_same = next(t["id"] for t in turns if t["speaker"] == speaker)
                last_main = add(speaker, self._sentence(rng, 6, hi), "self_correction", prev_same)
            elif spec.skill is Skill.CONTEXT_MEMORY and k >= 4 and k % 3 == 1:
                last_main = add(speaker, self._sentence(rng, 6, hi), "normal", turns[k - 4]["id"])
            else:
                lo = 14 if spec.skill in (Skill.BACKCHANNEL, Skill.INTERRUPTION) else 4
                last_main = add(speaker, self._sentence(rng, lo, max(lo, hi)), pause=pause)
            if spec.skill is Skill.BACKCHANNEL and k % 2 == 0:
                add(other, str(rng.choice(_BACKCHANNELS)), "backchannel", last_main)
            speaker = other

        raw = {
            "objective": f"discuss {spec.topic}",
            "roles": {"A": "user", "B": "AI voice assistant"},
            "turns": turns,
        }
        info = {"provider": "mock", "model": "mock-v1", "attempt": str(attempt)}
        return normalize_raw_dialogue(raw, spec, info, self.language)
