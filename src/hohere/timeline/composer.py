"""Conversation Timeline Composer.

Turns a validated dialogue plus its synthesized segments into two synchronized,
sample-accurate mono channels (A -> channel_a, B -> channel_b) and an event list.

All positions are integer sample indices. Placement is fully determined by the
dialogue seed, the timeline config and the segment lengths, so identical inputs
always yield identical timelines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hohere.config.schema import TimelineConfig
from hohere.planning import derive_seed
from hohere.schemas.dataset import Event, EventType
from hohere.schemas.dialogue import Dialogue, PauseHint, Speaker, Turn, TurnEvent
from hohere.synthesis.runner import Segment
from hohere.timeline.events import annotate


class CompositionError(RuntimeError):
    pass


@dataclass
class Placement:
    turn: Turn
    segment: Segment
    start: int
    length: int
    event_type: EventType
    related_turn: str | None = None
    fade_out: int = 0
    cut: bool = False
    cut_cause: str | None = None  # turn id that interrupted this one

    @property
    def speaker(self) -> Speaker:
        return self.turn.speaker

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass
class ComposedConversation:
    conversation_id: str
    sample_rate: int
    channel_a: np.ndarray = field(repr=False)
    channel_b: np.ndarray = field(repr=False)
    events: list[Event]
    placements: list[Placement] = field(repr=False)

    @property
    def num_samples(self) -> int:
        return int(self.channel_a.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / self.sample_rate


class TimelineComposer:
    def __init__(self, cfg: TimelineConfig, sample_rate: int = 24000) -> None:
        self.cfg = cfg
        self.sr = sample_rate

    # -- helpers -----------------------------------------------------------------

    def _ms(self, ms: float) -> int:
        return round(ms * self.sr / 1000)

    def _rand_ms(self, rng: np.random.Generator, rng_ms: tuple[int, int]) -> int:
        lo, hi = self._ms(rng_ms[0]), self._ms(rng_ms[1])
        return int(rng.integers(lo, hi + 1))

    @staticmethod
    def _rand_frac(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
        return float(rng.uniform(bounds[0], bounds[1]))

    # -- placement ---------------------------------------------------------------

    def place(self, dialogue: Dialogue, segments: list[Segment]) -> list[Placement]:
        cfg = self.cfg
        seg_by_id = {s.utterance_id: s for s in segments}
        missing = [t.id for t in dialogue.turns if t.id not in seg_by_id]
        if missing:
            raise CompositionError(f"missing segments for {missing}")

        rng = np.random.default_rng(derive_seed(dialogue.seed, "timeline"))
        gap_same = self._ms(cfg.same_channel_gap_ms)
        floor_end = self._rand_ms(rng, cfg.lead_silence_ms)
        chan_end: dict[Speaker, int] = {"A": 0, "B": 0}
        placed: dict[str, Placement] = {}
        order: list[Placement] = []
        last_main: Placement | None = None

        def commit(p: Placement, holds_floor: bool) -> None:
            nonlocal floor_end, last_main
            placed[p.turn.id] = p
            order.append(p)
            chan_end[p.speaker] = max(chan_end[p.speaker], p.end)
            if holds_floor:
                floor_end = max(floor_end, p.end)
                last_main = p

        def earliest(speaker: Speaker) -> int:
            return chan_end[speaker] + gap_same if chan_end[speaker] else 0

        def sequential_start(turn: Turn) -> int:
            if turn.pause is PauseHint.LONG or rng.random() < cfg.random_long_pause_prob:
                gap = self._rand_ms(rng, cfg.long_pause_ms)
            else:
                gap = self._rand_ms(rng, cfg.normal_pause_ms)
            return max(floor_end + gap, earliest(turn.speaker))

        for turn in dialogue.turns:
            seg = seg_by_id[turn.id]
            length = seg.num_samples
            host = placed.get(turn.ref) if turn.ref else None

            # Backchannel: short reaction inside the host utterance; does not take the floor.
            if turn.event is TurnEvent.BACKCHANNEL:
                if host is not None and host.speaker != turn.speaker and cfg.enable_backchannels:
                    pos = self._rand_frac(rng, cfg.backchannel_position)
                    start = host.start + int(pos * host.length)
                    start = min(start, host.end - length)  # finish inside the host if possible
                    start = max(start, host.start + self._ms(150), earliest(turn.speaker))
                    if start < host.end:
                        commit(
                            Placement(
                                turn, seg, start, length, EventType.BACKCHANNEL, host.turn.id
                            ),
                            holds_floor=False,
                        )
                        continue
                # No room inside the host: say it in the next gap instead.
                start = max(floor_end + self._rand_ms(rng, (80, 250)), earliest(turn.speaker))
                commit(
                    Placement(turn, seg, start, length, EventType.BACKCHANNEL, turn.ref),
                    holds_floor=True,
                )
                continue

            collides = (
                host is not None
                and last_main is not None
                and host is last_main
                and host.speaker != turn.speaker
            )

            # Interruption: start mid-host; host keeps talking briefly, then is cut off.
            if turn.event is TurnEvent.INTERRUPTION and collides and cfg.enable_interruptions:
                pos = self._rand_frac(rng, cfg.interruption_position)
                start = max(host.start + int(pos * host.length), earliest(turn.speaker))
                if start < host.end - self._ms(250):
                    cut_at = min(host.end, start + self._rand_ms(rng, cfg.interrupted_tail_ms))
                    host.length = cut_at - host.start
                    host.cut = True
                    host.cut_cause = turn.id
                    host.fade_out = min(self._ms(cfg.cutoff_fade_ms), host.length // 2)
                    chan_end[host.speaker] = host.end
                    floor_end = host.end
                    commit(
                        Placement(turn, seg, start, length, EventType.INTERRUPTION, host.turn.id),
                        holds_floor=True,
                    )
                    continue

            # Overlap: start shortly before the host ends; both utterances complete.
            if (
                turn.event in (TurnEvent.OVERLAP, TurnEvent.INTERRUPTION)
                and collides
                and cfg.enable_overlap
            ):
                ov = self._rand_ms(rng, cfg.overlap_ms)
                ov = min(ov, host.length // 2, length // 2)
                start = max(host.end - ov, earliest(turn.speaker))
                etype = EventType.OVERLAP if start < host.end else EventType.NORMAL_TURN
                commit(Placement(turn, seg, start, length, etype, host.turn.id), holds_floor=True)
                continue

            # Normal turn, self-correction, cutoff, and fallbacks.
            etype = (
                EventType.SELF_CORRECTION
                if turn.event is TurnEvent.SELF_CORRECTION
                else EventType.NORMAL_TURN
            )
            related = turn.ref if turn.event is TurnEvent.SELF_CORRECTION else None
            start = sequential_start(turn)
            if (
                turn.pause is PauseHint.NONE
                and cfg.enable_overlap
                and last_main is not None
                and last_main.speaker != turn.speaker
                and etype is EventType.NORMAL_TURN
                and rng.random() < cfg.transition_overlap_prob
            ):
                ov = self._rand_ms(rng, cfg.transition_overlap_ms)
                ov = min(ov, last_main.length // 4, length // 4)
                ov_start = max(last_main.end - ov, earliest(turn.speaker))
                if ov_start < last_main.end:
                    start = ov_start
                    etype = EventType.OVERLAP
                    related = last_main.turn.id

            p = Placement(turn, seg, start, length, etype, related)
            if turn.event is TurnEvent.CUTOFF and cfg.cutoff_truncate_audio:
                frac = self._rand_frac(rng, cfg.cutoff_position)
                p.length = max(self._ms(200), int(length * frac))
                p.length = min(p.length, length)
                p.cut = True
                p.fade_out = min(self._ms(cfg.cutoff_fade_ms), p.length // 2)
            elif turn.event is TurnEvent.CUTOFF:
                p.cut = True
                p.fade_out = min(self._ms(cfg.cutoff_fade_ms), p.length // 2)
            commit(p, holds_floor=True)

        return sorted(order, key=lambda p: (p.start, p.turn.id))

    # -- rendering ---------------------------------------------------------------

    def render(self, placements: list[Placement], tail: int) -> tuple[np.ndarray, np.ndarray]:
        total = max(p.end for p in placements) + tail
        channels = {"A": np.zeros(total, np.float32), "B": np.zeros(total, np.float32)}
        fade = self._ms(self.cfg.fade_ms)
        for p in placements:
            audio = p.segment.audio[: p.length].astype(np.float32, copy=True)
            n = audio.shape[0]
            f_in = min(fade, n // 2)
            if f_in:
                audio[:f_in] *= np.linspace(0.0, 1.0, f_in, dtype=np.float32)
            f_out = min(max(fade, p.fade_out), n // 2)
            if f_out:
                audio[-f_out:] *= np.linspace(1.0, 0.0, f_out, dtype=np.float32)
            ch = channels[p.speaker]
            if np.any(ch[p.start : p.start + n]):
                raise CompositionError(f"channel {p.speaker} self-overlap at {p.turn.id}")
            ch[p.start : p.start + n] = audio
        return channels["A"], channels["B"]

    def compose(self, dialogue: Dialogue, segments: list[Segment]) -> ComposedConversation:
        placements = self.place(dialogue, segments)
        rng = np.random.default_rng(derive_seed(dialogue.seed, "timeline-tail"))
        tail = self._rand_ms(rng, self.cfg.tail_silence_ms)
        channel_a, channel_b = self.render(placements, tail)
        events = annotate(
            dialogue.conversation_id,
            placements,
            self.sr,
            self._ms(self.cfg.min_pause_ms),
            self._ms(self.cfg.long_pause_ms[0]),
        )
        return ComposedConversation(
            conversation_id=dialogue.conversation_id,
            sample_rate=self.sr,
            channel_a=channel_a,
            channel_b=channel_b,
            events=events,
            placements=placements,
        )
