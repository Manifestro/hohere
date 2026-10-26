from __future__ import annotations

import itertools

import numpy as np
import pytest

from hohere.schemas.dataset import SPEECH_EVENTS, EventType
from hohere.schemas.dialogue import Skill
from hohere.storage.local import pcm16_to_float, to_pcm16
from hohere.timeline.composer import TimelineComposer
from hohere.validation.audio import validate_audio


def compose(cfg, runner, dialogue):
    _, segments = runner.synthesize_dialogue(dialogue)
    return TimelineComposer(cfg.timeline, cfg.speech.sample_rate).compose(dialogue, segments)


@pytest.mark.parametrize("skill", list(Skill))
def test_channels_are_synchronized_and_valid(cfg, synth_runner, mock_dialogue_factory, skill):
    conv = compose(cfg, synth_runner, mock_dialogue_factory(skill, turns=10))
    assert conv.channel_a.dtype == np.float32
    assert conv.channel_a.shape == conv.channel_b.shape
    a, b = pcm16_to_float(to_pcm16(conv.channel_a)), pcm16_to_float(to_pcm16(conv.channel_b))
    assert validate_audio(a, b, conv.sample_rate, conv.events, cfg) == []


def test_events_match_placed_audio_exactly(cfg, synth_runner, mock_dialogue_factory):
    d = mock_dialogue_factory(Skill.INTERRUPTION, turns=10)
    conv = compose(cfg, synth_runner, d)
    placements = {p.turn.id: p for p in conv.placements}
    channels = {"A": conv.channel_a, "B": conv.channel_b}
    for e in conv.events:
        assert isinstance(e.start_sample, int) and isinstance(e.end_sample, int)
        if e.event_type in SPEECH_EVENTS:
            p = placements[e.utterance_id]
            assert (e.start_sample, e.end_sample) == (p.start, p.end)
            ch = channels[e.speaker]
            assert np.any(ch[e.start_sample : e.end_sample])
            # nothing from this utterance leaks past its annotated end
            if e.end_sample < ch.shape[0]:
                assert ch[e.end_sample] == 0.0 or any(
                    q.start <= e.end_sample < q.end
                    for q in conv.placements
                    if q.speaker == e.speaker
                )


def test_speaker_isolation(cfg, synth_runner, mock_dialogue_factory):
    conv = compose(cfg, synth_runner, mock_dialogue_factory(Skill.BACKCHANNEL, turns=10))
    for speaker, ch in (("A", conv.channel_a), ("B", conv.channel_b)):
        mask = np.zeros(ch.shape[0], bool)
        for p in conv.placements:
            if p.speaker == speaker:
                mask[p.start : p.end] = True
        assert not np.any(ch[~mask])


def test_deterministic_with_same_seed_and_audio(cfg, synth_runner, mock_dialogue_factory):
    d = mock_dialogue_factory(Skill.INTERRUPTION, turns=12)
    c1, c2 = compose(cfg, synth_runner, d), compose(cfg, synth_runner, d)
    assert np.array_equal(c1.channel_a, c2.channel_a)
    assert np.array_equal(c1.channel_b, c2.channel_b)
    assert [e.model_dump() for e in c1.events] == [e.model_dump() for e in c2.events]
    d2 = d.model_copy(update={"seed": d.seed + 1})
    c3 = compose(cfg, synth_runner, d2)
    assert [e.start_sample for e in c3.events] != [e.start_sample for e in c1.events]


def test_interruption_cuts_off_host(cfg, synth_runner, mock_dialogue_factory):
    conv = compose(cfg, synth_runner, mock_dialogue_factory(Skill.INTERRUPTION, turns=10))
    by_id = {e.event_id: e for e in conv.events}
    interruptions = [e for e in conv.events if e.event_type is EventType.INTERRUPTION]
    assert interruptions
    for e in interruptions:
        host = by_id[e.related_event_id]
        assert host.speaker != e.speaker
        assert host.start_sample < e.start_sample < host.end_sample
        cut = [
            c
            for c in conv.events
            if c.event_type is EventType.CUTOFF and c.cause_event_id == e.event_id
        ]
        assert len(cut) == 1 and cut[0].end_sample == host.end_sample
        tail = (host.end_sample - e.start_sample) / conv.sample_rate * 1000
        lo, hi = cfg.timeline.interrupted_tail_ms
        assert lo - 1 <= tail <= hi + 1


def test_backchannels_inside_host(cfg, synth_runner, mock_dialogue_factory):
    conv = compose(cfg, synth_runner, mock_dialogue_factory(Skill.BACKCHANNEL, turns=10))
    by_id = {e.event_id: e for e in conv.events}
    bcs = [e for e in conv.events if e.event_type is EventType.BACKCHANNEL]
    assert bcs
    for e in bcs:
        host = by_id[e.related_event_id]
        assert host.speaker != e.speaker
        assert host.start_sample < e.start_sample < host.end_sample


def test_pauses_are_true_silence(cfg, synth_runner, mock_dialogue_factory):
    cfg.timeline.random_long_pause_prob = 0.0
    conv = compose(cfg, synth_runner, mock_dialogue_factory(Skill.TURN_TAKING, turns=10))
    pauses = [
        e for e in conv.events if e.event_type in (EventType.SHORT_PAUSE, EventType.LONG_PAUSE)
    ]
    assert any(e.event_type is EventType.LONG_PAUSE for e in pauses)
    for e in pauses:
        assert e.speaker is None
        assert not np.any(conv.channel_a[e.start_sample : e.end_sample])
        assert not np.any(conv.channel_b[e.start_sample : e.end_sample])


def test_disabled_events_fall_back_to_sequential(cfg, synth_runner, mock_dialogue_factory):
    cfg.timeline.enable_interruptions = False
    cfg.timeline.enable_overlap = False
    conv = compose(cfg, synth_runner, mock_dialogue_factory(Skill.INTERRUPTION, turns=10))
    kinds = {e.event_type for e in conv.events}
    assert EventType.INTERRUPTION not in kinds and EventType.OVERLAP not in kinds
    spans = sorted(
        (e.start_sample, e.end_sample) for e in conv.events if e.event_type in SPEECH_EVENTS
    )
    for (_, end1), (start2, _) in itertools.pairwise(spans):
        assert start2 >= end1


def test_cutoff_turn_is_truncated(cfg, synth_runner, mock_dialogue_factory):
    d = mock_dialogue_factory(Skill.INCOMPLETE_THOUGHTS, turns=8)
    _, segments = synth_runner.synthesize_dialogue(d)
    conv = TimelineComposer(cfg.timeline).compose(d, segments)
    seg_len = {s.utterance_id: s.num_samples for s in segments}
    cut_turn = next(t for t in d.turns if t.event.value == "cutoff")
    speech = next(
        e for e in conv.events if e.utterance_id == cut_turn.id and e.event_type in SPEECH_EVENTS
    )
    assert speech.end_sample - speech.start_sample < seg_len[cut_turn.id]
    assert any(
        e.event_type is EventType.CUTOFF and e.related_event_id == speech.event_id
        for e in conv.events
    )
