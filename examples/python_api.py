"""Use hohere stages directly from Python (offline, mock providers)."""

from pathlib import Path

from hohere.config.schema import HohereConfig
from hohere.generators.mock import MockGenerator
from hohere.planning import build_plan
from hohere.synthesis import AudioCache, SynthesisRunner
from hohere.synthesis.mock import MockSynthesizer
from hohere.timeline import TimelineComposer
from hohere.validation import validate_audio, validate_dialogue

cfg = HohereConfig()
spec = next(s for s in build_plan(cfg) if s.skill.value == "interruption")

dialogue = MockGenerator().generate(spec)
assert validate_dialogue(dialogue, cfg) == []

runner = SynthesisRunner(cfg.speech, MockSynthesizer(), AudioCache(Path("data/cache"), 24000))
voices, segments = runner.synthesize_dialogue(dialogue)

conv = TimelineComposer(cfg.timeline).compose(dialogue, segments)
print(conv.conversation_id, f"{conv.duration_seconds:.1f}s", voices.voices)
for e in conv.events[:10]:
    print(f"{e.event_type:15} {e.speaker or '-'} {e.start_sample:>8} {e.end_sample:>8}")
print("audio QC:", validate_audio(conv.channel_a, conv.channel_b, 24000, conv.events, cfg) or "ok")
