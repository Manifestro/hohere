"""Download Kokoro weights, every configured voice and spaCy's model at image build time.

Runs on CPU: synthesizes one short phrase per voice so every lazy download happens now.
"""

from hohere.config.schema import HohereConfig
from hohere.synthesis.kokoro import KokoroSynthesizer

cfg = HohereConfig().speech
cfg.device = "cpu"
tts = KokoroSynthesizer(cfg)
for voice in cfg.all_voices:
    audio = tts.synthesize("Hello.", voice, 1.0)
    print(f"{voice}: {audio.shape[0]} samples")
