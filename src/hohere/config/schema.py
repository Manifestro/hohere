"""Typed configuration schema. Every tunable of the pipeline lives here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hohere.config.topics import DEFAULT_TOPICS
from hohere.schemas.dialogue import Skill


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(_Strict):
    name: str = "hohere"
    seed: int = 42
    dataset_version: str = "0.1.0"


class SkillMixEntry(_Strict):
    """Share of the dataset target (words, or hours) for dialogues whose *primary* skill is
    this one."""

    share: float = Field(ge=0.0, le=1.0)
    description: str = ""
    # Typical composed length for this skill; falls back to estimated_conversation_seconds.
    estimated_seconds: float | None = Field(default=None, gt=0)
    # Typical words per conversation (both speakers); falls back to
    # estimated_conversation_words.
    estimated_words: float | None = Field(default=None, gt=0)


def _default_skill_mix() -> dict[Skill, SkillMixEntry]:
    return {
        Skill.SIMPLE_DIALOGUE: SkillMixEntry(
            share=0.40, description="Questions, answers, clarifications, short exchanges."
        ),
        Skill.CONTEXT_MEMORY: SkillMixEntry(
            share=0.20, description="Speakers refer back to things said several turns earlier."
        ),
        Skill.TURN_TAKING: SkillMixEntry(
            share=0.15, description="Varied pause lengths, hesitation, waiting for a reply."
        ),
        Skill.BACKCHANNEL: SkillMixEntry(
            share=0.10, description="'uh-huh', 'mm-hm', 'right' while the other speaker talks."
        ),
        Skill.INTERRUPTION: SkillMixEntry(
            share=0.10, description="Stopping an answer, clarifying, changing direction."
        ),
        Skill.INCOMPLETE_THOUGHTS: SkillMixEntry(
            share=0.05, description="Self-correction and resuming an interrupted sentence."
        ),
    }


class DatasetConfig(_Strict):
    target_hours: float = Field(default=20.0, gt=0)
    # When set, the dataset is sized by words (both speakers) instead of hours, and
    # generation stops once accepted dialogues reach it.
    target_words: int | None = Field(default=None, gt=0)
    language: Literal["en"] = "en"
    output_dir: Path = Path("./data/output")
    cache_dir: Path = Path("./data/cache")
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    # Used to turn hours into a conversation count when planning.
    estimated_conversation_seconds: float = Field(default=60.0, gt=0)
    estimated_conversation_words: float = Field(default=170.0, gt=0)
    # Planned conversations = needed * overprovision, to absorb rejections.
    overprovision: float = Field(default=1.15, ge=1.0)
    skill_mix: dict[Skill, SkillMixEntry] = Field(default_factory=_default_skill_mix)
    topics: list[str] = Field(default_factory=lambda: list(DEFAULT_TOPICS), min_length=1)

    @model_validator(mode="after")
    def _check_ratios(self) -> DatasetConfig:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        share = sum(e.share for e in self.skill_mix.values())
        if abs(share - 1.0) > 1e-6:
            raise ValueError(f"skill_mix shares must sum to 1.0, got {share}")
        return self


class DialogueConfig(_Strict):
    provider: Literal["qwen", "mock"] = "qwen"
    model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    # Falls back to $HOHERE_QWEN_BASE_URL / $HOHERE_QWEN_API_KEY when null.
    base_url: str | None = None
    api_key_env: str = "HOHERE_QWEN_API_KEY"
    temperature: float = 0.9
    top_p: float = 0.95
    max_tokens: int = 4096
    request_timeout_s: float = 180.0
    concurrency: int = Field(default=8, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    min_turns: int = Field(default=4, ge=2)
    max_turns: int = Field(default=20, ge=2)
    min_words_per_utterance: int = 1
    max_words_per_utterance: int = 60
    max_backchannel_words: int = 4
    # Speaker balance: a dialogue, not a monologue with backchannels.
    min_main_turns_per_speaker: int = Field(default=2, ge=1)
    min_speaker_word_share: float = Field(default=0.2, ge=0.0, le=0.5)
    max_consecutive_main_turns: int = Field(default=3, ge=1)
    words_per_second: float = 2.6  # for duration estimates before synthesis
    min_duration_seconds: float = 20.0
    max_duration_seconds: float = 180.0
    near_duplicate_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    llm_judge: bool = False  # optional Qwen coherence check

    @model_validator(mode="after")
    def _check_turns(self) -> DialogueConfig:
        if self.min_turns > self.max_turns:
            raise ValueError("min_turns must be <= max_turns")
        if self.min_duration_seconds >= self.max_duration_seconds:
            raise ValueError("min_duration_seconds must be < max_duration_seconds")
        return self


# Speaker B is the AI assistant (the model being trained): one fixed voice, always.
MODEL_VOICE = "af_heart"
# Speaker A is the human user: a different voice per conversation.
DEFAULT_USER_VOICES = [
    "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah", "af_alloy", "af_nova",
    "am_michael", "am_fenrir", "am_puck", "am_echo", "am_eric", "am_liam", "am_onyx",
    "bf_emma", "bf_isabella", "bf_alice", "bm_george", "bm_fable", "bm_lewis", "bm_daniel",
]  # fmt: skip


class SpeechConfig(_Strict):
    provider: Literal["kokoro", "mock"] = "kokoro"
    model: str = "hexgrad/Kokoro-82M"
    sample_rate: Literal[24000] = 24000
    device: str | None = None  # "cuda", "cpu", "mps"; null = auto
    # Speaker B (AI assistant): fixed voice and speed in every conversation.
    model_voice: str = MODEL_VOICE
    model_speed: float = Field(default=1.0, gt=0)
    # TTS for speaker B. "same" uses `provider`; chatterbox_* clone model_reference_audio in
    # an isolated worker env (scripts/setup_chatterbox.sh, $HOHERE_CHATTERBOX_PYTHON).
    model_provider: Literal["same", "chatterbox_turbo", "chatterbox", "mock"] = "same"
    model_reference_audio: Path | None = None
    chatterbox_python: str | None = None
    # Speaker A (human user): voice drawn per conversation, speed in [speed_min, speed_max].
    user_voices: list[str] = Field(default_factory=lambda: list(DEFAULT_USER_VOICES))
    speed_min: float = 0.95
    speed_max: float = 1.05
    # Small extra per-utterance speed variation for the user (the model voice never varies).
    utterance_speed_jitter: float = 0.02
    trim_silence: bool = True
    trim_threshold_db: float = -45.0
    trim_padding_ms: int = 20
    max_retries: int = 3
    batch_size: int = 16

    @model_validator(mode="after")
    def _check_voices(self) -> SpeechConfig:
        if not 0 < self.speed_min <= self.speed_max:
            raise ValueError("require 0 < speed_min <= speed_max")
        if not self.user_voices:
            raise ValueError("user_voices must not be empty")
        if self.model_voice in self.user_voices:
            raise ValueError(f"model_voice {self.model_voice!r} must not be in user_voices")
        if self.model_provider.startswith("chatterbox"):
            if self.model_reference_audio is None:
                raise ValueError("chatterbox model_provider needs model_reference_audio")
            if self.model_speed != 1.0:
                raise ValueError("chatterbox has no speed control: model_speed must be 1.0")
        return self

    @property
    def all_voices(self) -> list[str]:
        return [self.model_voice, *self.user_voices]


class TimelineConfig(_Strict):
    lead_silence_ms: tuple[int, int] = (200, 600)
    tail_silence_ms: tuple[int, int] = (300, 800)
    normal_pause_ms: tuple[int, int] = (200, 800)
    long_pause_ms: tuple[int, int] = (1000, 2500)
    # Probability that an unmarked turn transition gets a long pause anyway.
    random_long_pause_prob: float = Field(default=0.05, ge=0, le=1)
    # Probability that an unmarked transition starts slightly before the previous turn ends.
    transition_overlap_prob: float = Field(default=0.08, ge=0, le=1)
    transition_overlap_ms: tuple[int, int] = (50, 250)
    overlap_ms: tuple[int, int] = (300, 900)
    # Backchannel start, as a fraction of the host utterance.
    backchannel_position: tuple[float, float] = (0.3, 0.75)
    # Interruption start, as a fraction of the interrupted utterance.
    interruption_position: tuple[float, float] = (0.35, 0.75)
    # How long the interrupted speaker keeps talking after being interrupted.
    interrupted_tail_ms: tuple[int, int] = (150, 500)
    # Self-cutoff: the text already trails off; optionally also clip the audio tail.
    cutoff_truncate_audio: bool = True
    cutoff_position: tuple[float, float] = (0.75, 0.95)
    # Silences shorter than this are not annotated as pause events.
    min_pause_ms: int = 100
    # Minimum silence between two utterances of the same speaker on one channel.
    same_channel_gap_ms: int = 60
    fade_ms: int = 8
    cutoff_fade_ms: int = 25
    enable_backchannels: bool = True
    enable_interruptions: bool = True
    enable_overlap: bool = True

    @field_validator(
        "lead_silence_ms", "tail_silence_ms", "normal_pause_ms", "long_pause_ms",
        "transition_overlap_ms", "overlap_ms", "interrupted_tail_ms",
        "backchannel_position", "interruption_position", "cutoff_position",
    )  # fmt: skip
    @classmethod
    def _ordered(cls, v: tuple) -> tuple:
        if v[0] > v[1] or v[0] < 0:
            raise ValueError(f"range must be non-negative and ordered, got {v}")
        return v


class QualityConfig(_Strict):
    validate_audio: bool = True
    validate_dialogue: bool = True
    reject_invalid: bool = True
    clipping_threshold: float = 0.999
    max_clipped_fraction: float = 0.001
    silence_rms_db: float = -50.0
    leakage_rms_db: float = -60.0
    edge_amplitude_max: float = 0.05


class PublishingConfig(_Strict):
    provider: Literal["huggingface"] = "huggingface"
    private: bool = True
    repository_id: str | None = None
    include_text: bool = True
    token_env: str = "HF_TOKEN"


class HohereConfig(_Strict):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    publishing: PublishingConfig = Field(default_factory=PublishingConfig)

    def fingerprint(self) -> str:
        """Stable hash of the full configuration, recorded in reports and manifests."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_config(path: str | Path) -> HohereConfig:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    return HohereConfig.model_validate(data)


def dump_config(cfg: HohereConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
