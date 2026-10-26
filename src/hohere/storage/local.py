"""Local dataset store. Every artifact is written atomically so any stage can resume.

Layout under `output_dir`:

    plan.json                      planned conversation specs
    splits.json                    immutable split manifest
    dialogues/accepted/<id>.json   validated dialogues
    dialogues/rejected/<id>.json   every rejected attempt, with reasons
    segments/<id>/                 raw synthesized utterances + segments.json
    conversations/<id>/            channel_a.wav, channel_b.wav, events.json, metadata.json,
                                   quality.json
    rejected/<id>.<stage>.json     synthesis / compose / audio rejections
    reports/                       per-run reports and dataset validation reports
    manifest.json                  release manifest
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from hohere.schemas.dataset import ConversationMetadata, EventsFile
from hohere.schemas.dialogue import ConversationSpec, Dialogue, Rejection
from hohere.synthesis.runner import Segment


class SplitManifestConflict(RuntimeError):
    pass


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False) + "\n")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def to_pcm16(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)


def pcm16_to_float(audio: np.ndarray) -> np.ndarray:
    return (audio.astype(np.float32) / 32768.0).astype(np.float32)


def write_wav_pcm16(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{os.getpid()}.wav")
    sf.write(tmp, pcm, sample_rate, subtype="PCM_16")
    tmp.replace(path)


class DatasetStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- paths -------------------------------------------------------------------

    @property
    def plan_path(self) -> Path:
        return self.root / "plan.json"

    @property
    def splits_path(self) -> Path:
        return self.root / "splits.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def dialogue_path(self, cid: str) -> Path:
        return self.root / "dialogues" / "accepted" / f"{cid}.json"

    def dialogue_rejection_path(self, cid: str) -> Path:
        return self.root / "dialogues" / "rejected" / f"{cid}.json"

    def segments_dir(self, cid: str) -> Path:
        return self.root / "segments" / cid

    def conversation_dir(self, cid: str) -> Path:
        return self.root / "conversations" / cid

    def rejection_path(self, cid: str, stage: str) -> Path:
        return self.root / "rejected" / f"{cid}.{stage}.json"

    # -- plan & splits -----------------------------------------------------------

    def save_plan(self, specs: list[ConversationSpec], splits: dict) -> None:
        if self.splits_path.exists():
            existing = read_json(self.splits_path)
            if existing != splits:
                raise SplitManifestConflict(
                    f"{self.splits_path} already exists with different splits; "
                    "splits are immutable for a dataset — use a new output_dir"
                )
        else:
            write_json(self.splits_path, splits)
        write_json(self.plan_path, [s.model_dump(mode="json") for s in specs])

    def load_plan(self) -> list[ConversationSpec]:
        if not self.plan_path.exists():
            raise FileNotFoundError(f"no plan at {self.plan_path}; run `hohere generate` first")
        return [ConversationSpec.model_validate(x) for x in read_json(self.plan_path)]

    # -- dialogues ---------------------------------------------------------------

    def has_dialogue(self, cid: str) -> bool:
        return self.dialogue_path(cid).exists()

    def save_dialogue(self, d: Dialogue) -> None:
        write_json(self.dialogue_path(d.conversation_id), d.model_dump(mode="json"))

    def load_dialogue(self, cid: str) -> Dialogue:
        return Dialogue.model_validate(read_json(self.dialogue_path(cid)))

    def accepted_dialogue_ids(self) -> list[str]:
        d = self.root / "dialogues" / "accepted"
        return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []

    def save_dialogue_rejections(self, cid: str, rejections: list[Rejection]) -> None:
        write_json(
            self.dialogue_rejection_path(cid), [r.model_dump(mode="json") for r in rejections]
        )

    def dialogue_exhausted(self, cid: str, max_attempts: int) -> bool:
        """True if all attempts for this conversation were already rejected."""
        p = self.dialogue_rejection_path(cid)
        return p.exists() and len(read_json(p)) >= max_attempts and not self.has_dialogue(cid)

    # -- generic rejections ------------------------------------------------------

    def save_rejection(self, rejection: Rejection) -> None:
        write_json(
            self.rejection_path(rejection.conversation_id, rejection.stage),
            rejection.model_dump(mode="json"),
        )

    def is_rejected(self, cid: str, stage: str) -> bool:
        return self.rejection_path(cid, stage).exists()

    def all_rejections(self) -> list[dict]:
        out: list[dict] = []
        for p in sorted((self.root / "dialogues" / "rejected").glob("*.json")):
            out.extend(read_json(p))
        for p in sorted((self.root / "rejected").glob("*.json")):
            out.append(read_json(p))
        return out

    # -- segments ----------------------------------------------------------------

    def has_segments(self, cid: str) -> bool:
        return (self.segments_dir(cid) / "segments.json").exists()

    def save_segments(
        self, cid: str, segments: list[Segment], voices: dict, speeds: dict, sample_rate: int
    ) -> None:
        d = self.segments_dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        for s in segments:
            tmp = d / f".{s.utterance_id}.tmp{os.getpid()}.wav"
            sf.write(tmp, s.audio, sample_rate, subtype="FLOAT")
            tmp.replace(d / f"{s.utterance_id}.wav")
        # segments.json is written last: its presence marks the stage complete.
        write_json(
            d / "segments.json",
            {
                "conversation_id": cid,
                "sample_rate": sample_rate,
                "voices": voices,
                "speeds": speeds,
                "segments": [s.meta() for s in segments],
            },
        )

    def load_segments(self, cid: str) -> tuple[dict, list[Segment]]:
        d = self.segments_dir(cid)
        info = read_json(d / "segments.json")
        segments = []
        for m in info["segments"]:
            audio, _ = sf.read(d / f"{m['utterance_id']}.wav", dtype="float32")
            segments.append(
                Segment(
                    utterance_id=m["utterance_id"],
                    speaker=m["speaker"],
                    text=m["text"],
                    voice=m["voice"],
                    speed=m["speed"],
                    cache_key=m["cache_key"],
                    audio=audio,
                    gain=m.get("gain", 1.0),
                )
            )
        return info, segments

    # -- conversations -----------------------------------------------------------

    def has_conversation(self, cid: str) -> bool:
        return (self.conversation_dir(cid) / "metadata.json").exists()

    def save_conversation(
        self,
        cid: str,
        pcm_a: np.ndarray,
        pcm_b: np.ndarray,
        sample_rate: int,
        events: EventsFile,
        metadata: ConversationMetadata,
        quality: dict,
    ) -> None:
        d = self.conversation_dir(cid)
        write_wav_pcm16(d / "channel_a.wav", pcm_a, sample_rate)
        write_wav_pcm16(d / "channel_b.wav", pcm_b, sample_rate)
        write_json(d / "events.json", events.model_dump(mode="json"))
        write_json(d / "quality.json", quality)
        # metadata.json last: marks the conversation complete.
        write_json(d / "metadata.json", metadata.model_dump(mode="json"))

    def conversation_ids(self) -> list[str]:
        d = self.root / "conversations"
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if (p / "metadata.json").exists())

    def load_metadata(self, cid: str) -> ConversationMetadata:
        return ConversationMetadata.model_validate(
            read_json(self.conversation_dir(cid) / "metadata.json")
        )

    def load_events(self, cid: str) -> EventsFile:
        return EventsFile.model_validate(read_json(self.conversation_dir(cid) / "events.json"))

    def load_channels(self, cid: str) -> tuple[np.ndarray, np.ndarray, int]:
        d = self.conversation_dir(cid)
        a, sr_a = sf.read(d / "channel_a.wav", dtype="int16")
        b, sr_b = sf.read(d / "channel_b.wav", dtype="int16")
        if sr_a != sr_b:
            raise ValueError(f"{cid}: channel sample rates differ")
        return pcm16_to_float(a), pcm16_to_float(b), sr_a
