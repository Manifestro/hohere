"""Pipeline orchestration. Each stage is independently runnable and resumable:
it skips work whose output already exists and never discards rejected items."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from hohere.config.schema import HohereConfig
from hohere.generators import DialogueGenerationError, DialogueGenerator, make_generator
from hohere.planning import build_plan, derive_seed, plan_summary, split_manifest
from hohere.schemas.dataset import ConversationMetadata, EventsFile
from hohere.schemas.dialogue import ConversationSpec, Dialogue, Rejection, RejectionReason
from hohere.storage.local import DatasetStore, pcm16_to_float, read_json, to_pcm16, write_json
from hohere.storage.manifest import build_manifest, software_versions
from hohere.synthesis import (
    AudioCache,
    SpeechSynthesizer,
    SynthesisError,
    SynthesisRunner,
    make_synthesizer,
)
from hohere.timeline.composer import CompositionError, TimelineComposer
from hohere.validation.audio import validate_audio
from hohere.validation.dataset import validate_dataset
from hohere.validation.dialogue import (
    DuplicateIndex,
    dialogue_words,
    repair_dialogue,
    validate_dialogue,
)

log = logging.getLogger("hohere.pipeline")
T = TypeVar("T")


def _progress(items: Iterable[T], total: int, label: str, enabled: bool) -> Iterator[T]:
    if not enabled:
        yield from items
        return
    columns = (
        TextColumn(f"[bold]{label}"), BarColumn(), MofNCompleteColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(),
    )  # fmt: skip
    with Progress(*columns, transient=False) as bar:
        task = bar.add_task(label, total=total)
        for item in items:
            yield item
            bar.advance(task)


class Pipeline:
    def __init__(
        self,
        cfg: HohereConfig,
        generator: DialogueGenerator | None = None,
        synthesizer: SpeechSynthesizer | None = None,
        show_progress: bool = True,
    ) -> None:
        self.cfg = cfg
        self.store = DatasetStore(cfg.dataset.output_dir)
        self._generator = generator
        self._synthesizer = synthesizer
        self.show_progress = show_progress

    @property
    def generator(self) -> DialogueGenerator:
        if self._generator is None:
            self._generator = make_generator(self.cfg)
        return self._generator

    @property
    def synthesizer(self) -> SpeechSynthesizer:
        if self._synthesizer is None:
            self._synthesizer = make_synthesizer(self.cfg)
        return self._synthesizer

    # -- reports -----------------------------------------------------------------

    def _report(self, stage: str, started: float, counts: dict[str, Any], extra=None) -> dict:
        report = {
            "stage": stage,
            "started_at": datetime.fromtimestamp(started, UTC).isoformat(timespec="seconds"),
            "elapsed_seconds": round(time.time() - started, 2),
            "config_fingerprint": self.cfg.fingerprint(),
            "seed": self.cfg.project.seed,
            "software": software_versions(),
            "counts": counts,
            **(extra or {}),
        }
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        write_json(self.store.reports_dir / "runs" / f"{stamp}_{stage}.json", report)
        log.info("stage %s done: %s", stage, counts)
        return report

    # -- plan --------------------------------------------------------------------

    def plan(self) -> list[ConversationSpec]:
        """Build the plan once; afterwards the stored plan is authoritative."""
        if self.store.plan_path.exists():
            return self.store.load_plan()
        specs = build_plan(self.cfg)
        self.store.save_plan(specs, split_manifest(specs))
        write_json(self.store.reports_dir / "plan_summary.json", plan_summary(specs))
        return specs

    # -- generate ----------------------------------------------------------------

    def generate(self, limit: int | None = None) -> dict:
        started = time.time()
        specs = self.plan()
        dc = self.cfg.dialogue
        dup_index = DuplicateIndex(dc.near_duplicate_threshold)
        for cid in self.store.accepted_dialogue_ids():
            dup_index.add(self.store.load_dialogue(cid))
        todo = [
            s
            for s in specs
            if not self.store.has_dialogue(s.conversation_id)
            and not self.store.dialogue_exhausted(s.conversation_id, dc.max_attempts)
        ]
        target_words = self.cfg.dataset.target_words
        accepted_words = [0]
        if target_words is not None:
            # Interleave skills so that stopping at the word target keeps the skill mix.
            rng = np.random.default_rng(derive_seed(self.cfg.project.seed, "generation-order"))
            todo = [todo[i] for i in rng.permutation(len(todo))]
            accepted_words[0] = sum(
                sum(dialogue_words(self.store.load_dialogue(c)).values())
                for c in self.store.accepted_dialogue_ids()
            )
        if limit is not None:
            todo = todo[:limit]
        lock = threading.Lock()
        counts: Counter[str] = Counter(skipped=len(specs) - len(todo))
        reasons: Counter[str] = Counter()

        def target_reached() -> bool:
            return target_words is not None and accepted_words[0] >= target_words

        def work(spec: ConversationSpec) -> str:
            if target_reached():
                return "skipped_target_reached"
            cid = spec.conversation_id
            path = self.store.dialogue_rejection_path(cid)
            history = (
                [Rejection.model_validate(r) for r in read_json(path)] if path.exists() else []
            )
            feedback = None
            for attempt in range(len(history), dc.max_attempts):
                found: list[RejectionReason]
                dialogue: Dialogue | None = None
                try:
                    dialogue = self.generator.generate(spec, attempt=attempt, feedback=feedback)
                    repairs = repair_dialogue(dialogue, self.cfg)
                    if repairs:
                        dialogue.generator["repairs"] = "; ".join(repairs)
                    found = (
                        validate_dialogue(dialogue, self.cfg)
                        if self.cfg.quality.validate_dialogue
                        else []
                    )
                    if not found and dc.llm_judge and hasattr(self.generator, "judge"):
                        found = [
                            RejectionReason(code="llm_judge", message=m)
                            for m in self.generator.judge(dialogue)
                        ]
                except DialogueGenerationError as e:
                    found = [RejectionReason(code="generation_error", message=str(e)[:500])]
                if not found and dialogue is not None:
                    with lock:
                        dup = dup_index.check(dialogue)
                        if dup is None:
                            if target_reached():
                                return "skipped_target_reached"
                            dup_index.add(dialogue)
                            self.store.save_dialogue(dialogue)
                            accepted_words[0] += sum(dialogue_words(dialogue).values())
                            if history:
                                self.store.save_dialogue_rejections(cid, history)
                            return "accepted"
                    found = [dup]
                history.append(
                    Rejection(
                        conversation_id=cid,
                        stage="dialogue",
                        attempt=attempt,
                        reasons=found,
                        payload=dialogue.model_dump(mode="json") if dialogue else None,
                    )
                )
                self.store.save_dialogue_rejections(cid, history)
                with lock:
                    reasons.update(r.code for r in found)
                feedback = "\n".join(f"- {r.code}: {r.message}" for r in found[:8])
            return "rejected"

        with ThreadPoolExecutor(max_workers=dc.concurrency) as pool:
            futures = [pool.submit(work, s) for s in todo]
            for fut in _progress(
                as_completed(futures), len(futures), "generate", self.show_progress
            ):
                try:
                    counts[fut.result()] += 1
                except Exception:
                    log.exception("unexpected generation failure")
                    counts["error"] += 1
        extra = {"rejection_reasons": dict(reasons)}
        if target_words is not None:
            extra |= {"target_words": target_words, "accepted_words": accepted_words[0]}
            counts["accepted_words"] = accepted_words[0]
        return self._report("generate", started, dict(counts), extra)

    # -- synthesize --------------------------------------------------------------

    def synthesize(
        self,
        limit: int | None = None,
        shard: tuple[int, int] | None = None,
        warm_speaker: str | None = None,
    ) -> dict:
        """`shard=(i, n)` processes every n-th accepted dialogue starting at i, so n worker
        processes can share one GPU without overlapping. `warm_speaker` only fills the TTS
        cache with that speaker's turns (no segments are written)."""
        started = time.time()
        sc = self.cfg.speech
        runner = SynthesisRunner(
            sc,
            self.synthesizer,
            AudioCache(Path(self.cfg.dataset.cache_dir) / "tts", sc.sample_rate),
        )
        accepted = self.store.accepted_dialogue_ids()
        if shard is not None:
            i, n = shard
            accepted = accepted[i::n]
        ids = [
            cid
            for cid in accepted
            if not self.store.has_segments(cid) and not self.store.is_rejected(cid, "synthesis")
        ]
        if limit is not None:
            ids = ids[:limit]
        counts: Counter[str] = Counter()
        if warm_speaker is not None:
            for cid in _progress(ids, len(ids), f"warm {warm_speaker}", self.show_progress):
                try:
                    runner.warm_cache(self.store.load_dialogue(cid), warm_speaker)
                    counts["warmed"] += 1
                except SynthesisError as e:  # the full pass will retry and record it
                    log.warning("warm-up failed for %s: %s", cid, e)
                    counts["warm_errors"] += 1
            counts["cache_hits"] = runner.cache_hits
            counts["cache_misses"] = runner.cache_misses
            return self._report(f"warm_{warm_speaker}", started, dict(counts))
        for cid in _progress(ids, len(ids), "synthesize", self.show_progress):
            dialogue = self.store.load_dialogue(cid)
            try:
                assignment, segments = runner.synthesize_dialogue(dialogue)
            except SynthesisError as e:
                self.store.save_rejection(
                    Rejection(
                        conversation_id=cid,
                        stage="synthesis",
                        reasons=[RejectionReason(code="synthesis_failed", message=str(e)[:500])],
                    )
                )
                counts["rejected"] += 1
                continue
            self.store.save_segments(
                cid, segments, assignment.voices, assignment.speeds, sc.sample_rate
            )
            counts["synthesized"] += 1
        counts["cache_hits"] = runner.cache_hits
        counts["cache_misses"] = runner.cache_misses
        return self._report("synthesize", started, dict(counts))

    # -- compose -----------------------------------------------------------------

    def compose(self, limit: int | None = None) -> dict:
        started = time.time()
        sr = self.cfg.speech.sample_rate
        composer = TimelineComposer(self.cfg.timeline, sr)
        ids = [
            cid
            for cid in self.store.accepted_dialogue_ids()
            if self.store.has_segments(cid)
            and not self.store.has_conversation(cid)
            and not self.store.is_rejected(cid, "compose")
        ]
        if limit is not None:
            ids = ids[:limit]
        counts: Counter[str] = Counter()
        reasons: Counter[str] = Counter()
        for cid in _progress(ids, len(ids), "compose", self.show_progress):
            dialogue = self.store.load_dialogue(cid)
            seg_info, segments = self.store.load_segments(cid)
            try:
                conv = composer.compose(dialogue, segments)
            except CompositionError as e:
                self.store.save_rejection(
                    Rejection(
                        conversation_id=cid,
                        stage="compose",
                        reasons=[RejectionReason(code="composition_failed", message=str(e))],
                    )
                )
                counts["rejected"] += 1
                continue
            pcm_a, pcm_b = to_pcm16(conv.channel_a), to_pcm16(conv.channel_b)
            found: list[RejectionReason] = []
            if self.cfg.quality.validate_audio:
                found = validate_audio(
                    pcm16_to_float(pcm_a), pcm16_to_float(pcm_b), sr, conv.events, self.cfg
                )
            status = "failed" if found else "passed"
            reasons.update(r.code for r in found)
            meta = ConversationMetadata(
                id=cid,
                language=dialogue.language,
                sample_rate=sr,
                duration_samples=conv.num_samples,
                duration_seconds=round(conv.num_samples / sr, 6),
                speakers=seg_info["voices"],
                speaker_speeds=seg_info["speeds"],
                scenario=dialogue.scenario_family,
                skill=dialogue.skill,
                topic=dialogue.topic,
                seed=dialogue.seed,
                generator={
                    "dialogue": dialogue.generator.get("provider", "?"),
                    "dialogue_model": dialogue.generator.get("model", "?"),
                    "speech": self.cfg.speech.provider,
                    "speech_model": self.cfg.speech.model
                    if self.cfg.speech.provider != "mock"
                    else "mock-tts-v1",
                    "model_voice_speech": self.cfg.speech.model_provider,
                },
                split=dialogue.split,
                quality_status=status,
                words_a=dialogue_words(dialogue)["A"],
                words_b=dialogue_words(dialogue)["B"],
                config_fingerprint=self.cfg.fingerprint(),
            )
            self.store.save_conversation(
                cid,
                pcm_a,
                pcm_b,
                sr,
                EventsFile(conversation_id=cid, sample_rate=sr, events=conv.events),
                meta,
                {"status": status, "reasons": [r.model_dump() for r in found]},
            )
            if found and self.cfg.quality.reject_invalid:
                self.store.save_rejection(
                    Rejection(conversation_id=cid, stage="audio", reasons=found)
                )
            counts[status] += 1
        return self._report("compose", started, dict(counts), {"failure_reasons": dict(reasons)})

    # -- validate / export -------------------------------------------------------

    def validate(self, audio: bool = True) -> dict:
        started = time.time()
        report = validate_dataset(self.store, self.cfg, audio=audio)
        self._report("validate", started, report["summary"])
        return report

    def manifest(self) -> dict:
        manifest = build_manifest(self.store, self.cfg)
        write_json(self.store.manifest_path, manifest)
        return manifest

    def export(self, out_dir: Path, include_text: bool | None = None) -> dict:
        from hohere.publishing.huggingface import export_dataset

        started = time.time()
        include = self.cfg.publishing.include_text if include_text is None else include_text
        manifest = export_dataset(self.store, self.cfg, Path(out_dir), include_text=include)
        self._report("export", started, manifest["totals"], {"export_dir": str(out_dir)})
        return manifest

    def run(self, limit: int | None = None) -> dict:
        return {
            "generate": self.generate(limit),
            "synthesize": self.synthesize(limit),
            "compose": self.compose(limit),
            "validate": self.validate()["summary"],
        }
