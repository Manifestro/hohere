"""Chatterbox TTS (voice cloning) via an isolated worker process.

chatterbox-tts pins torch/transformers/numpy versions incompatible with the hohere env, so
it runs in its own venv (`scripts/setup_chatterbox.sh`) and is driven over stdin/stdout by
`chatterbox_worker.py`. The voice is cloned from `speech.model_reference_audio`.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

from hohere.synthesis.base import SynthesisError

WORKER = Path(__file__).with_name("chatterbox_worker.py")


class ChatterboxSynthesizer:
    sample_rate = 24000

    def __init__(
        self,
        reference_audio: str | Path,
        variant: str = "turbo",
        python: str | None = None,
        device: str | None = None,
        worker_cmd: list[str] | None = None,
    ) -> None:
        self.reference_audio = str(Path(reference_audio).resolve())
        if not Path(self.reference_audio).exists():
            raise SynthesisError(f"reference audio not found: {self.reference_audio}")
        self.variant = variant
        self.name = f"chatterbox-{variant}"
        self.model = (
            "ResembleAI/chatterbox-turbo" if variant == "turbo" else "ResembleAI/chatterbox"
        )
        python = python or os.environ.get("HOHERE_CHATTERBOX_PYTHON")
        if worker_cmd is None:
            if not python:
                raise SynthesisError(
                    "no Chatterbox env: set speech.chatterbox_python or $HOHERE_CHATTERBOX_PYTHON "
                    "(create it with scripts/setup_chatterbox.sh)"
                )
            worker_cmd = [python, str(WORKER), "--variant", variant, "--device", device or "cuda"]
        self._cmd = worker_cmd
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._tmp = tempfile.TemporaryDirectory(prefix="hohere-chatterbox-")
        atexit.register(self.close)

    def _start(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            ready = json.loads(self._read_line())
            if not ready.get("ready"):
                raise SynthesisError(f"worker did not start: {ready}")
            if ready["sr"] != self.sample_rate:
                raise SynthesisError(f"worker sample rate {ready['sr']} != {self.sample_rate}")
        return self._proc

    def _read_line(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise SynthesisError(f"chatterbox worker exited (code {self._proc.poll()})")
        return line

    def synthesize(self, text: str, voice: str, speed: float) -> np.ndarray:
        # Chatterbox has no speed control; the model voice is always synthesized at 1.0.
        with self._lock:
            proc = self._start()
            rid = str(next(self._ids))
            out = os.path.join(self._tmp.name, f"{rid}.wav")
            req = {"id": rid, "text": text, "ref": self.reference_audio, "out": out}
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.flush()
            reply = json.loads(self._read_line())
        if not reply.get("ok"):
            raise SynthesisError(f"chatterbox failed: {reply.get('error')}")
        audio, sr = sf.read(out, dtype="float32")
        os.unlink(out)
        if sr != self.sample_rate:
            raise SynthesisError(f"unexpected sample rate {sr}")
        return audio.reshape(-1)

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()  # type: ignore[union-attr]
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
