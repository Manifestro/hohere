"""Standalone Chatterbox TTS worker, run inside its own venv (chatterbox-tts pins torch,
transformers and numpy versions that cannot share an env with hohere/vLLM).

No hohere imports. Protocol: one JSON request per stdin line, one JSON reply per stdout line.

    request: {"id": "...", "text": "...", "ref": "/path/ref.wav", "out": "/path/out.wav"}
    reply:   {"id": "...", "ok": true, "num_samples": 12345, "sr": 24000}
             {"id": "...", "ok": false, "error": "..."}

The first stdout line is {"ready": true, "sr": ..., "model": ...} once the model is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["turbo", "orig"], default="turbo")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # The protocol gets a private copy of stdout; everything else written to fd 1 (library
    # prints, progress bars, C extensions) is redirected to stderr from here on.
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    import torch
    import torchaudio as ta

    if args.variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS as Model

        model_id = "ResembleAI/chatterbox-turbo"
    else:
        from chatterbox.tts import ChatterboxTTS as Model

        model_id = "ResembleAI/chatterbox"
    model = Model.from_pretrained(device=args.device)
    out.write(json.dumps({"ready": True, "sr": model.sr, "model": model_id}) + "\n")
    out.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        try:
            with torch.inference_mode():
                wav = model.generate(req["text"], audio_prompt_path=req["ref"])
            wav = wav.detach().cpu().float().reshape(1, -1)
            ta.save(req["out"], wav, model.sr)
            reply = {"id": req.get("id"), "ok": True, "num_samples": wav.shape[-1], "sr": model.sr}
        except Exception as e:  # report and keep serving
            traceback.print_exc()
            reply = {"id": req.get("id"), "ok": False, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
