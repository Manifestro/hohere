"""Stand-in for chatterbox_worker.py: same stdin/stdout protocol, writes a short tone."""

import json
import sys

import numpy as np
import soundfile as sf

SR = 24000
print("loading model... (stray library output)", flush=True)
print(json.dumps({"ready": True, "sr": SR, "model": "fake"}), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if req["text"] == "FAIL":
        print(json.dumps({"id": req["id"], "ok": False, "error": "boom"}), flush=True)
        continue
    n = int(SR * 0.1 * max(1, len(req["text"].split())))
    t = np.arange(n) / SR
    sf.write(req["out"], (0.3 * np.sin(2 * np.pi * 330 * t)).astype("float32"), SR)
    print(json.dumps({"id": req["id"], "ok": True, "num_samples": n, "sr": SR}), flush=True)
