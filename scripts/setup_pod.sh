#!/usr/bin/env bash
# GPU pod setup for image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404.
#
# The image already has the matching torch/torchaudio build. Installing `moshi` with
# its dependencies would pull *another* torch and silently break the CUDA build, so
# the model stack goes in with --no-deps on top of the light dependencies it needs.
set -euo pipefail
cd "$(dirname "$0")/.."

python -V
python -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"

# 1. hohere itself (Part A code only; no model deps via [models])
pip install --break-system-packages -e . --no-deps
pip install --break-system-packages numpy soundfile typer huggingface-hub

# 2. model stack, never touching the image's torch
pip install --break-system-packages "moshi==0.2.13" --no-deps
pip install --break-system-packages \
    "sentencepiece" "einops" "safetensors" "huggingface-hub" "soundfile" \
    "numpy<2.3" "pyyaml" "tqdm" "protobuf"
pip install --break-system-packages "silero-vad>=5.1" "scipy>=1.11" "onnxruntime"

# 3. torchaudio must match the installed torch exactly
python - <<'PY'
import subprocess, sys
import torch
try:
    import torchaudio
    ok = torchaudio.__version__.split("+")[0] == torch.__version__.split("+")[0]
except Exception:
    ok = False
if not ok:
    version = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages",
                           "--no-cache-dir",
                           f"torchaudio=={torch.__version__.split('+')[0]}",
                           "--index-url", f"https://download.pytorch.org/whl/cu128"])
    print("installed torchaudio for torch", version)
else:
    print("torchaudio already matches torch")
PY

# 4. ffmpeg (soundfile reads FLAC without it, the models do not need it either)
command -v ffmpeg >/dev/null || apt-get update && apt-get install -y --no-install-recommends ffmpeg

python - <<'PY'
import torch, torchaudio, silero_vad, scipy, moshi
print("torch", torch.__version__, "torchaudio", torchaudio.__version__,
      "silero", silero_vad.__version__, "moshi", getattr(moshi, "__version__", "?"))
print("cuda", torch.cuda.is_available())
PY
