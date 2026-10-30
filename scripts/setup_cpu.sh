#!/usr/bin/env bash
# Plain Ubuntu 24.04 CPU box: a uv venv, CPU torch wheels, then the model stack.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# CPU wheels for torch/torchaudio: never the CUDA build, never a second torch
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
uv pip install -e .

# the model stack on top of that torch
uv pip install "silero-vad>=5.1" "scipy>=1.11"
uv pip install "moshi>=0.2.13" --no-deps
uv pip install sentencepiece einops safetensors pyyaml tqdm protobuf

python - <<'PY'
import torch, torchaudio, silero_vad, scipy
print("torch", torch.__version__, "torchaudio", torchaudio.__version__,
      "silero", silero_vad.__version__, "scipy", scipy.__version__)
print("cuda", torch.cuda.is_available(), "threads", torch.get_num_threads())
PY

echo "now: hohere index --splits small --state work/state.db && hohere plan --state work/state.db"
