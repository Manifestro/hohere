#!/usr/bin/env bash
# Create the isolated Chatterbox env used for the model (assistant) voice.
# chatterbox-tts pins torch/transformers/numpy, so it cannot live in the hohere env.
# Usage: bash scripts/setup_chatterbox.sh [env-dir]   then export the printed variable.
set -euo pipefail
ENV="${1:-$HOME/.hohere/chatterbox-env}"
export HF_HUB_ENABLE_HF_TRANSFER=0
[ -x "$ENV/bin/python" ] || uv venv "$ENV" --python 3.11 -q
# setuptools<81 still ships pkg_resources, which the Perth watermarker imports.
VIRTUAL_ENV="$ENV" uv pip install -q "chatterbox-tts==0.1.7" "setuptools<81"
echo "export HOHERE_CHATTERBOX_PYTHON=$ENV/bin/python"
