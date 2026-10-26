# hohere generation image for RunPod GPU pods.
#
# Everything slow is baked in: system packages, the project env (Kokoro + torch), a separate
# vLLM env, Kokoro weights and the spaCy model. Qwen weights live on the RunPod global
# volume, so a fresh pod only loads weights and starts generating.
#
# Built by .github/workflows/image.yml; never build locally.

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG VLLM_VERSION=0.32.0
ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    HF_HOME=/opt/hf-cache \
    HOHERE_HOME=/opt/hohere \
    VLLM_ENV=/opt/vllm-env \
    PATH=/root/.local/bin:$PATH

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends espeak-ng tmux \
 && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# vLLM in its own env so its torch pin never clashes with Kokoro's.
RUN uv venv "$VLLM_ENV" --python 3.12 \
 && VIRTUAL_ENV="$VLLM_ENV" uv pip install --no-cache "vllm==${VLLM_VERSION}" ninja

# Project env from the lockfile (deps first for layer caching, then the source).
WORKDIR /opt/hohere
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --frozen --all-extras --no-install-project --no-cache
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN uv sync --frozen --all-extras --no-cache

# Bake Kokoro weights, all voices we use and spaCy's English model (misaki fetches it lazily).
RUN uv run --no-sync python scripts/prefetch_kokoro.py

# RunPod's /start.sh runs /post_start.sh after sshd is up: start vLLM there.
COPY scripts/post_start.sh /post_start.sh
RUN chmod +x /post_start.sh scripts/*.sh
