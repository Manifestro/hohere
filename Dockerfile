# hohere generation image for RunPod GPU pods.
#
# One env holds hohere, Kokoro and vLLM (a single torch; CUDA libs come with the wheels, so
# no CUDA base image is needed; gcc is for Triton's JIT modules). Kokoro weights, voices
# and the spaCy model are baked in.
# Qwen weights live on the RunPod global volume. scripts/start.sh replaces RunPod's start
# script: SSH from $PUBLIC_KEY, then vLLM, then sleep.
#
# Built by .github/workflows/image.yml; never build locally.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    HF_HOME=/opt/hf-cache \
    HOHERE_HOME=/opt/hohere \
    PATH=/opt/hohere/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl git openssh-server espeak-ng tmux rsync gcc libc6-dev \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd \
 && curl -LsSf https://astral.sh/uv/install.sh | sh

# Dependencies first (cached layer), then the source.
WORKDIR /opt/hohere
COPY pyproject.toml uv.lock .python-version README.md LICENSE ./
RUN uv sync --frozen --all-extras --no-install-project
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
RUN uv sync --frozen --all-extras \
 && chmod +x scripts/*.sh

# Bake Kokoro weights and every voice we use.
RUN uv run --no-sync python scripts/prefetch_kokoro.py

CMD ["/opt/hohere/scripts/start.sh"]
