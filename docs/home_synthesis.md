# Synthesis on a home GPU (Windows + WSL2, RTX 3060 12 GB)

Text is generated on a RunPod GPU server; the audio is synthesized at home. Use WSL2
(Ubuntu inside Windows): it is the same Linux environment the pipeline is tested on, and
CUDA works through the normal Windows NVIDIA driver.

## 1. One-time setup

**Windows (PowerShell as Administrator):**

```powershell
wsl --install -d Ubuntu-24.04
```

Reboot, open "Ubuntu", create a user. Keep the regular NVIDIA Windows driver up to date;
do **not** install a Linux NVIDIA driver inside WSL.

**Ubuntu (WSL):**

```bash
nvidia-smi                      # must show the RTX 3060
sudo apt-get update && sudo apt-get install -y espeak-ng git curl tmux ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh && exec bash
cd ~ && git clone https://github.com/Manifestro/hohere.git && cd hohere   # Linux disk, not /mnt/c
uv sync --extra kokoro --extra hf          # no vLLM at home
bash scripts/setup_chatterbox.sh           # isolated env for the assistant voice
echo 'export HOHERE_CHATTERBOX_PYTHON=$HOME/.hohere/chatterbox-env/bin/python' >> ~/.bashrc
uv run hf auth login                       # token with write access to the manifestro org
```

Copy the assistant reference voice (private, never committed):

```bash
mkdir -p data/voices
cp /mnt/c/Users/<you>/Downloads/telegram_audio.ogg data/voices/assistant_ref.ogg
```

## 2. Each release

On the GPU server (RunPod pod from the hohere image):

```bash
uv run --no-sync hohere generate -c configs/v0.3.yaml
uv run --no-sync hohere dialogues push -c configs/v0.3.yaml
```

At home, inside `tmux` (keeps running if the terminal closes):

```bash
cd ~/hohere && git pull
uv run hohere dialogues pull -c configs/v0.3.yaml
uv run hohere synthesize -c configs/v0.3.yaml --workers 3   # resumable; re-run after any stop
uv run hohere compose    -c configs/v0.3.yaml
uv run hohere validate   -c configs/v0.3.yaml
uv run hohere export     -c configs/v0.3.yaml -o exports/v0.3
uv run hohere publish    -c configs/v0.3.yaml --export-dir exports/v0.3 --tag v0.3.0
```

`--workers 3` first caches all user voices (Kokoro) in one process, then runs three
Chatterbox workers (~3.5 GB VRAM each, ~10.5 GB total). Use `--workers 2` if the same GPU
drives your monitors and you see CUDA out-of-memory errors.

## Windows settings for long runs

- Power plan: never sleep while plugged in (Settings → System → Power).
- `C:\Users\<you>\.wslconfig` — give WSL enough RAM, e.g.:

  ```ini
  [wsl2]
  memory=24GB
  ```

- Keep data inside the WSL filesystem (`~/hohere/data`); `/mnt/c` is much slower.

## Disk space

The TTS cache and per-utterance segments are kept as float32 WAV for exact
reproducibility, plus the final 16-bit two-channel conversations. Rough need: **~1 GB per
hour of dataset** (10 h ≈ 10 GB). For 50 h and above, storage will be slimmed down
before the run.
