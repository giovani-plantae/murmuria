<div align="center">

<img src="assets/icon.svg" width="120" height="120" alt="murmuria" />

# murmuria

**Self-hosted speech-to-text, accelerated on your own GPU.**

[![CI](https://github.com/giovani-plantae/murmuria/actions/workflows/ci.yml/badge.svg)](https://github.com/giovani-plantae/murmuria/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

</div>

A thin, self-hosted server that wraps **Whisper** (through whisper.cpp) and exposes it over
**HTTP** and **WebSocket** — send audio, get text back. GPU-accelerated via **Vulkan** (AMD/Intel)
or **CUDA** (NVIDIA), with a CPU fallback. No cloud, no API keys, no per-minute billing. 🎉

## 🚀 Running

The fastest way to try it: a self-contained image is published to GHCR on every release —
model baked in, nothing to build or download. Two commands and you're up on
`http://localhost:8000`:

```bash
docker pull ghcr.io/giovani-plantae/murmuria:latest
docker run -d --name murmuria --device /dev/dri -p 8000:80 -e MURMURIA_PORT=80 \
    ghcr.io/giovani-plantae/murmuria:latest
```

`--device /dev/dri` hands the GPU to the container (Vulkan). No GPU? Drop that flag and add
`-e MURMURIA_CPU=1`. See [GPU in Docker](#gpu-in-docker) if the GPU isn't picked up.

Then jump to [Try it](#-try-it) to send your first clip.

### 🦀 From source (local dev)

Needs Rust (via [rustup](https://rustup.rs)) and `just` (`cargo install just`):

```bash
just setup          # install native libs (asks for sudo) + download the model (~573 MB)
just run            # serve on :8000 — auto-detects the GPU backend (see below)
just dev            # same, with auto-reload on edits
just doctor         # diagnostics: is the model present? does the backend load?
```

`just run --cpu` forces CPU; extra flags pass through (`just run --port 9000`).
Run `just` on its own to list every recipe.

### 📦 Always-on deploy (Docker Compose)

Builds from source, restarts on reboot, model baked into the image:

```bash
docker compose up -d --build     # build + serve in the background
docker compose logs -f           # follow logs
docker compose down              # stop & remove
```

Prefer the published image (no build)? Use the GHCR compose file — handy for a Portainer stack:

```bash
docker compose -f docker-compose-portainer.yml up -d
```

Both expect the GPU at `/dev/dri`; see [GPU in Docker](#gpu-in-docker) for the one host-specific
setting you may need to adjust.

## 🎮 GPU acceleration

murmuria runs inference on a GPU when it can and falls back to CPU otherwise. **You don't pick the
backend** — it's auto-detected at launch and printed in the logs. This section is about seeing
*what* it picked and *why*.

### How the backend is chosen

`just run` selects the backend in this order and announces it before building:

1. `--cpu` flag or `MURMURIA_CPU=1` → **CPU** (forced)
2. `nvidia-smi` found on `PATH` → **CUDA** (NVIDIA — fastest there)
3. a render node at `/dev/dri` → **Vulkan** (AMD / Intel; also works on NVIDIA)
4. nothing above → **CPU**

You'll see a line like `murmuria → Vulkan backend` from `just`, then on startup the server prints
the active backend:

```
murmuria: loading model … (language: en, backend: GPU (Vulkan))
```

> **NVIDIA note:** building the CUDA backend needs the CUDA toolkit (`nvcc`) installed — `just setup`
> only installs the Vulkan stack. Without the toolkit, NVIDIA cards still work fine through Vulkan
> (they expose `/dev/dri` too).

### What GPU do I have?

After `just setup` (which installs `vulkan-tools`), find out what's on the machine and whether the
graphics driver exposes it:

```bash
vulkaninfo --summary                 # GPUs Vulkan can actually use (look at the "GPU0" names)
lspci | grep -iE 'vga|3d|display'    # every graphics device on the box
nvidia-smi                           # NVIDIA only — if this works, CUDA is available
```

If `vulkaninfo --summary` lists your GPU, the Vulkan backend will use it. If it errors, or the only
device is `llvmpipe` (a software renderer), the driver is missing — install
`mesa-vulkan-drivers` (AMD/Intel) and try again.

### Is the GPU actually being used?

```bash
just doctor          # loads the model, initializes the backend, reports ✓ / ✗
just doctor --cpu    # same, but tests the CPU path explicitly
```

`doctor` prints the same `backend: …` line as `just run`. If it says `CPU` when you expected a GPU,
detection found none (or it was forced off) — re-check the steps above.

### GPU in Docker

Containers reach the GPU through `/dev/dri` (Vulkan). The catch: the container runs as a non-root
user, so it also needs the **host group IDs** that own the `/dev/dri` nodes — and those GIDs differ
per machine. Two options:

- **`just docker-up`** — builds and **auto-detects** `/dev/dri` plus the owning GIDs, then passes
  them through. Easiest, no editing.
- **`docker compose`** — set the GIDs yourself. Find them with `ls -ln /dev/dri`, then list them
  under `group_add:` in the compose file (the defaults assume `render=992`, `video=44`).

No GPU on the host? Remove the `devices:` / `group_add:` block and set `MURMURIA_CPU=1`.

## ⚙️ Configuration

Everything has sensible defaults, so murmuria runs with **no config at all**. To tweak anything,
copy the example file (its values are the defaults, ready to edit) and adjust:

```bash
cp .env.example .env
```

| Variable | Default | What it does |
|---|---|---|
| `MURMURIA_MODEL` | `models/ggml-large-v3-turbo-q5_0.bin` | Path to the ggml model file (`just model` downloads it). |
| `MURMURIA_LANGUAGE` | `en` | Forced transcription language (whisper code: `en`, `pt`, `es`, …). |
| `MURMURIA_PORT` | first free from `8000` | Pins the listen port. Unset, the server scans upward from `8000` for a free one (like `artisan serve`). |
| `MURMURIA_CPU` | _(unset)_ | Set to `1` to force CPU even on a GPU build. |
| `MURMURIA_ADVERTISE_IP` | auto (primary LAN IP) | IP announced over mDNS. Pin it on multi-homed hosts (Docker/VPN) so clients don't get an unreachable bridge address. |

`just` loads `.env` automatically. The `--port` / `--cpu` CLI flags take precedence over the
matching variable.

## ✅ Try it

```bash
curl -F file=@tests/fixtures/jfk.wav -F response_format=json http://localhost:8000/inference
# {"text":" And so, my fellow Americans, ask not what your country can do for you..."}
```

Model `large-v3-turbo-q5_0`: ~2–5s/clip on an iGPU, ~10s on CPU. `just doctor` tells you
what's missing if something goes sideways.

## 🎙️ API

- `POST /inference` — multipart `file` (WAV), plus optional `language` (a whisper
  code/name or `auto`; overrides `MURMURIA_LANGUAGE` for this request), `temperature`
  (`0.0`–`1.0`) and `response_format` (`json`) → `{ "text": ... }`. Invalid fields get a `400`.
- `WS /stream` — send binary frames of 16 kHz mono `f32` PCM (LE); receive
  `{"type":"partial","text":...}` every ~2 s (30 s sliding window) and `{"type":"final",...}`
  on any text message (flush) or on close.
- `GET /health` — `{ "service": "murmuria", "version": ... }`, used by clients to confirm
  an address really is a murmuria server.

## 📡 Network discovery (mDNS)

The server advertises itself over mDNS/Bonjour, so clients don't need a hardcoded IP:

- **Native clients** can browse the DNS-SD service type `_murmuria._tcp.local.`
- **Browsers** (which can't browse DNS-SD) resolve the fixed hostname **`murmuria.local`**
  through the OS itself — just open `http://murmuria.local:8000`.

It's best-effort: if the mDNS stack doesn't come up, the server runs normally and clients
fall back to the configured address. On the host: macOS has it built in, Linux needs `avahi`,
Windows 10+ has it natively.
