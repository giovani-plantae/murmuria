<div align="center">

<img src="assets/icon.svg" width="120" height="120" alt="murmuria" />

# murmuria

**Self-hosted speech-to-text, accelerated on your own GPU.**

[![CI](https://github.com/giovani-plantae/murmuria/actions/workflows/ci.yml/badge.svg)](https://github.com/giovani-plantae/murmuria/actions/workflows/ci.yml)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

</div>

A thin, self-hosted server that wraps **Whisper** (through whisper.cpp) and exposes it over
**HTTP** and **WebSocket** — send audio, get text back. GPU-accelerated via **Vulkan**, so it
runs on AMD GPUs (RADV) as well as on CPU. No cloud, no API keys, no per-minute billing. 🎉

## 🚀 Running

### 🦀 Local dev (on your GPU)

Needs Rust (rustup) + `just` (`cargo install just`):

```bash
# install the native libs (sudo) and download the model (~573 MB)
just setup
# serve on :8000 — auto-detects the GPU backend (NVIDIA CUDA → Vulkan → CPU)
just run
# force CPU instead (~10s/clip): no GPU stack needed
just run --cpu
# dev loop with auto-reload
just dev
# diagnostics: is the model present? does the backend load?
just doctor
```

### 🐳 Docker (reproducible / production)

Only needs Docker (GPU: Linux with `/dev/dri`). The model is baked into the image:

```bash
# build + serve in the background, auto-restarts on reboot
docker compose up -d --build
# or a one-shot script that builds and auto-detects & passes the GPU
just docker-up
```

### 📦 Published image (GHCR)

A self-contained image is published to the GitHub Container Registry on every
release — pull and run, no build and no model download:

```bash
docker pull ghcr.io/giovani-plantae/murmuria:latest
```

For an always-on deploy (e.g. a Portainer stack), use the published-image compose:

```bash
docker compose -f docker-compose-portainer.yml up -d
```

## ⚙️ Configuration

Everything has sensible defaults, so murmuria runs with **no config at all**.
To tweak anything, copy the example file (its values are the defaults, ready to
edit) and adjust:

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

`just` loads `.env` automatically. The `--port` / `--cpu` CLI flags take
precedence over the matching variable.

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
