# murmuria — task runner.  Install:  cargo install just   (or)  apt install just
# List recipes:  just

set dotenv-load := true

model_name := "large-v3-turbo-q5_0"
model_file := "models/ggml-" + model_name + ".bin"

# List the recipes
default:
    @just --list

# Install the native system deps (asks for sudo) and download the model
setup:
    sudo apt-get update && sudo apt-get install -y \
        build-essential cmake pkg-config clang libclang-dev \
        libvulkan-dev glslc vulkan-tools mesa-vulkan-drivers
    @just model

# Run a cargo subcommand with the auto-detected GPU backend. `--cpu` in the args
# forces a CPU build; anything else is forwarded to the binary after `--`.
[private]
_run subcommand *args:
    #!/usr/bin/env bash
    set -euo pipefail
    feature=""
    if [[ " {{args}} " == *" --cpu "* ]]; then
        echo "murmuria → CPU backend (forced)" >&2
    elif command -v nvidia-smi >/dev/null 2>&1; then
        feature="--features cuda"; echo "murmuria → NVIDIA CUDA backend" >&2
    elif [ -e /dev/dri ]; then
        feature="--features vulkan"; echo "murmuria → Vulkan backend" >&2
    else
        echo "murmuria → no GPU detected, CPU backend" >&2
    fi
    rest="$(printf '%s' "{{args}}" | sed 's/--cpu//g' | xargs || true)"
    MURMURIA_MODEL={{model_file}} cargo {{subcommand}} $feature ${rest:+-- $rest}

# Start the server, auto-detecting the backend (NVIDIA CUDA → Vulkan → CPU). `just run --cpu` forces CPU; extra flags forward (`just run --port 9000`)
run *args:
    @just _run run {{args}}

# Start the server, release build (same auto-detection as `run`)
run-release *args:
    @just _run "run --release" {{args}}

# Dev loop with auto-reload (bacon)
dev:
    bacon run

# Self-diagnostic: model present? backend loads? (`just doctor --cpu` to test CPU)
doctor *args:
    @just _run run doctor {{args}}

# Tests
test:
    cargo test --features vulkan

# Format
fmt:
    cargo fmt --all

# Lint (warnings = errors, same gate as CI)
lint:
    cargo clippy --all-targets --features vulkan -- -D warnings

# fmt + lint + test — run before pushing
check: fmt lint test

# Download the ggml model into models/ (e.g. `just model_name=small model`)
model:
    mkdir -p models
    test -f {{model_file}} || curl -L -o {{model_file}} \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{{model_name}}.bin

# Build + run via Docker (reproducible / prod): GPU and model handled by the script
docker-up:
    bash scripts/run-server.sh
