# Builds the murmuria server (Rust) with whisper.cpp compiled for the Vulkan
# backend (GPU), and bakes in the Whisper model so the image runs standalone
# (no model mount needed). Build with the repo ROOT as context:
#   docker build -f docker/Dockerfile -t murmuria .
#
# Ubuntu 24.04 (both stages) on purpose: its Vulkan-Hpp headers (1.3.275) include
# VK_EXT_layer_settings, which the bundled whisper.cpp needs — Debian bookworm's
# older headers fail to compile ggml-vulkan. Matching runtime to builder also
# keeps glibc compatible.

FROM ubuntu:24.04 AS builder

# Cap compile parallelism (Rust + the whisper.cpp cmake build) so it doesn't peg
# an interactive machine.
ARG BUILD_JOBS=4
ENV CARGO_BUILD_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.cargo/bin:${PATH}"

# Rust (rustup) + whisper.cpp build deps. whisper-rs-sys compiles whisper.cpp
# (cmake/C++), generates bindings (libclang), links the Vulkan loader and
# compiles GLSL shaders (glslc/glslang).
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates build-essential cmake clang libclang-dev \
      libvulkan-dev glslc glslang-tools spirv-tools \
 && rm -rf /var/lib/apt/lists/* \
 && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /app
# Pre-build dependencies (incl. whisper.cpp) against a dummy main so editing src/
# later doesn't recompile the whole dependency tree.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
 && cargo build --release --features vulkan \
 && rm -rf src

COPY src ./src
# Bump mtimes so cargo rebuilds the crate over the dummy-main artifact — BuildKit
# may preserve older source mtimes, which would make cargo skip the rebuild and
# ship the empty dummy binary.
RUN touch src/*.rs && cargo build --release --features vulkan

# ---- model ------------------------------------------------------------------
# Downloaded in its own stage so it caches independently of the code build, then
# baked into the runtime image below.
FROM ubuntu:24.04 AS model
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fL -o /ggml-large-v3-turbo-q5_0.bin \
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin

# ---- runtime ----------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
# libcap2-bin provides setcap, used below to let the non-root user bind port 80.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libvulkan1 mesa-vulkan-drivers libgomp1 libcap2-bin \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 murmuria

COPY --from=builder /app/target/release/murmuria /usr/local/bin/murmuria

# Bake the model at the default MURMURIA_MODEL path (root-owned, world-readable
# so the non-root user can read it) — the image needs no model mount to run.
COPY --from=model /ggml-large-v3-turbo-q5_0.bin /models/ggml-large-v3-turbo-q5_0.bin

# Grant the bind capability so the non-root user can listen on the privileged
# port 80 (the conventional in-container port) without running as root.
RUN setcap cap_net_bind_service=+ep /usr/local/bin/murmuria

USER murmuria
EXPOSE 80
ENTRYPOINT ["murmuria"]
