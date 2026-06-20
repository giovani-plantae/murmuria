#!/usr/bin/env bash
# Builds (first time) and runs the murmuria transcription server.
# Uses the AMD iGPU via Vulkan (/dev/dri) when available, else CPU.
# The model is baked into the image, so there's nothing to download here.
#
#   bash scripts/run-server.sh        # start on http://localhost:8000
#
# Cleanup:
#   docker rm -f murmuria && docker rmi murmuria
set -euo pipefail

PORT=8000
IMAGE=murmuria
CONTAINER=murmuria
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

# Build with half the cores so the compile doesn't peg an interactive machine.
BUILD_JOBS=$(( ( $(nproc) + 1 ) / 2 ))
docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || docker build --build-arg "BUILD_JOBS=$BUILD_JOBS" -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"

# Pass the GPU through on Linux. The container runs as a non-root user, so it
# also needs the host group(s) that own the DRI nodes to read /dev/dri.
gpu_args=()
if [ -e /dev/dri ]; then
  gpu_args+=(--device /dev/dri)
  seen_gids=" "
  for node in /dev/dri/renderD* /dev/dri/card*; do
    [ -e "$node" ] || continue
    gid="$(stat -c '%g' "$node")"
    case "$seen_gids" in *" $gid "*) continue ;; esac
    seen_gids+="$gid "
    gpu_args+=(--group-add "$gid")
  done
fi

# The container listens on 80 (matches the compose); publish it on $PORT.
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" "${gpu_args[@]}" \
  -p "127.0.0.1:$PORT:80" \
  -e "MURMURIA_PORT=80" \
  "$IMAGE"

[ "${#gpu_args[@]}" -gt 0 ] && mode="(GPU via Vulkan)" || mode="(CPU)"
echo "murmuria up → http://localhost:$PORT/inference  $mode"
