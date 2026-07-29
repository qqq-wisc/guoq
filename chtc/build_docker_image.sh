#!/bin/bash
# Build (and optionally push) the GUOQ + routing-solvers image for CHTC.
#   ./chtc/build_docker_image.sh <tag> [--push]
# Build context is the REPO ROOT so the Dockerfile can COPY src/, rules/, etc.
#
# The Dockerfile also builds FastLS from LOCAL source, supplied as a named build
# context. Override its path with FASTLS_SRC=/path/to/FastLS if it lives elsewhere.
set -euo pipefail

# BuildKit is REQUIRED: the Dockerfile uses `# syntax=`, RUN --mount cache mounts,
# multi-stage COPY --from, and a named `--build-context` (all BuildKit features).
export DOCKER_BUILDKIT=1

TAG="${1:?usage: build_docker_image.sh <tag> [--push]}"
IMAGE="amxu/guoq-scmr:${TAG}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FASTLS_SRC="${FASTLS_SRC:-/Users/amandaxu/Desktop/uwisc/research/FastLS}"

# Fail fast with a clear message if the docker daemon isn't reachable, instead
# of blocking silently.
if ! docker info >/dev/null 2>&1; then
    echo "error: docker daemon not reachable (is Docker running?)" >&2
    exit 1
fi

# FastLS is built from this local checkout; fail clearly if it's missing.
if [[ ! -f "${FASTLS_SRC}/Cargo.toml" ]]; then
    echo "error: FastLS source not found at ${FASTLS_SRC} (set FASTLS_SRC=/path/to/FastLS)" >&2
    exit 1
fi

echo "building ${IMAGE} (context: ${REPO_ROOT}, fastls: ${FASTLS_SRC})"
# buildx is required for --build-context; --load makes the single-platform image
# available to the local docker daemon (for `docker push`). --progress=plain
# streams each step's output so the build is never silent.
docker buildx build --platform linux/amd64 --progress=plain --load \
    -f "${REPO_ROOT}/chtc/Dockerfile" \
    --build-context "fastls=${FASTLS_SRC}" \
    -t "${IMAGE}" "${REPO_ROOT}"

if [[ "${2:-}" == "--push" ]]; then
    docker push "${IMAGE}"
fi
echo "built ${IMAGE}"
