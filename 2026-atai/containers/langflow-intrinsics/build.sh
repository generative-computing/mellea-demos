#!/usr/bin/env bash
set -euo pipefail

IMAGE_TAG="${1:-quay.io/mellea/langflow-intrinsics:latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Building $IMAGE_TAG ..."
docker build -t "$IMAGE_TAG" "$SCRIPT_DIR"

echo "Done: $IMAGE_TAG"
