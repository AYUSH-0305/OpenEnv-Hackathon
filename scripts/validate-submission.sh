#!/usr/bin/env sh
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <base_url> [docker_image_tag]"
  echo "Example: $0 https://your-space.hf.space med-recon-env"
  exit 1
fi

BASE_URL="$1"
IMAGE_TAG="${2:-med-recon-env}"

echo "[VALIDATION] Checking environment at $BASE_URL"
RESET_URL="${BASE_URL%/}/reset"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$RESET_URL")
if [ "$HTTP_STATUS" -ne 200 ]; then
  echo "ERROR: /reset returned HTTP $HTTP_STATUS"
  exit 1
fi

echo "[VALIDATION] /reset endpoint returned HTTP 200"

echo "[VALIDATION] Building Docker image '$IMAGE_TAG' from ./server"
docker build -t "$IMAGE_TAG" ./server

echo "[VALIDATION] Docker build succeeded"

if ! command -v openenv >/dev/null 2>&1; then
  echo "ERROR: openenv CLI is not installed. Install it and retry."
  exit 1
fi

echo "[VALIDATION] Running openenv validate on openenv.yaml"
openenv validate openenv.yaml

echo "[VALIDATION] openenv validate passed"

echo "[VALIDATION] All submission checks passed successfully."
