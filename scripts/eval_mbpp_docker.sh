#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLES="${1:?Usage: eval_mbpp_docker.sh predictions.jsonl tests.jsonl output_directory}"
TESTS="${2:?Official test file is required}"
RESULTS="${3:?Output directory is required}"
IMAGE="${EVAL_IMAGE:-inflow-mbpp:py311}"
SAMPLES="$(cd "$(dirname "$SAMPLES")" && pwd)/$(basename "$SAMPLES")"
TESTS="$(cd "$(dirname "$TESTS")" && pwd)/$(basename "$TESTS")"
mkdir -p "$RESULTS"
RESULTS="$(cd "$RESULTS" && pwd)"
if [[ -f "$RESULTS/eval_results.json" ]]; then
  echo "Reusing execution results; score validates program identity."
  exit 0
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" -f "$ROOT/docker/open_mbpp.Dockerfile" "$ROOT/docker"
fi
docker run --rm --network none --read-only --tmpfs /tmp:rw,nosuid,size=256m \
  --cpus 2 --memory 2g --pids-limit 64 --cap-drop ALL \
  --security-opt no-new-privileges \
  --mount "type=bind,source=$ROOT/src/rig/open_tasks/mbpp_worker.py,target=/worker.py,readonly" \
  --mount "type=bind,source=$SAMPLES,target=/inputs/samples.jsonl,readonly" \
  --mount "type=bind,source=$TESTS,target=/inputs/tests.jsonl,readonly" \
  --mount "type=bind,source=$RESULTS,target=/results" "$IMAGE" python -I /worker.py
