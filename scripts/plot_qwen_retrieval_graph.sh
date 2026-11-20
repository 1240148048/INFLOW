#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/artifacts/opsd_selection_t2000_u500}"
OUTPUT="${OUTPUT:-$ROOT/artifacts/qwen_retrieval_pca}"

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
"$PYTHON" -m rig.opsd_retrieval_visualize \
  --candidates "$SOURCE_RUN/candidates/shared.jsonl.gz" \
  --retrieval-dir "$SOURCE_RUN/retrieval" \
  --subsets-dir "$SOURCE_RUN/subsets" \
  --output-dir "$OUTPUT" \
  --context-count "${CONTEXT_COUNT:-3}" \
  --max-targets "${MAX_TARGETS:-60}"
