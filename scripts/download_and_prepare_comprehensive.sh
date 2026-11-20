#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"

DATA_DIR="${COMPREHENSIVE_DATA_DIR:-$ROOT/data}"
OUTPUT="${COMPREHENSIVE_OUTPUT:-$ROOT/artifacts/comprehensive_500_500.jsonl.gz}"
SCIKNOW_DIR="$DATA_DIR/sciknoweval"
MMLU_DIR="$DATA_DIR/mmlu_pro"
mkdir -p "$SCIKNOW_DIR" "$MMLU_DIR" "$(dirname "$OUTPUT")"

if [[ -s "$OUTPUT" ]]; then
  echo "Reusing prepared comprehensive dataset: $OUTPUT"
  exit 0
fi

SCIKNOW_FILE="$SCIKNOW_DIR/data/v2/sciknoweval_test_v2.jsonl"
if [[ ! -s "$SCIKNOW_FILE" ]]; then
  "$PYTHON" - "$SCIKNOW_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="hicai-zju/SciKnowEval", repo_type="dataset",
                filename="data/v2/sciknoweval_test_v2.jsonl", local_dir=sys.argv[1])
PY
else
  echo "Reusing SciKnowEval source: $SCIKNOW_FILE"
fi

if ! find "$MMLU_DIR" -type f -name '*.parquet' -print -quit | grep -q .; then
  "$PYTHON" - "$MMLU_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id="TIGER-Lab/MMLU-Pro", repo_type="dataset",
                  allow_patterns=["*.parquet", "**/*.parquet"],
                  local_dir=sys.argv[1], max_workers=1)
PY
else
  echo "Reusing MMLU-Pro parquet files below: $MMLU_DIR"
fi

"$PYTHON" -m rig.prepare_comprehensive \
  --sciknow-source "$SCIKNOW_FILE" \
  --mmlu-source "$MMLU_DIR" \
  --output "$OUTPUT"

echo "Prepared comprehensive dataset: $OUTPUT"
