#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-$ROOT/configs/single_gpu_48gb.json}"
DATA="${DATA_FILE:-$ROOT/artifacts/comprehensive_500_500.jsonl.gz}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/artifacts/context_candidates}"
mkdir -p "$SOURCE_RUN/candidates"

python -m rig.selection prepare \
  --config "$CONFIG" \
  --model "$MODEL" \
  --input "$DATA" \
  --first-output "$SOURCE_RUN/candidates/first.jsonl.gz" \
  --output "$SOURCE_RUN/candidates/standard.jsonl.gz" \
  --resume
