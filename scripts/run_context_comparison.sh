#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

CONFIG="${CONFIG:-$ROOT/configs/single_gpu_48gb.json}"
DATA="${DATA_FILE:-$ROOT/artifacts/comprehensive_500_500.jsonl.gz}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to a run containing candidates/standard.jsonl.gz}"
OUTPUT="${OUTPUT:-$ROOT/artifacts/context_comparison_u500}"
MAX_UPDATES="${MAX_UPDATES:-500}"
EVAL_EVERY_UPDATES="${EVAL_EVERY_UPDATES:-10}"
METHODS="${METHODS:-ours,activation,uncertainty,difficulty}"
RERUN_METHODS="${RERUN_METHODS:-}"
REBUILD_SUBSETS="${REBUILD_SUBSETS:-0}"
CONTEXT_COUNT="${CONTEXT_COUNT:-3}"
ACTIVATION_CONTEXT_COUNT="${ACTIVATION_CONTEXT_COUNT:-33}"
BATCH_SIZE="${BATCH_SIZE:-8}"

ARGS=(
  --config "$CONFIG"
  --data "$DATA"
  --model "$MODEL"
  --source-run "$SOURCE_RUN"
  --output-dir "$OUTPUT"
  --max-updates "$MAX_UPDATES"
  --eval-every-updates "$EVAL_EVERY_UPDATES"
  --methods "$METHODS"
  --context-count "$CONTEXT_COUNT"
  --activation-context-count "$ACTIVATION_CONTEXT_COUNT"
  --batch-size "$BATCH_SIZE"
)
if [[ -n "$RERUN_METHODS" ]]; then
  ARGS+=(--rerun-methods "$RERUN_METHODS")
fi
if [[ "$REBUILD_SUBSETS" == "1" ]]; then
  ARGS+=(--rebuild-subsets)
fi

python -m rig.context_epoch_experiment "${ARGS[@]}"
