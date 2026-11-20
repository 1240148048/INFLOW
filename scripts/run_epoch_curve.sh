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
SOURCE_RUN="${SOURCE_RUN:?Set SOURCE_RUN to the completed selection-curve run}"
OUTPUT="${OUTPUT:-$SOURCE_RUN/epoch_curve_e50}"
EPOCHS="${EPOCHS:-50}"
EVAL_ROLLOUTS="${EVAL_ROLLOUTS:-3}"
METHODS="${METHODS:-random,difficulty,uncertainty,redundancy,activation,full,ours-inverse,ours}"
FULL_EPOCH_SCALE="${FULL_EPOCH_SCALE:-5}"
RERUN_METHODS="${RERUN_METHODS:-}"
REBUILD_SUBSETS="${REBUILD_SUBSETS:-0}"

ARGS=(
  --config "$CONFIG"
  --data "$DATA"
  --model "$MODEL"
  --source-run "$SOURCE_RUN"
  --output-dir "$OUTPUT"
  --epochs "$EPOCHS"
  --eval-rollouts "$EVAL_ROLLOUTS"
  --methods "$METHODS"
  --full-epoch-scale "$FULL_EPOCH_SCALE"
  --max-new-tokens 10
)

if [[ -n "$RERUN_METHODS" ]]; then
  ARGS+=(--rerun-methods "$RERUN_METHODS")
fi
if [[ "$REBUILD_SUBSETS" == "1" ]]; then
  ARGS+=(--rebuild-subsets)
fi

python -m rig.epoch_curve_experiment "${ARGS[@]}"
