#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CONFIG="${CONFIG:-$ROOT/configs/single_gpu_48gb.json}"
ARTIFACTS="${DATA_ROOT:-$ROOT}/artifacts"
DATA="${DATA_FILE:-$ARTIFACTS/comprehensive_500_500.jsonl.gz}"
ROLLOUTS="${ROLLOUTS:-3}"
EPOCHS="${EPOCHS:-10}"
RIG_RANK="${RIG_RANK:-64}"
mkdir -p "$ARTIFACTS" "$ARTIFACTS/checkpoints" "$ARTIFACTS/eval" "$ARTIFACTS/reports"

if [[ ! -s "$DATA" ]]; then
  echo "Prepared comprehensive dataset not found: $DATA" >&2
  echo "Run: bash scripts/download_and_prepare_comprehensive.sh" >&2
  exit 2
fi
COUNTS=$(python -c 'import sys; from rig.io import read_jsonl; c={"train":0,"test":0}; [c.__setitem__(r["split"],c.get(r["split"],0)+1) for r in read_jsonl(sys.argv[1])]; print(c["train"],c["test"])' "$DATA")
read -r TRAIN_TOTAL TEST_TOTAL <<< "$COUNTS"
if [[ "$TRAIN_TOTAL" -ne 500 || "$TEST_TOTAL" -ne 500 ]]; then
  echo "Expected comprehensive train/test counts 500/500, found $TRAIN_TOTAL/$TEST_TOTAL" >&2
  exit 2
fi
python -c 'import collections,sys; from rig.io import read_jsonl; expected={"Biology":50,"Chemistry":50,"Material":50,"Physics":50,"Math":100,"Psychology":40,"Global Facts":40,"History":40,"Economics":40,"Law":40}; rows=list(read_jsonl(sys.argv[1])); actual={s:dict(collections.Counter(r["domain"] for r in rows if r["split"]==s)) for s in ("train","test")}; assert actual["train"]==expected and actual["test"]==expected, f"Wrong domain composition: {actual}"; print("Verified domain composition:", actual["train"])' "$DATA"

FIRST="$ARTIFACTS/er5_first_n500.jsonl.gz"
STANDARD="$ARTIFACTS/er5_standard_n500.jsonl.gz"
SELECTED="$ARTIFACTS/er5_weighted_selected_n100.jsonl.gz"
POSITIVE="$ARTIFACTS/er5_positive_selected_p20.jsonl.gz"

FIRST_ADAPTER="$ARTIFACTS/checkpoints/er5_first_n500_e${EPOCHS}"
STANDARD_ADAPTER="$ARTIFACTS/checkpoints/er5_standard_n500_e${EPOCHS}"
SELECTED_ADAPTER="$ARTIFACTS/checkpoints/er5_weighted_n100_e${EPOCHS}"
POSITIVE_ADAPTER="$ARTIFACTS/checkpoints/er5_positive_p20_e${EPOCHS}"

ORIGINAL_EVAL="$ARTIFACTS/eval/er5_original_n500_r${ROLLOUTS}"
FIRST_EVAL="$ARTIFACTS/eval/er5_first_n500_r${ROLLOUTS}_e${EPOCHS}"
STANDARD_EVAL="$ARTIFACTS/eval/er5_standard_n500_r${ROLLOUTS}_e${EPOCHS}"
SELECTED_EVAL="$ARTIFACTS/eval/er5_weighted_n100_r${ROLLOUTS}_e${EPOCHS}"
POSITIVE_EVAL="$ARTIFACTS/eval/er5_positive_p20_r${ROLLOUTS}_e${EPOCHS}"
REPORT="$ARTIFACTS/reports/er5_comparison_e${EPOCHS}_r${ROLLOUTS}.json"

MODEL_ARGS=()
if [[ -n "${RIG_MODEL:-}" ]]; then
  MODEL_ARGS+=(--model "$RIG_MODEL")
fi

PREPARATION_START=$(date +%s)
python -m rig.selection prepare \
  --config "$CONFIG" \
  "${MODEL_ARGS[@]}" \
  --input "$DATA" \
  --first-output "$FIRST" \
  --output "$STANDARD" \
  --resume
PREPARATION_SECONDS=$(($(date +%s) - PREPARATION_START))

for FILE in "$FIRST" "$STANDARD"; do
  COUNT=$(python -c 'import sys; from rig.io import read_jsonl; print(sum(1 for _ in read_jsonl(sys.argv[1])))' "$FILE")
  if [[ "$COUNT" -ne 500 ]]; then
    echo "Expected 500 candidates in $FILE, found $COUNT" >&2
    exit 2
  fi
done

SELECTION_START=$(date +%s)
python -m rig.selection select \
  --config "$CONFIG" --input "$STANDARD" --output "$SELECTED" --rank "$RIG_RANK"
python -m rig.selection select \
  --config "$CONFIG" --input "$STANDARD" --output "$POSITIVE" --rank "$RIG_RANK" \
  --positive-only
SELECTION_SECONDS=$(($(date +%s) - SELECTION_START))

train_variant() {
  local data=$1
  local output=$2
  local start
  start=$(date +%s)
  python -m rig.train_single_gpu \
    --config "$CONFIG" "${MODEL_ARGS[@]}" \
    --selected "$data" --epochs "$EPOCHS" --output-dir "$output"
  TRAINING_SECONDS=$(($(date +%s) - start))
}

train_variant "$FIRST" "$FIRST_ADAPTER"
FIRST_TRAINING_SECONDS=$TRAINING_SECONDS
train_variant "$STANDARD" "$STANDARD_ADAPTER"
STANDARD_TRAINING_SECONDS=$TRAINING_SECONDS
train_variant "$SELECTED" "$SELECTED_ADAPTER"
SELECTED_TRAINING_SECONDS=$TRAINING_SECONDS
train_variant "$POSITIVE" "$POSITIVE_ADAPTER"
POSITIVE_TRAINING_SECONDS=$TRAINING_SECONDS

evaluate_variant() {
  local output=$1
  local adapter=${2:-}
  local adapter_args=()
  if [[ -n "$adapter" ]]; then
    adapter_args+=(--adapter "$adapter")
  fi
  python -m rig.evaluate_single_gpu \
    --config "$CONFIG" "${MODEL_ARGS[@]}" \
    --data "$DATA" "${adapter_args[@]}" \
    --num-rollouts "$ROLLOUTS" --output "$output"
}

if [[ ! -f "$ORIGINAL_EVAL/metrics.json" ]]; then
  evaluate_variant "$ORIGINAL_EVAL"
fi
evaluate_variant "$FIRST_EVAL" "$FIRST_ADAPTER"
evaluate_variant "$STANDARD_EVAL" "$STANDARD_ADAPTER"
evaluate_variant "$SELECTED_EVAL" "$SELECTED_ADAPTER"
evaluate_variant "$POSITIVE_EVAL" "$POSITIVE_ADAPTER"

python -m rig.report \
  --output "$REPORT" \
  --original-metrics "$ORIGINAL_EVAL/metrics.json" \
  --first-metrics "$FIRST_EVAL/metrics.json" \
  --standard-metrics "$STANDARD_EVAL/metrics.json" \
  --selected-metrics "$SELECTED_EVAL/metrics.json" \
  --positive-metrics "$POSITIVE_EVAL/metrics.json" \
  --first-training "$FIRST_ADAPTER/training_metrics.json" \
  --standard-training "$STANDARD_ADAPTER/training_metrics.json" \
  --selected-training "$SELECTED_ADAPTER/training_metrics.json" \
  --positive-training "$POSITIVE_ADAPTER/training_metrics.json" \
  --weighted-selection "${SELECTED}.metrics.json" \
  --positive-selection "${POSITIVE}.metrics.json" \
  --preparation-seconds "$PREPARATION_SECONDS" \
  --selection-seconds "$SELECTION_SECONDS" \
  --first-training-seconds "$FIRST_TRAINING_SECONDS" \
  --standard-training-seconds "$STANDARD_TRAINING_SECONDS" \
  --selected-training-seconds "$SELECTED_TRAINING_SECONDS" \
  --positive-training-seconds "$POSITIVE_TRAINING_SECONDS"

echo "Five-way ER reliability comparison completed: $REPORT"
