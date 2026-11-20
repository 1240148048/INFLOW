#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export MODEL="${MODEL:-$ROOT/models/Qwen3-4B-Instruct-2507}"
MODEL_TAG="$(basename "$MODEL")"
export METHODS=full,redundancy,consistency,neuron,mutual,js
export JS_TASK_PROMPT=1
export SKIP_ORIGINAL=0
export OPEN_MIX_RECIPE=hotpot_coqa_squad
export UPDATES=250
export CURVE=0
export OUTPUT="${OUTPUT:-$ROOT/artifacts/inflow_${MODEL_TAG}_seed${SEED:-42}_hotpot_medium_all_250_joint15_jscluster_v8}"
unset WALL_BUDGET_SECONDS
echo "[Run] Original, Full, Redundancy, Consistency, NEURON, Mutual, JS"
echo "[Data] HotpotQA medium 200 / CoQA 150 / SQuAD 150 per split"
echo "[Protocol] 1 initial answer + 5 retrieved answers; LLM clustering for JS only; certainty=exp(-mean predictive entropy)"
echo "[Train] 250 updates per method; shared answer-only prompts; final evaluation only"
echo "[Select] JS uses global top 20%; no task quotas; Full retains its full-pool design"
echo "[Output] $OUTPUT"
exec bash "$ROOT/scripts/run_autodl_open_500.sh"
