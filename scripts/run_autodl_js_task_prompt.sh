#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
export MODEL="${MODEL:-$ROOT/models/Qwen3-4B-Instruct-2507}"
MODEL_TAG="$(basename "$MODEL")"
export METHODS=js
export JS_TASK_PROMPT=1
export SKIP_ORIGINAL=1
export OPEN_MIX_RECIPE=hotpot_coqa_squad
export UPDATES="${UPDATES:-250}"
export CURVE=0
export OUTPUT="${OUTPUT:-$ROOT/artifacts/inflow_${MODEL_TAG}_seed${SEED:-42}_hotpot_medium_js_joint15_jscluster_v8}"
unset WALL_BUDGET_SECONDS
echo "[JS] HotpotQA medium 200 / CoQA 150 / SQuAD 150; global top 20%, no task quotas"
echo "[JS] answer-only prompts + updated LLM clustering; updates=$UPDATES, final evaluation only"
echo "[JS] output=$OUTPUT"
exec bash "$ROOT/scripts/run_autodl_open_500.sh"
