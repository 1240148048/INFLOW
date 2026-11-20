#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPEN_MIX_RECIPE=mbpp_coqa_squad
export MODEL="${MODEL:-$ROOT/models/Qwen3-4B-Instruct-2507}"
export METHODS=full,redundancy,consistency,neuron,mutual,js
export UPDATES=250
export CURVE=0
export WALL_BUDGET_SECONDS="${WALL_BUDGET_SECONDS:-10800}"
MODEL_TAG="$("${PYTHON:-python}" -c 'from pathlib import Path; import re,sys; print(re.sub(r"[^A-Za-z0-9_-]","_",Path(sys.argv[1]).name))' "$MODEL")"
export OUTPUT="${OUTPUT:-$ROOT/artifacts/inflow_${MODEL_TAG}_seed${SEED:-42}_three_task_250_table}"
exec bash "$ROOT/scripts/run_autodl_open_500.sh"
