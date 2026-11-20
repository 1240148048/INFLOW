#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-$ROOT/models/Olmo-3-7B-Instruct}"
CONFIG="${CONFIG:-$ROOT/configs/olmo3_7b_single_gpu.json}"
OUTPUT="${OUTPUT:-$ROOT/artifacts/olmo3_7b_opsd_u500}"

# opsd_train treats BATCH_SIZE trajectories as one optimizer update.
# Therefore 2,000 trajectories / 4 trajectories per update = 500 updates.
MODEL="$MODEL" \
CONFIG="$CONFIG" \
OUTPUT="$OUTPUT" \
TRAJECTORIES="${TRAJECTORIES:-2000}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
EVAL_EVERY_UPDATES="${EVAL_EVERY_UPDATES:-50}" \
METHODS="${METHODS:-full,redundancy,consistency,neuron,ours,mutual}" \
PLOT_METHODS="${PLOT_METHODS:-full,redundancy,consistency,neuron,ours,mutual}" \
bash "$ROOT/scripts/run_neuron_opsd_comparison.sh"
