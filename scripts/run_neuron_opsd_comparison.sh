#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-$ROOT/configs/neuron_opsd_single_gpu.json}"
DATA_FILE="${DATA_FILE:-$ROOT/artifacts/comprehensive_500_500.jsonl.gz}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT="${OUTPUT:-$ROOT/artifacts/opsd_selection_t2000_u500}"
TRAJECTORIES="${TRAJECTORIES:-2000}"
EPOCHS="${EPOCHS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_EVERY_UPDATES="${EVAL_EVERY_UPDATES:-50}"
METHODS="${METHODS:-full,redundancy,consistency,neuron,ours,mutual}"
PLOT_METHODS="${PLOT_METHODS:-$METHODS}"

if (( EPOCHS == 0 && TRAJECTORIES % BATCH_SIZE != 0 )); then
  echo "TRAJECTORIES must be divisible by BATCH_SIZE." >&2
  exit 2
fi
TOTAL_UPDATES=$((TRAJECTORIES / BATCH_SIZE))
if (( EPOCHS == 0 && TOTAL_UPDATES % EVAL_EVERY_UPDATES != 0 )); then
  echo "TOTAL_UPDATES must be divisible by EVAL_EVERY_UPDATES." >&2
  exit 2
fi
CHECKPOINT_STRIDE=$((BATCH_SIZE * EVAL_EVERY_UPDATES))

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
mkdir -p "$OUTPUT/candidates" "$OUTPUT/retrieval" "$OUTPUT/teacher" \
  "$OUTPUT/subsets" "$OUTPUT/checkpoints" "$OUTPUT/eval"

CANDIDATES="$OUTPUT/candidates/shared.jsonl.gz"
CANDIDATES_READY=0
if [[ ! -f "$CANDIDATES.done" ]]; then
  :
elif "$PYTHON" -c \
  'import gzip,json,sys; rows=(json.loads(x) for x in gzip.open(sys.argv[1], "rt") if x.strip()); assert all(r["metadata"]["neuron_opsd"].get("neuron_ids") for r in rows)' \
  "$CANDIDATES"; then
  CANDIDATES_READY=1
fi
if (( CANDIDATES_READY == 0 )); then
  rm -f "$CANDIDATES" "$CANDIDATES.done"
  "$PYTHON" -m rig.opsd_prepare \
    --config "$CONFIG" --input "$DATA_FILE" --output "$CANDIDATES" \
    --model "$MODEL" --resume
else
  echo "Reusing candidates: $CANDIDATES"
fi

RETRIEVAL_READY=$CANDIDATES_READY
for METHOD in full redundancy consistency neuron ours mutual; do
  RETRIEVED="$OUTPUT/retrieval/${METHOD}_all.jsonl.gz"
  case "$METHOD" in
    full|redundancy) EXPECTED="final_layer_cosine_v1" ;;
    consistency) EXPECTED="no_retrieval_v1" ;;
    neuron) EXPECTED="neuron_jaccard_v1" ;;
    ours) EXPECTED="rcri_final_layer_v1" ;;
    mutual) EXPECTED="layerwise_gaussian_information_v1" ;;
  esac
  if [[ ! -f "$RETRIEVED" ]] || ! "$PYTHON" -c \
    'import gzip,json,sys; row=json.loads(next(gzip.open(sys.argv[1], "rt"))); assert row["metadata"]["opsd_retrieval"]["definition"] == sys.argv[2]' \
    "$RETRIEVED" "$EXPECTED" 2>/dev/null; then
    RETRIEVAL_READY=0
  fi
done
if (( RETRIEVAL_READY == 0 )); then
  "$PYTHON" -m rig.opsd_select \
    --input "$CANDIDATES" --output-dir "$OUTPUT/retrieval" --context-count 3
else
  echo "Reusing retrieval records from $OUTPUT/retrieval"
fi

for METHOD in full consistency neuron ours mutual; do
  TEACHER_DATA="$OUTPUT/teacher/${METHOD}_all.jsonl.gz"
  case "$METHOD" in
    full) EXPECTED="final_layer_cosine_v1" ;;
    consistency) EXPECTED="no_retrieval_v1" ;;
    neuron) EXPECTED="neuron_jaccard_v1" ;;
    ours) EXPECTED="rcri_final_layer_v1" ;;
    mutual) EXPECTED="layerwise_gaussian_information_v1" ;;
  esac
  TEACHER_READY=0
  if [[ -f "$TEACHER_DATA.done" && -f "$TEACHER_DATA" ]] && "$PYTHON" -c \
    'import gzip,json,sys; row=json.loads(next(gzip.open(sys.argv[1], "rt"))); assert row["metadata"]["opsd_retrieval"]["definition"] == sys.argv[2]' \
    "$TEACHER_DATA" "$EXPECTED" 2>/dev/null; then
    TEACHER_READY=1
  fi
  if (( TEACHER_READY == 0 )); then
    rm -f "$TEACHER_DATA" "$TEACHER_DATA.done"
    "$PYTHON" -m rig.opsd_teacher_prepare \
      --config "$CONFIG" \
      --retrieved "$OUTPUT/retrieval/${METHOD}_all.jsonl.gz" \
      --candidates "$CANDIDATES" --output "$TEACHER_DATA" \
      --method "$METHOD" --model "$MODEL" --resume
  else
    echo "Reusing five-sample $METHOD teacher data: $TEACHER_DATA"
  fi
done

"$PYTHON" -m rig.opsd_finalize \
  --full "$OUTPUT/teacher/full_all.jsonl.gz" \
  --redundancy "$OUTPUT/teacher/full_all.jsonl.gz" \
  --consistency "$OUTPUT/teacher/consistency_all.jsonl.gz" \
  --neuron "$OUTPUT/teacher/neuron_all.jsonl.gz" \
  --ours "$OUTPUT/teacher/ours_all.jsonl.gz" \
  --mutual "$OUTPUT/teacher/mutual_all.jsonl.gz" \
  --output-dir "$OUTPUT/subsets" --fraction 0.2

mkdir -p "$OUTPUT/case_study"
"$PYTHON" -m rig.opsd_case_study \
  --candidates "$CANDIDATES" \
  --teacher "$OUTPUT/teacher/ours_all.jsonl.gz" \
  --selected "$OUTPUT/subsets/ours.jsonl.gz" \
  --output "$OUTPUT/case_study/ours_all.jsonl.gz" \
  --index-csv "$OUTPUT/case_study/ours_index.csv"

if [[ ! -f "$OUTPUT/eval/original/metrics.json" ]] || \
   ! "$PYTHON" -c 'import json,sys; assert "rollout_accuracy_std" in json.load(open(sys.argv[1]))["overall"]' \
      "$OUTPUT/eval/original/metrics.json"; then
  "$PYTHON" -m rig.opsd_evaluate \
    --config "$CONFIG" --data "$DATA_FILE" --model "$MODEL" \
    --output "$OUTPUT/eval/original"
else
  echo "Reusing original-model evaluation."
fi

IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
for METHOD in "${METHOD_ARRAY[@]}"; do
  case "$METHOD" in
    full|redundancy|consistency|neuron|ours|mutual) ;;
    *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
  esac
  CHECKPOINT="$OUTPUT/checkpoints/$METHOD"
  SELECTED="$OUTPUT/subsets/$METHOD.jsonl.gz"
  METHOD_TRAJECTORIES="$TRAJECTORIES"
  TRAIN_EXTRA=(--checkpoint-every-trajectories "$CHECKPOINT_STRIDE")
  if (( EPOCHS > 0 )); then
    SELECTED_COUNT=$("$PYTHON" -c \
      'import gzip,sys; print(sum(1 for line in gzip.open(sys.argv[1], "rt") if line.strip()))' \
      "$SELECTED")
    METHOD_TRAJECTORIES=$((SELECTED_COUNT * EPOCHS))
    if (( METHOD_TRAJECTORIES % BATCH_SIZE != 0 )); then
      echo "$METHOD trajectories are not divisible by BATCH_SIZE." >&2
      exit 2
    fi
    TRAIN_EXTRA=(--final-checkpoint-only)
  fi
  case "$METHOD" in
    full|redundancy) EXPECTED="final_layer_cosine_v1" ;;
    consistency) EXPECTED="no_retrieval_v1" ;;
    neuron) EXPECTED="neuron_jaccard_v1" ;;
    ours) EXPECTED="rcri_final_layer_v1" ;;
    mutual) EXPECTED="layerwise_gaussian_information_v1" ;;
  esac
  CHECKPOINT_READY=0
  if [[ -f "$CHECKPOINT/training_metrics.json" ]] && "$PYTHON" -c \
    'import json,sys; assert json.load(open(sys.argv[1]))["retrieval_definition"] == sys.argv[2]' \
    "$CHECKPOINT/training_metrics.json" "$EXPECTED"; then
    CHECKPOINT_READY=1
  fi
  if (( CHECKPOINT_READY == 0 )); then
    rm -rf "$CHECKPOINT" "$OUTPUT/eval/$METHOD"
    "$PYTHON" -m rig.opsd_train \
      --config "$CONFIG" --selected "$SELECTED" \
      --candidates "$CANDIDATES" --output-dir "$CHECKPOINT" \
      --model "$MODEL" --trajectories "$METHOD_TRAJECTORIES" \
      "${TRAIN_EXTRA[@]}"
  else
    echo "Reusing trained $METHOD adapter: $CHECKPOINT"
  fi

  if (( EPOCHS > 0 )); then
    METRICS="$OUTPUT/eval/$METHOD/final/metrics.json"
    if [[ ! -f "$METRICS" ]]; then
      "$PYTHON" -m rig.opsd_evaluate \
        --config "$CONFIG" --data "$DATA_FILE" --model "$MODEL" \
        --adapter "$CHECKPOINT" \
        --output "$OUTPUT/eval/$METHOD/final"
    else
      echo "Reusing $METHOD final evaluation."
    fi
    continue
  else
    EVAL_STRIDE="$CHECKPOINT_STRIDE"
  fi
  for TRAJECTORY_STEP in $(seq "$EVAL_STRIDE" "$EVAL_STRIDE" "$METHOD_TRAJECTORIES"); do
    UPDATE=$((TRAJECTORY_STEP / BATCH_SIZE))
    if (( EPOCHS > 0 )); then
      POINT=$((TRAJECTORY_STEP / SELECTED_COUNT))
      POINT_DIR="epoch-$POINT"
    else
      POINT="$UPDATE"
      POINT_DIR="update-$POINT"
    fi
    METRICS="$OUTPUT/eval/$METHOD/$POINT_DIR/metrics.json"
    if [[ ! -f "$METRICS" ]]; then
      "$PYTHON" -m rig.opsd_evaluate \
        --config "$CONFIG" --data "$DATA_FILE" --model "$MODEL" \
        --adapter "$CHECKPOINT/checkpoint-$TRAJECTORY_STEP" \
        --output "$OUTPUT/eval/$METHOD/$POINT_DIR"
    else
      echo "Reusing $METHOD evaluation at $POINT_DIR."
    fi
  done
done

"$PYTHON" - "$OUTPUT" "$PLOT_METHODS" "$TOTAL_UPDATES" "$EVAL_EVERY_UPDATES" "$EPOCHS" <<'PY'
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

root = Path(sys.argv[1])
methods = [value.strip() for value in sys.argv[2].split(",") if value.strip()]
total_updates = int(sys.argv[3])
evaluation_stride = int(sys.argv[4])
epochs = int(sys.argv[5])
points = list(range(1, epochs + 1)) if epochs else list(
    range(evaluation_stride, total_updates + 1, evaluation_stride)
)

labels = {
    "full": "Full",
    "redundancy": "Redundancy",
    "consistency": "Consistency",
    "neuron": "NEURON",
    "ours": "Ours",
    "mutual": "Mutual",
}
styles = {
    "full": dict(color="#6b8fb3", marker="v", linestyle=":"),
    "redundancy": dict(color="#1b9e77", marker="D", linestyle="-"),
    "consistency": dict(color="#9b59b6", marker="^", linestyle="-"),
    "neuron": dict(color="#4c78a8", marker="P", linestyle="-"),
    "ours": dict(color="#d43d2f", marker="*", linestyle="-", linewidth=2.6),
    "mutual": dict(color="#7a6f9b", marker="X", linestyle="--", linewidth=2.1),
}

original = json.loads(
    (root / "eval" / "original" / "metrics.json").read_text(encoding="utf-8")
)["overall"]
original_key = next(key for key in original if key.startswith("avg_at_"))
original_accuracy = float(original[original_key])

if epochs:
    records = []
    summary = {"original": original, "training_epochs": epochs, "methods": {}}
    for method in methods:
        path = root / "eval" / method / "final" / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["overall"]
        accuracy_key = next(key for key in metrics if key.startswith("avg_at_"))
        majority_key = next(key for key in metrics if key.startswith("maj_at_"))
        records.append({
            "method": method,
            "training_epochs": epochs,
            "accuracy": float(metrics[accuracy_key]),
            "accuracy_std": float(metrics["rollout_accuracy_std"]),
            "majority_accuracy": float(metrics[majority_key]),
        })
        summary["methods"][method] = metrics
    (root / "final_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if records:
        with (root / "final_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(summary, indent=2))
    print(f"Saved final results to {root / 'final_results.json'}")
    raise SystemExit(0)

records = []
summary = {"original": original, "methods": {}}
for method in methods:
    summary["methods"][method] = {}
    for point in points:
        point_dir = f"epoch-{point}" if epochs else f"update-{point}"
        path = root / "eval" / method / point_dir / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["overall"]
        accuracy_key = next(key for key in metrics if key.startswith("avg_at_"))
        row = {
            "method": method,
            "training_epoch" if epochs else "optimizer_update": point,
            "accuracy": float(metrics[accuracy_key]),
            "accuracy_std": float(metrics["rollout_accuracy_std"]),
            "majority_accuracy": float(
                metrics[next(key for key in metrics if key.startswith("maj_at_"))]
            ),
        }
        records.append(row)
        summary["methods"][method][str(point)] = metrics

(root / "curve_results.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)
with (root / "curve_results.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, axis = plt.subplots(figsize=(10.4, 6.0))
for method in methods:
    values = [record for record in records if record["method"] == method]
    if not values:
        continue
    x_key = "training_epoch" if epochs else "optimizer_update"
    x = np.asarray([record[x_key] for record in values])
    y = 100.0 * np.asarray([record["accuracy"] for record in values])
    sd = 100.0 * np.asarray([record["accuracy_std"] for record in values])
    style = styles[method].copy()
    linewidth = style.pop("linewidth", 1.9)
    axis.plot(
        x, y, label=labels[method], linewidth=linewidth, markersize=6.2,
        markevery=1, markeredgecolor="white", markeredgewidth=0.55, **style
    )
    axis.fill_between(x, y - sd, y + sd, color=style["color"], alpha=0.09, linewidth=0)

axis.axhline(
    100.0 * original_accuracy, color="#222222", linestyle="--", linewidth=1.6,
    label="Original"
)
axis.set_xlabel("Training epoch" if epochs else "Optimizer update")
axis.set_ylabel("Rollout accuracy (%)")
axis.set_xticks(points)
axis.grid(axis="y", color="#dfe4e8", linewidth=0.8)
axis.set_axisbelow(True)
axis.margins(x=0.015, y=0.12)
handles, legend_labels = axis.get_legend_handles_labels()
# Matplotlib fills multi-column legends by columns. Reorder handles so the
# visible first row follows the requested method order.
ncols = 4
nrows = int(np.ceil(len(handles) / ncols))
column_major = []
for column in range(ncols):
    for row in range(nrows):
        index = row * ncols + column
        if index < len(handles):
            column_major.append(index)
axis.legend(
    [handles[index] for index in column_major],
    [legend_labels[index] for index in column_major],
    loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncols,
    frameon=False, columnspacing=1.4, handlelength=2.6
)
fig.text(
    0.985, 0.018, "Bands: SD across three rollout-index accuracies.",
    ha="right", va="bottom", color="#6f767c", fontsize=9.5
)
fig.tight_layout(rect=(0.02, 0.055, 0.99, 0.88))
fig.savefig(root / "accuracy_update_curve.png", dpi=260, bbox_inches="tight")
fig.savefig(root / "accuracy_update_curve.pdf", bbox_inches="tight")
plt.close(fig)
print(json.dumps(summary, indent=2))
print(f"Saved curve to {root / 'accuracy_update_curve.png'}")
PY
