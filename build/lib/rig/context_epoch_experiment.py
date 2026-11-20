from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .context_selection import METHODS
from .curve_experiment import run, validate_dataset, write_resolved_config
from .io import Example, read_jsonl


def complete_records(
    path: Path, expected_points: int, max_updates: int
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    required = {"test_log_loss", "test_accuracy"}
    return (
        records
        if len(records) == expected_points
        and all(required <= row.keys() for row in records)
        and int(records[-1].get("update", -1)) == max_updates
        else []
    )


def save_records(output: Path, records: list[dict[str, Any]]) -> None:
    ordered = sorted(
        records, key=lambda row: (row["variant"], row.get("update", row["epoch"]))
    )
    (output / "context_epoch_results.json").write_text(
        json.dumps({"records": ordered}, indent=2), encoding="utf-8"
    )
    with (output / "context_epoch_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(dict.fromkeys(key for row in ordered for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


def plot_metric(
    output: Path,
    records: list[dict[str, Any]],
    methods: list[str],
    metric: str,
    error: str,
    original: str,
    ylabel: str,
    filename: str,
    scale: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    styles = {
        "ours": ("#D13C32", "*", 2.8),
        "activation": ("#4C78A8", "P", 1.8),
        "uncertainty": ("#9B59B6", "^", 1.8),
        "difficulty": ("#E69F00", "s", 1.8),
    }
    with plt.rc_context({"font.size": 10.5, "axes.labelsize": 11.5}):
        figure, axis = plt.subplots(figsize=(8.8, 5.15))
        baseline = float(records[0][original]) * scale
        bounds = [baseline]
        for method in methods:
            rows = sorted(
                [row for row in records if row["variant"] == method],
                key=lambda row: row.get("update", row["epoch"]),
            )
            updates = np.asarray([row.get("update", row["epoch"]) for row in rows])
            values = np.asarray([row[metric] for row in rows]) * scale
            spread = np.asarray([row[error] for row in rows]) * scale
            color, marker, width = styles[method]
            axis.fill_between(
                updates,
                values - spread,
                values + spread,
                color=color,
                alpha=0.08 if method == "ours" else 0.05,
                linewidth=0,
            )
            axis.plot(
                updates,
                values,
                label=method,
                color=color,
                linewidth=width,
                marker=marker,
                markevery=2,
                markersize=8 if method == "ours" else 5.5,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
            bounds.extend((values - spread).tolist())
            bounds.extend((values + spread).tolist())
        axis.axhline(
            baseline,
            color="#222222",
            linestyle="--",
            linewidth=1.3,
            label="original",
        )
        lower, upper = min(bounds), max(bounds)
        margin = max(0.03 * scale, 0.08 * max(upper - lower, 1e-8))
        axis.set_ylim(lower - margin, upper + margin)
        maximum_update = max(row.get("update", row["epoch"]) for row in records)
        axis.set_xlim(0, maximum_update)
        axis.set_xticks([0, 100, 200, 300, 400, 500])
        axis.set_xlabel("Optimizer update")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D8DDDF", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            ncol=len(methods) + 1,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            columnspacing=1.3,
            handlelength=2.2,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.91))
        figure.savefig(output / f"{filename}.png", dpi=240)
        figure.savefig(output / f"{filename}.pdf")
        plt.close(figure)


def validate_candidate_metadata(path: Path, need_qair: bool) -> None:
    examples = [Example.from_dict(row) for row in read_jsonl(path)]
    if len(examples) != 500:
        raise ValueError(f"Expected 500 candidate examples, found {len(examples)}.")
    if not all("activation_neuron_ids" in item.metadata.get("rig", {}) for item in examples):
        raise ValueError(
            "Candidates predate activation-overlap retrieval. Regenerate the 500 "
            "candidates with the current `rig.selection prepare` first."
        )
    if need_qair and not all("sampled_gradient_features" in item.metadata.get("qair", {}) for item in examples):
        raise ValueError("The candidate file does not contain complete qair features.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the four retrieval-conditioned self-distillation baselines."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-updates", type=int, default=500)
    parser.add_argument("--eval-every-updates", type=int, default=10)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--rerun-methods", default="")
    parser.add_argument("--rebuild-subsets", action="store_true")
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--activation-context-count", type=int, default=33)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown or not methods:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.max_updates < 1 or args.eval_every_updates < 1:
        raise ValueError("Update counts must be positive.")
    if args.max_updates % args.eval_every_updates:
        raise ValueError("--max-updates must be divisible by --eval-every-updates.")
    rerun = {item.strip() for item in args.rerun_methods.split(",") if item.strip()}
    if rerun - set(methods):
        raise ValueError("--rerun-methods must be included in --methods.")

    root = Path(__file__).resolve().parents[2]
    python = sys.executable
    data = Path(args.data).resolve()
    validate_dataset(data, 500)
    source = Path(args.source_run).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "resolved_config.json"
    config = write_resolved_config(
        args.config,
        config_path,
        args.model,
        generation_rollouts=5,
        eval_rollouts=1,
        max_new_tokens=10,
    )

    standard = source / "candidates" / "standard.jsonl.gz"
    qair = source / "candidates" / "standard_qair.jsonl.gz"
    if not standard.exists():
        raise FileNotFoundError(f"Missing prepared candidates: {standard}")
    validate_candidate_metadata(standard, need_qair=False)
    if "ours" in methods and not qair.exists():
        run(
            [
                python,
                "-m",
                "rig.qa_information",
                "extract",
                "--config",
                str(config_path),
                "--input",
                str(standard),
                "--output",
                str(qair),
                "--model",
                config["model"]["name"],
            ],
            root,
        )
    candidates = qair if "ours" in methods else standard
    validate_candidate_metadata(candidates, need_qair="ours" in methods)

    subset_dir = output / "context_subsets"
    manifest_path = subset_dir / "context_manifest.json"
    if args.rebuild_subsets or not manifest_path.exists():
        run(
            [
                python,
                "-m",
                "rig.context_selection",
                "--input",
                str(candidates),
                "--output-dir",
                str(subset_dir),
                "--methods",
                ",".join(methods),
                "--fraction",
                "0.2",
                "--context-count",
                str(args.context_count),
                "--activation-context-count",
                str(args.activation_context_count),
            ],
            root,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selections = {row["method"]: Path(row["path"]) for row in manifest["subsets"]}
    missing = set(methods) - set(selections)
    if missing:
        raise ValueError(f"Missing context subsets for: {sorted(missing)}")

    combined_path = output / "context_epoch_results.json"
    combined = (
        list(json.loads(combined_path.read_text(encoding="utf-8")).get("records", []))
        if combined_path.exists()
        else []
    )
    expected_points = args.max_updates // args.eval_every_updates
    for method in methods:
        method_dir = output / method
        records = complete_records(
            method_dir / "epoch_metrics.json", expected_points, args.max_updates
        )
        if not records or method in rerun:
            run(
                [
                    python,
                    "-m",
                    "rig.train_epoch_curve_single_gpu",
                    "--config",
                    str(config_path),
                    "--model",
                    config["model"]["name"],
                    "--selected",
                    str(selections[method]),
                    "--candidate-pool",
                    str(candidates),
                    "--context-distillation",
                    "--data",
                    str(data),
                    "--output-dir",
                    str(method_dir),
                    "--variant",
                    method,
                    "--epochs",
                    "1",
                    "--num-rollouts",
                    "1",
                    "--batch-size",
                    str(args.batch_size),
                    "--max-updates",
                    str(args.max_updates),
                    "--eval-every-updates",
                    str(args.eval_every_updates),
                ],
                root,
            )
            records = complete_records(
                method_dir / "epoch_metrics.json", expected_points, args.max_updates
            )
        combined = [row for row in combined if row["variant"] != method] + records
        save_records(output, combined)

    plot_metric(
        output,
        combined,
        methods,
        "test_accuracy",
        "test_accuracy_standard_error",
        "original_test_accuracy",
        "Test accuracy (%)",
        "context_accuracy_epoch_curve",
        scale=100.0,
    )
    plot_metric(
        output,
        combined,
        methods,
        "test_log_loss",
        "test_log_loss_standard_error",
        "original_test_log_loss",
        "Test choice log loss (lower is better)",
        "context_log_loss_epoch_curve",
    )
    print(f"Saved comparison curves and records to {output}")


if __name__ == "__main__":
    main()
