from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from .curve_experiment import (
    run,
    validate_dataset,
    write_resolved_config,
)


VARIANTS = (
    "random",
    "difficulty",
    "uncertainty",
    "redundancy",
    "activation",
    "full",
    "ours-inverse",
    "ours",
)


def load_epoch_records(path: Path, epochs: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    return (
        records
        if len(records) == epochs
        and all("test_log_loss" in record for record in records)
        else []
    )


def save_combined(output: Path, records: list[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda row: (row["variant"], row["epoch"]))
    (output / "epoch_curve_results.json").write_text(
        json.dumps({"records": ordered}, indent=2), encoding="utf-8"
    )
    with (output / "epoch_curve_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fieldnames = list(
            dict.fromkeys(key for record in ordered for key in record)
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in ordered:
            row = dict(record)
            if "rollout_index_accuracies" in row:
                row["rollout_index_accuracies"] = json.dumps(
                    row["rollout_index_accuracies"]
                )
            writer.writerow(row)


def load_combined(output: Path) -> list[dict[str, Any]]:
    path = output / "epoch_curve_results.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("records", []))


def plot_epoch_curve(
    output: Path,
    records: list[dict[str, Any]],
    variants: list[str],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    style = {
        "random": ("#8A9496", "o", "-", 1.65),
        "difficulty": ("#E69F00", "s", "-", 1.6),
        "uncertainty": ("#9B59B6", "^", "-", 1.6),
        "redundancy": ("#009E73", "D", "-", 1.6),
        "activation": ("#4C78A8", "P", "-", 1.6),
        "full": ("#6B86A5", "v", ":", 1.8),
        "ours-inverse": ("#4D5966", "X", "--", 1.7),
        "ours": ("#D13C32", "*", "-", 2.8),
    }
    with plt.rc_context({"font.size": 10.5, "axes.labelsize": 11.5}):
        figure, axis = plt.subplots(figsize=(9.2, 5.3))
        original_loss = float(
            next(
                row["original_test_log_loss"]
                for row in records
                if "original_test_log_loss" in row
            )
        )
        bounds = [original_loss]
        for variant in variants:
            rows = sorted(
                [row for row in records if row["variant"] == variant],
                key=lambda row: row["epoch"],
            )
            epochs = np.asarray([row["epoch"] for row in rows])
            loss = np.asarray([row["test_log_loss"] for row in rows])
            spread = np.asarray(
                [row["test_log_loss_standard_error"] for row in rows]
            )
            color, marker, linestyle, width = style[variant]
            axis.fill_between(
                epochs,
                loss - spread,
                loss + spread,
                color=color,
                alpha=0.055 if variant != "ours" else 0.085,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                epochs,
                loss,
                label=variant,
                color=color,
                linestyle=linestyle,
                linewidth=width,
                marker=marker,
                markevery=1 if variant == "full" else 5,
                markersize=8 if variant == "ours" else 5.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                alpha=1.0 if variant == "ours" else 0.86,
                zorder=4 if variant == "ours" else 2,
            )
            bounds.extend((loss - spread).tolist())
            bounds.extend((loss + spread).tolist())
        axis.axhline(
            original_loss,
            color="#222222",
            linestyle="--",
            linewidth=1.25,
            label="original",
            zorder=0,
        )
        lower, upper = min(bounds), max(bounds)
        margin = max(0.12, 0.08 * (upper - lower))
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlim(1, max(row["epoch"] for row in records))
        axis.set_xlabel("Training exposure (20%-subset epochs)")
        axis.set_ylabel("Test choice log loss (lower is better)")
        axis.set_xticks([1, 5, 10, 20, 30, 40, 50])
        axis.grid(axis="y", color="#D8DDDF", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            ncol=min(len(variants) + 1, 5),
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            columnspacing=1.4,
            handlelength=2.4,
        )
        figure.text(
            0.99,
            0.012,
            "Bands: standard error across test questions.",
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#62696B",
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.91))
        figure.savefig(output / "log_loss_epoch_curve.png", dpi=240)
        figure.savefig(output / "log_loss_epoch_curve.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 20% selections and full data over training epochs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-rollouts", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--methods", default=",".join(VARIANTS))
    parser.add_argument("--full-epoch-scale", type=int, default=5)
    parser.add_argument("--rerun-methods", default="")
    parser.add_argument("--rebuild-subsets", action="store_true")
    args = parser.parse_args()
    variants = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(variants) - set(VARIANTS)
    if unknown or not variants:
        raise ValueError(f"Invalid epoch-curve methods: {sorted(unknown)}")
    if args.full_epoch_scale < 1:
        raise ValueError("--full-epoch-scale must be positive.")
    rerun_methods = {
        item.strip() for item in args.rerun_methods.split(",") if item.strip()
    }
    unknown_reruns = rerun_methods - set(VARIANTS)
    if unknown_reruns:
        raise ValueError(f"Invalid rerun methods: {sorted(unknown_reruns)}")

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
        eval_rollouts=args.eval_rollouts,
        max_new_tokens=args.max_new_tokens,
    )
    model_args = ["--model", config["model"]["name"]]

    manifest_path = source / "subsets" / "manifest.json"
    if args.rebuild_subsets or not manifest_path.exists():
        qair_candidates = source / "candidates" / "standard_qair.jsonl.gz"
        candidates = (
            qair_candidates
            if qair_candidates.exists()
            else source / "candidates" / "standard.jsonl.gz"
        )
        if not candidates.exists():
            raise FileNotFoundError(
                f"No subset manifest or candidate file found below {source}."
            )
        subset_methods = [
            variant for variant in variants if variant not in {"random", "full"}
        ]
        if not subset_methods:
            subset_methods = ["uncertainty"]
        run(
            [
                python,
                "-m",
                "rig.curve_selection",
                "--input",
                str(candidates),
                "--output-dir",
                str(source / "subsets"),
                "--fractions",
                "0.2",
                "--seed",
                str(config["seed"]),
                "--methods",
                ",".join(subset_methods),
            ],
            root,
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["subsets"]
    selections = {
        row["method"]: Path(row["path"])
        for row in manifest
        if row["method"] == "full"
        or (
            row["method"] in set(VARIANTS) - {"random", "full"}
            and abs(float(row["fraction"]) - 0.2) < 1e-9
        )
    }
    # Random is redrawn from the complete 500-example pool every epoch.
    if "full" in selections:
        selections["random"] = selections["full"]
    missing = set(variants) - set(selections)
    if missing:
        raise ValueError(f"Missing selection files for {sorted(missing)}")

    combined = load_combined(output)
    for variant in variants:
        variant_dir = output / variant
        metrics_path = variant_dir / "epoch_metrics.json"
        training_epochs = (
            (args.epochs + args.full_epoch_scale - 1) // args.full_epoch_scale
            if variant == "full"
            else args.epochs
        )
        records = load_epoch_records(metrics_path, training_epochs)
        if not records or variant in rerun_methods:
            command = [
                    python,
                    "-m",
                    "rig.train_epoch_curve_single_gpu",
                    "--config",
                    str(config_path),
                    *model_args,
                    "--selected",
                    str(selections[variant]),
                    "--data",
                    str(data),
                    "--output-dir",
                    str(variant_dir),
                    "--variant",
                    variant,
                    "--epochs",
                    str(training_epochs),
                    "--num-rollouts",
                    str(args.eval_rollouts),
                ]
            if variant == "random":
                command.extend(["--random-sample-count", "100"])
            run(command, root)
            records = load_epoch_records(metrics_path, training_epochs)
        combined = [row for row in combined if row["variant"] != variant]
        for raw_record in records:
            record = dict(raw_record)
            record["training_epoch"] = int(raw_record["epoch"])
            if variant == "full":
                record["epoch"] = int(raw_record["epoch"]) * args.full_epoch_scale
            combined.append(record)
        save_combined(output, combined)
    plotted_variants = [
        variant
        for variant in VARIANTS
        if variant in variants
        and any(
            row["variant"] == variant and "test_log_loss" in row
            for row in combined
        )
    ]
    plot_epoch_curve(output, combined, plotted_variants)
    print(f"Final epoch curve: {output / 'log_loss_epoch_curve.png'}")


if __name__ == "__main__":
    main()
