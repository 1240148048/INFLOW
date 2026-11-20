from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .curve_selection import METHODS, parse_fractions
from .io import load_config, read_jsonl


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_").lower()


def run(
    command: list[str],
    root: Path,
    extra_environment: dict[str, str] | None = None,
) -> None:
    print("+ " + " ".join(command), flush=True)
    environment = os.environ.copy()
    source = str(root / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    environment.setdefault("HF_DATASETS_OFFLINE", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    if extra_environment:
        environment.update(extra_environment)
    subprocess.run(command, cwd=root, env=environment, check=True)


def write_resolved_config(
    source: str,
    output: Path,
    model: str | None,
    generation_rollouts: int,
    eval_rollouts: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    config = load_config(source)
    if model:
        config["model"]["name"] = model
    config["generation"]["num_samples"] = generation_rollouts
    config["generation"]["max_new_tokens"] = max_new_tokens
    config["evaluation"]["num_rollouts"] = eval_rollouts
    config["evaluation"]["max_new_tokens"] = max_new_tokens
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def validate_candidates(path: Path, expected: int) -> None:
    rows = list(read_jsonl(path))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} candidates in {path}, found {len(rows)}.")
    required = {
        "sample_summaries",
        "standard_sample_index",
        "standard_predictive_entropy",
        "standard_activation_count",
        "standard_embedding",
        "need_operator",
        "reliability",
    }
    for row in rows:
        missing = required - set(row.get("metadata", {}).get("rig", {}))
        if missing:
            raise ValueError(
                f"Candidate {row.get('uid')} lacks {sorted(missing)}. "
                "It was produced by an older pipeline; regenerate candidates."
            )


def validate_qair_candidates(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    rows = list(read_jsonl(path))
    return len(rows) == expected and all(
        row.get("metadata", {}).get("qair", {}).get("version")
        == "qair_uncertainty_retrieval_v4"
        and row.get("metadata", {}).get("qair", {}).get("gradient_feature")
        and row.get("metadata", {}).get("qair", {}).get(
            "sampled_gradient_features"
        )
        and row.get("metadata", {}).get("qair", {}).get(
            "question_change_feature"
        )
        and float(
            row.get("metadata", {}).get("qair", {}).get("gradient_noise", 0.0)
        )
        > 0.0
        for row in rows
    )


def validate_dataset(path: Path, expected: int) -> None:
    counts = {"train": 0, "test": 0}
    for row in read_jsonl(path):
        if row.get("split") in counts:
            counts[row["split"]] += 1
    if counts != {"train": expected, "test": expected}:
        raise ValueError(
            f"Expected {expected}/{expected} train/test rows in {path}, found {counts}."
        )


def metric_record(
    method: str,
    fraction: float,
    count: int,
    metrics_path: Path,
    training_path: Path | None,
    adapter: Path | None,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]
    training = (
        json.loads(training_path.read_text(encoding="utf-8"))
        if training_path is not None
        else {}
    )
    rollout_key = next(key for key in metrics if key.startswith("avg_at_"))
    majority_key = next(key for key in metrics if key.startswith("maj_at_"))
    return {
        "method": method,
        "fraction": fraction,
        "selected_count": count,
        "mean_accuracy": metrics[rollout_key],
        "rollout_accuracy_std": metrics["rollout_accuracy_std"],
        "rollout_index_accuracies": metrics["rollout_index_accuracies"],
        "majority_accuracy": metrics[majority_key],
        "training_seconds": float(training.get("training_seconds", 0.0)),
        "optimizer_steps": int(training.get("optimizer_steps", 0)),
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "metrics": str(metrics_path.resolve()),
    }


def save_results(output_dir: Path, records: list[dict[str, Any]]) -> None:
    records = sorted(records, key=lambda row: (row["method"], row["fraction"]))
    (output_dir / "curve_results.json").write_text(
        json.dumps(
            {
                "records": records,
                "total_training_seconds": sum(row["training_seconds"] for row in records),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "curve_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(
            dict.fromkeys(key for record in records for key in record)
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["rollout_index_accuracies"] = json.dumps(row["rollout_index_accuracies"])
            writer.writerow(row)


def load_existing_records(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "curve_results.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def upsert_record(
    records: list[dict[str, Any]], record: dict[str, Any]
) -> None:
    key = (record["method"], round(float(record["fraction"]), 12))
    for index, existing in enumerate(records):
        existing_key = (
            existing["method"],
            round(float(existing["fraction"]), 12),
        )
        if existing_key == key:
            records[index] = record
            return
    records.append(record)


def plot_results(output_dir: Path, records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    style = {
        "random": ("#8A9496", "o", "-"),
        "difficulty": ("#E69F00", "s", "-"),
        "uncertainty": ("#9B59B6", "^", "-"),
        "redundancy": ("#009E73", "D", "-"),
        "activation": ("#4C78A8", "P", "-"),
        "ours": ("#D13C32", "*", "-"),
        "ours-inverse": ("#4D5966", "X", "--"),
    }
    original = next(row for row in records if row["method"] == "original")
    full = next((row for row in records if row["method"] == "full"), None)
    with plt.rc_context({"font.size": 10.5, "axes.labelsize": 11.5}):
        figure, axis = plt.subplots(figsize=(9.2, 5.3))
        available = {row["method"] for row in records if row["method"] in METHODS}
        bounds = []
        for method in METHODS:
            if method not in available:
                continue
            rows = sorted(
                [row for row in records if row["method"] == method],
                key=lambda row: row["fraction"],
            )
            x = [100 * row["fraction"] for row in rows]
            y = [100 * row["mean_accuracy"] for row in rows]
            error = [100 * row["rollout_accuracy_std"] for row in rows]
            color, marker, linestyle = style[method]
            emphasis = method == "ours"
            axis.errorbar(
                x, y, yerr=error, fmt="none", ecolor=color,
                elinewidth=0.8, capsize=2, alpha=0.2, zorder=1,
            )
            axis.plot(
                x, y, label=method, color=color, marker=marker,
                linestyle=linestyle, linewidth=2.7 if emphasis else 1.55,
                markersize=8 if emphasis else 5.2,
                markeredgecolor="white", markeredgewidth=0.6,
                alpha=1.0 if emphasis else 0.82,
                zorder=4 if emphasis else 2,
            )
            bounds.extend(value for mean, sd in zip(y, error) for value in (mean - sd, mean + sd))

        original_accuracy = 100 * original["mean_accuracy"]
        full_accuracy = 100 * full["mean_accuracy"] if full is not None else None
        axis.axhline(
            original_accuracy, color="#222222", linestyle="--",
            linewidth=1.25, label="original", zorder=0,
        )
        if full_accuracy is not None:
            axis.axhline(
                full_accuracy, color="#6F777A", linestyle=":",
                linewidth=1.4, label="full data", zorder=0,
            )
        bounds.append(original_accuracy)
        if full_accuracy is not None:
            bounds.append(full_accuracy)
        lower, upper = min(bounds), max(bounds)
        margin = max(0.12, 0.08 * (upper - lower))
        axis.set_ylim(lower - margin, upper + margin)
        axis.set_xlim(7, 93)
        axis.set_xlabel("Selected training data (%)")
        axis.set_ylabel("Rollout accuracy (%)")
        axis.set_xticks(list(range(10, 100, 10)))
        axis.grid(axis="y", color="#D8DDDF", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            ncol=3, frameon=False, loc="lower center",
            bbox_to_anchor=(0.5, 1.01), columnspacing=1.4,
            handlelength=2.4,
        )
        figure.text(
            0.99, 0.012,
            "Error bars: SD across three rollout-index accuracies.",
            ha="right", va="bottom", fontsize=8.5, color="#62696B",
        )
        figure.tight_layout(rect=(0, 0.035, 1, 0.91))
        figure.savefig(output_dir / "accuracy_selection_curve.png", dpi=240)
        figure.savefig(output_dir / "accuracy_selection_curve.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RIG selection curve")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-file")
    parser.add_argument("--generation-rollouts", type=int, default=5)
    parser.add_argument("--eval-rollouts", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=10)
    parser.add_argument("--expected-train", type=int, default=500)
    parser.add_argument("--fractions", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--methods", default=",".join((*METHODS, "full")))
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--rebuild-subsets", action="store_true")
    parser.add_argument("--qair-hidden-dim", type=int, default=16)
    parser.add_argument("--qair-residual-dim", type=int, default=8)
    parser.add_argument("--qair-noise-floor", type=float, default=1e-6)
    parser.add_argument("--rerun-methods", default="")
    args = parser.parse_args()
    if args.generation_rollouts < 2:
        raise ValueError("At least two generation rollouts are required for difficulty.")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - (set(METHODS) | {"full"})
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    selection_methods = [method for method in methods if method != "full"]
    rerun_methods = {
        item.strip() for item in args.rerun_methods.split(",") if item.strip()
    }
    unknown_reruns = rerun_methods - (set(METHODS) | {"full", "original"})
    if unknown_reruns:
        raise ValueError(f"Unknown rerun methods: {sorted(unknown_reruns)}")
    fractions = parse_fractions(args.fractions)
    root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"Prepared dataset not found; building it at {data_path}", flush=True)
        run(
            ["bash", "scripts/download_and_prepare_comprehensive.sh"],
            root,
            {"COMPREHENSIVE_OUTPUT": str(data_path)},
        )
    else:
        print(f"Reusing prepared dataset: {data_path}", flush=True)
    validate_dataset(data_path, args.expected_train)
    config_path = output_dir / "resolved_config.json"
    config = write_resolved_config(
        args.config,
        config_path,
        args.model,
        args.generation_rollouts,
        args.eval_rollouts,
        args.max_new_tokens,
    )
    python = sys.executable
    model_args = ["--model", config["model"]["name"]]
    candidates = Path(args.candidate_file).resolve() if args.candidate_file else output_dir / "candidates" / "standard.jsonl.gz"
    first = (
        candidates.with_name(candidates.name + ".first.jsonl.gz")
        if args.candidate_file
        else output_dir / "candidates" / "first.jsonl.gz"
    )
    if candidates.exists():
        validate_candidates(candidates, args.expected_train)
        print(f"Reusing prepared candidates: {candidates}", flush=True)
    else:
        print(f"Candidates not found; generating: {candidates}", flush=True)
        run(
            [
                python, "-m", "rig.selection", "prepare",
                "--config", str(config_path), "--input", str(data_path),
                "--first-output", str(first), "--output", str(candidates), "--resume",
            ],
            root,
        )
    validate_candidates(candidates, args.expected_train)
    selection_candidates = candidates
    if set(selection_methods) & {"ours", "ours-inverse"}:
        qair_candidates = output_dir / "candidates" / "standard_qair.jsonl.gz"
        if validate_qair_candidates(qair_candidates, args.expected_train):
            print(f"Reusing QA information candidates: {qair_candidates}", flush=True)
        else:
            run(
                [
                    python, "-m", "rig.qa_information", "extract",
                    "--config", str(config_path), "--input", str(candidates),
                    "--output", str(qair_candidates), *model_args,
                    "--hidden-dim", str(args.qair_hidden_dim),
                    "--residual-dim", str(args.qair_residual_dim),
                    "--noise-floor", str(args.qair_noise_floor),
                ],
                root,
            )
        if not validate_qair_candidates(qair_candidates, args.expected_train):
            raise ValueError(f"Invalid QA information candidate file: {qair_candidates}")
        selection_candidates = qair_candidates
    subsets = output_dir / "subsets"
    manifest_path = subsets / "manifest.json"
    compatible_manifest = False
    if manifest_path.exists() and not args.rebuild_subsets:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        diagnostics = manifest_payload.get("diagnostics", {})
        manifest_methods = set(diagnostics.get("methods", []))
        compatible_manifest = set(selection_methods).issubset(manifest_methods)
        if set(selection_methods) & {"ours", "ours-inverse"}:
            ours_info = diagnostics.get("ours") or diagnostics.get("ours_inverse") or {}
            compatible_manifest = (
                compatible_manifest
                and ours_info.get("version")
                == "uncertainty_directed_retrieval_v4"
            )
    if compatible_manifest:
        print(f"Reusing prepared subsets: {subsets}", flush=True)
    else:
        run(
            [
                python, "-m", "rig.curve_selection", "--input", str(selection_candidates),
                "--output-dir", str(subsets), "--fractions", args.fractions,
                "--seed", str(config["seed"]), "--rank", str(args.rank),
                "--ridge", str(args.ridge), "--noise", str(args.noise),
                "--methods", ",".join(selection_methods),
            ],
            root,
        )
    if args.selection_only:
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["subsets"]
    eval_root = output_dir / "eval"
    checkpoint_root = output_dir / "checkpoints"
    base_eval = eval_root / "original"
    if not (base_eval / "metrics.json").exists():
        run(
            [python, "-m", "rig.evaluate_single_gpu", "--config", str(config_path),
             *model_args, "--data", str(data_path), "--output", str(base_eval),
             "--num-rollouts", str(args.eval_rollouts)],
            root,
        )
    records = load_existing_records(output_dir)
    upsert_record(
        records,
        metric_record("original", 0.0, 0, base_eval / "metrics.json", None, None),
    )
    runnable = sorted(
        [row for row in manifest if row["method"] in methods],
        key=lambda row: (row["method"] != "full", row["method"], row["fraction"]),
    )
    for item in runnable:
        method = item["method"]
        percent = int(round(100 * item["fraction"]))
        run_name = f"{method}_p{percent:03d}"
        adapter = checkpoint_root / run_name
        evaluation = eval_root / run_name
        training_metrics = adapter / "training_metrics.json"
        if not training_metrics.exists() or method in rerun_methods:
            run(
                [python, "-m", "rig.train_single_gpu", "--config", str(config_path),
                 *model_args, "--selected", item["path"], "--epochs", str(args.epochs),
                 "--output-dir", str(adapter)],
                root,
            )
        if not (evaluation / "metrics.json").exists() or method in rerun_methods:
            run(
                [python, "-m", "rig.evaluate_single_gpu", "--config", str(config_path),
                 *model_args, "--data", str(data_path), "--adapter", str(adapter),
                 "--output", str(evaluation), "--num-rollouts", str(args.eval_rollouts)],
                root,
            )
        upsert_record(
            records,
            metric_record(method, item["fraction"], item["count"], evaluation / "metrics.json", training_metrics, adapter),
        )
        save_results(output_dir, records)
    save_results(output_dir, records)
    plot_results(output_dir, records)
    print(f"Final curve: {output_dir / 'accuracy_selection_curve.png'}")


if __name__ == "__main__":
    main()
