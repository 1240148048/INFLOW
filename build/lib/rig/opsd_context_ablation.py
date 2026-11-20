from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .io import Example, load_config, read_jsonl, write_jsonl


METHODS = ("redundancy", "neuron", "mutual", "ours")
LABELS = {
    "redundancy": "Redundancy",
    "neuron": "NEURON",
    "mutual": "Mutual",
    "ours": "Ours",
}
STYLES = {
    "redundancy": dict(color="#1b9e77", marker="D"),
    "neuron": dict(color="#4c78a8", marker="P"),
    "mutual": dict(color="#7a6f9b", marker="X", linestyle="--"),
    "ours": dict(color="#d43d2f", marker="*", linewidth=2.6),
}


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_ints(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1:
        raise ValueError("Context counts must be positive integers.")
    return result


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")
    return methods


def metric_accuracy(path: Path) -> tuple[float, float, float]:
    overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
    average_key = next(key for key in overall if key.startswith("avg_at_"))
    majority_key = next(key for key in overall if key.startswith("maj_at_"))
    return (
        float(overall[average_key]),
        float(overall["rollout_accuracy_std"]),
        float(overall[majority_key]),
    )


def fixed_subset(source_run: Path, method: str) -> list[Example]:
    path = source_run / "subsets" / f"{method}.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"The fixed k=3 subset is missing: {path}. Run the main Qwen experiment first."
        )
    return [Example.from_dict(row) for row in read_jsonl(path)]


def selected_retrieval(
    all_retrieval: Path, fixed: list[Example], destination: Path, context_count: int
) -> None:
    by_uid = {
        example.uid: example for example in (
            Example.from_dict(row) for row in read_jsonl(all_retrieval)
        )
    }
    rows = []
    for source in fixed:
        if source.uid not in by_uid:
            raise KeyError(f"{source.uid} is absent from {all_retrieval}")
        example = by_uid[source.uid]
        if len(example.demo_ids) != context_count:
            raise ValueError(
                f"{example.uid} has {len(example.demo_ids)} contexts, expected {context_count}."
            )
        example.metadata["opsd_selection"] = source.metadata["opsd_selection"]
        rows.append(example.to_dict())
    write_jsonl(destination, rows)


def teacher_ready(path: Path, expected: int, context_count: int) -> bool:
    if not path.exists() or not Path(f"{path}.done").exists():
        return False
    rows = list(read_jsonl(path))
    return len(rows) == expected and all(
        len(row.get("demo_ids", [])) == context_count for row in rows
    )


def remove_stale_teacher(path: Path, context_count: int) -> None:
    if not path.exists():
        return
    rows = list(read_jsonl(path))
    if any(len(row.get("demo_ids", [])) != context_count for row in rows):
        path.unlink()
        Path(f"{path}.done").unlink(missing_ok=True)


def plot(output: Path, records: list[dict], original: float) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for method in METHODS:
        rows = sorted(
            (row for row in records if row["method"] == method),
            key=lambda row: row["context_count"],
        )
        if not rows:
            continue
        x = np.asarray([row["context_count"] for row in rows])
        y = 100.0 * np.asarray([row["accuracy"] for row in rows])
        error = 100.0 * np.asarray([row["accuracy_std"] for row in rows])
        style = dict(STYLES[method])
        style.setdefault("linestyle", "-")
        style.setdefault("linewidth", 2.0)
        axis.plot(x, y, label=LABELS[method], markersize=7.5, **style)
        axis.fill_between(x, y - error, y + error, color=style["color"], alpha=0.10)
    axis.axhline(
        100.0 * original, color="#222222", linestyle="--", linewidth=1.7,
        label="Original",
    )
    axis.set_xticks(sorted({row["context_count"] for row in records}))
    axis.set_xlabel("Retrieved demonstrations per query")
    axis.set_ylabel("Test accuracy (%)")
    axis.grid(axis="y", color="#d9dee3", linewidth=0.8, alpha=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    figure.savefig(output / "context_count_ablation.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "context_count_ablation.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen context-count ablation with fixed k=3 selected targets."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--context-counts", default="1,2,3,4,5")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    counts = parse_ints(args.context_counts)
    methods = parse_methods(args.methods)
    config = load_config(args.config)
    batch_size = int(config["training"]["batch_size"])
    trajectories = args.updates * batch_size
    source_run = Path(args.source_run)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    candidates = Path(args.candidates)
    if not candidates.exists():
        raise FileNotFoundError(candidates)

    original_source = source_run / "eval" / "original" / "metrics.json"
    original_output = output / "eval" / "original" / "metrics.json"
    if original_source.exists() and not original_output.exists():
        original_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_source, original_output)
    if not original_output.exists():
        run([
            sys.executable, "-m", "rig.opsd_evaluate", "--config", args.config,
            "--data", args.data, "--model", args.model,
            "--output", str(original_output.parent),
        ])
    original, _, _ = metric_accuracy(original_output)

    fixed = {method: fixed_subset(source_run, method) for method in methods}
    records: list[dict] = []
    for count in counts:
        count_root = output / f"k-{count}"
        retrieval_dir = count_root / "retrieval"
        retrieval_dir.mkdir(parents=True, exist_ok=True)
        manifest = retrieval_dir / "manifest.json"
        retrieval_valid = False
        if manifest.exists():
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            retrieval_valid = all(
                payload.get(method, {}).get("context_count") == count
                for method in ("neuron", "ours", "mutual")
            )
        if args.force or not retrieval_valid:
            run([
                sys.executable, "-m", "rig.opsd_select", "--input", str(candidates),
                "--output-dir", str(retrieval_dir), "--context-count", str(count),
            ])

        for method in methods:
            method_root = count_root / method
            selected = method_root / "selected.jsonl.gz"
            teacher = method_root / "teacher.jsonl.gz"
            checkpoint = method_root / "checkpoint"
            metrics = method_root / "eval" / "metrics.json"
            method_root.mkdir(parents=True, exist_ok=True)

            if count == 3 and not args.force:
                source_selected = source_run / "subsets" / f"{method}.jsonl.gz"
                teacher = source_selected
            else:
                selected_retrieval(
                    retrieval_dir / f"{method}_all.jsonl.gz",
                    fixed[method], selected, count,
                )
                if args.force:
                    teacher.unlink(missing_ok=True)
                    Path(f"{teacher}.done").unlink(missing_ok=True)
                remove_stale_teacher(teacher, count)
                if not teacher_ready(teacher, len(fixed[method]), count):
                    run([
                        sys.executable, "-m", "rig.opsd_teacher_prepare",
                        "--config", args.config, "--retrieved", str(selected),
                        "--candidates", str(candidates), "--output", str(teacher),
                        "--method", method, "--model", args.model, "--resume",
                    ])

            training_metrics = checkpoint / "training_metrics.json"
            valid_checkpoint = False
            if training_metrics.exists() and not args.force:
                payload = json.loads(training_metrics.read_text(encoding="utf-8"))
                valid_checkpoint = (
                    int(payload.get("optimizer_updates", -1)) == args.updates
                    and int(payload.get("context_count", -1)) == count
                )
            if not valid_checkpoint:
                shutil.rmtree(checkpoint, ignore_errors=True)
                shutil.rmtree(metrics.parent, ignore_errors=True)
                run([
                    sys.executable, "-m", "rig.opsd_train", "--config", args.config,
                    "--selected", str(teacher), "--candidates", str(candidates),
                    "--output-dir", str(checkpoint), "--model", args.model,
                    "--trajectories", str(trajectories), "--final-checkpoint-only",
                ])
            if not metrics.exists() or args.force:
                run([
                    sys.executable, "-m", "rig.opsd_evaluate", "--config", args.config,
                    "--data", args.data, "--model", args.model,
                    "--adapter", str(checkpoint), "--output", str(metrics.parent),
                ])
            accuracy, standard_deviation, majority = metric_accuracy(metrics)
            records.append({
                "method": method,
                "context_count": count,
                "selected_examples": len(fixed[method]),
                "optimizer_updates": args.updates,
                "accuracy": accuracy,
                "accuracy_std": standard_deviation,
                "majority_accuracy": majority,
            })

    with (output / "context_count_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "context_count_results.json").write_text(
        json.dumps({"original_accuracy": original, "records": records}, indent=2),
        encoding="utf-8",
    )
    plot(output, records, original)
    print(f"Saved context-count ablation to {output}")


if __name__ == "__main__":
    main()
