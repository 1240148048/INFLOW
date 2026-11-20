from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def accuracy(path: Path) -> tuple[float, float]:
    overall = json.loads(path.read_text(encoding="utf-8"))["overall"]
    key = next(name for name in overall if name.startswith("avg_at_"))
    return float(overall[key]), float(overall["rollout_accuracy_std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--ablation-run", required=True)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()

    source = Path(args.source_run)
    output = Path(args.ablation_run)
    points = list(range(args.eval_every, args.total_updates + 1, args.eval_every))
    methods = {
        "ours": ("Ours", source, "#d43d2f", "*", "-"),
        "no_certainty": ("w/o Certainty", output, "#4c78a8", "o", "-"),
        "no_similarity": ("w/o Similarity", output, "#1b9e77", "D", "-"),
        "random20": ("w/o JS (Random 20%)", output, "#7a6f9b", "X", "--"),
    }
    records = []
    for method, (_label, root, _color, _marker, _linestyle) in methods.items():
        for point in points:
            path = root / "eval" / method / f"update-{point}" / "metrics.json"
            if not path.exists():
                continue
            mean, std = accuracy(path)
            records.append(
                {
                    "method": method,
                    "optimizer_update": point,
                    "accuracy": mean,
                    "accuracy_std": std,
                }
            )

    if not records:
        raise FileNotFoundError("No ablation evaluation metrics were found.")
    with (output / "module_ablation_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    original, _ = accuracy(source / "eval" / "original" / "metrics.json")
    fig, axis = plt.subplots(figsize=(9.2, 5.3))
    for method, (label, _root, color, marker, linestyle) in methods.items():
        values = [row for row in records if row["method"] == method]
        if not values:
            continue
        x = np.asarray([row["optimizer_update"] for row in values])
        y = 100.0 * np.asarray([row["accuracy"] for row in values])
        sd = 100.0 * np.asarray([row["accuracy_std"] for row in values])
        axis.plot(
            x,
            y,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.4 if method == "ours" else 1.9,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.55,
            label=label,
        )
        axis.fill_between(x, y - sd, y + sd, color=color, alpha=0.09, linewidth=0)
    axis.axhline(
        100.0 * original,
        color="#222222",
        linestyle="--",
        linewidth=1.6,
        label="Original",
    )
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel("Rollout accuracy (%)")
    axis.set_xticks(points)
    axis.grid(axis="y", color="#dfe4e8", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.9))
    fig.savefig(output / "module_ablation_curve.png", dpi=260, bbox_inches="tight")
    fig.savefig(output / "module_ablation_curve.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results to {output / 'module_ablation_results.csv'}")
    print(f"Saved curve to {output / 'module_ablation_curve.png'}")


if __name__ == "__main__":
    main()
