"""Render measured Avg@3 update trajectories in the paper's Figure 2 layout."""

import csv
import json
import statistics
from pathlib import Path

from .paper_baselines import PAPER_METHODS


STYLES = {
    "full": ("Full", "#829db6", ":"),
    "redundancy": ("Redundancy", "#16866b", "-"),
    "consistency": ("Consistency", "#a564b8", "-"),
    "neuron": ("NEURON", "#4477aa", "-"),
    "mutual": ("Mutual", "#736795", "--"),
    "js": ("InFlow", "#d9472e", "-"),
    "original": ("Original", "#333333", "--"),
}


def read_curve(run, methods=PAPER_METHODS):
    run = Path(run)
    config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
    training = config["training"]
    updates = sorted(set(range(training["checkpoint_every"], training["updates"] + 1,
                               training["checkpoint_every"])) | {training["updates"]})
    model = Path(config["model"]["name"]).name
    curve = []
    evaluation_ids = set()
    for method in (*methods, "original"):
        for update in updates if method != "original" else [0]:
            folder = run / "curve" / f"step-{update:06d}" if update else run
            path = folder / "metrics" / f"{method}.json"
            item = json.loads(path.read_text(encoding="utf-8"))
            values = item["rollout_index_accuracies"]
            if len(values) != 3 or item["optimizer_update"] != update or item["method"] != method:
                raise ValueError(f"Unexpected Figure 2 data in {path}")
            evaluation_ids.add(item["evaluation_id"])
            average = statistics.mean(values)
            if abs(average - item["overall_accuracy"]) > 1e-7:
                raise ValueError(f"Rollout average does not match Overall in {path}")
            curve.append({"model": model, "method": method, "optimizer_update": update,
                          "mean_accuracy": average, "rollout_sd": statistics.pstdev(values),
                          **{f"rollout_{i + 1}": value for i, value in enumerate(values)}})
    if len(evaluation_ids) != 1:
        raise ValueError(f"Methods do not share the same held-out questions in {run}")
    return model, updates, curve


def plot_curves(runs, output, methods=PAPER_METHODS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not 1 <= len(runs) <= 4:
        raise ValueError("Figure 2 accepts one to four backbone runs")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(PAPER_METHODS):
        raise ValueError("Select distinct trainable Figure 2 methods. Original is always included.")
    inputs = [read_curve(run, methods) for run in runs]
    if len({x[0] for x in inputs}) != len(inputs):
        raise ValueError("Each panel must have a distinct backbone")
    columns = 1 if len(inputs) == 1 else 2
    rows = (len(inputs) + columns - 1) // columns
    fig, grid = plt.subplots(rows, columns, figsize=(5.5 * columns, 3.05 * rows), squeeze=False)
    for i, (ax, (model, updates, data)) in enumerate(zip(grid.flat, inputs)):
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7ea", linewidth=.55)
        ax.spines[["top", "right"]].set_visible(False)
        for method in (*methods, "original"):
            values = [r for r in data if r["method"] == method]
            label, color, line = STYLES[method]
            if method == "original":
                ax.axhline(values[0]["mean_accuracy"], color=color, linestyle=line, linewidth=1, label=label)
                continue
            x = [r["optimizer_update"] for r in values]
            y = [r["mean_accuracy"] for r in values]
            ax.fill_between(x, [r["mean_accuracy"] - r["rollout_sd"] for r in values],
                            [r["mean_accuracy"] + r["rollout_sd"] for r in values], color=color, alpha=.12)
            ax.plot(x, y, color=color, linestyle=line, marker="o", markersize=2.1,
                    linewidth=1.45 if method == "js" else .95, label=label)
        ax.set(xlabel="Optimizer update", ylabel="Rollout accuracy (%)", xticks=updates)
        ax.tick_params(labelsize=8)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, .15), mode="expand",
                  ncol=4, frameon=False, fontsize=7, borderaxespad=0)
        ax.text(.5, -.40, f"({chr(97 + i)}) {model}", ha="center", transform=ax.transAxes, fontsize=10)
    for ax in list(grid.flat)[len(inputs):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=1.9)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(destination.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    data = [row for _, _, curve in inputs for row in curve]
    with destination.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved Figure 2 trajectories to {destination}.pdf/.png/.csv")
