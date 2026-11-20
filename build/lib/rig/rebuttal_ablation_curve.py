"""Complete the five new MC ablation curves from existing selected subsets.

This entry point never invokes candidate generation, teacher preparation, or
subset selection. It uses the original 500-update schedule and evaluates each
saved adapter at updates 50, 100, ..., 500 with the original evaluation config.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

from .io import read_jsonl


STEPS = tuple(range(50, 501, 50))
VARIANTS = (
    "last_layer_certainty", "last_layer_no_certainty", "sample_1_1_js",
    "sample_5_5_js", "sample_1_5_hard_flip",
)
EXISTING = ("ours", "no_certainty", "no_similarity", "random20", "original")
FIELDS = ("method", "optimizer_update", "avg_percent", "sd_percent")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def adapter_exists(path):
    root = path / "student" if (path / "student/adapter_config.json").exists() else path
    return (root / "adapter_config.json").is_file() and any(
        (root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def metric_record(path, method, step):
    record = dict(method=method, optimizer_update=step, avg_percent="", sd_percent="",
                  status="pending", source=str(path))
    if path.is_file():
        overall = read_json(path)["overall"]
        mean, sd = float(overall["avg_at_3"]), float(overall["rollout_accuracy_std"])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (mean, sd)):
            raise ValueError(f"Invalid fraction-valued Avg@3/SD in {path}")
        record.update(avg_percent=f"{100 * mean:.2f}", sd_percent=f"{100 * sd:.2f}",
                      status="measured")
    return record


def collect_results(previous, output, variants=VARIANTS, existing_csv=None):
    """Keep old comparison curves; never fill new curves from old endpoint CSVs."""
    rows = []
    csv_path = existing_csv or previous / "all_results.csv"
    provenance_path = previous / "all_results_sources.json"
    provenance = {}
    if provenance_path.is_file():
        provenance = {(r["method"], int(r["optimizer_update"])): r
                      for r in read_json(provenance_path)}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            seen = set()
            for row in csv.DictReader(handle):
                method, step = row["method"], int(row["optimizer_update"])
                if method not in EXISTING:
                    continue
                key = method, step
                if key in seen:
                    raise ValueError(f"Duplicate existing point {key} in {csv_path}")
                seen.add(key)
                origin = provenance.get(key, {})
                rows.append(dict(
                    method=method, optimizer_update=step,
                    avg_percent=row["avg_percent"], sd_percent=row["sd_percent"],
                    status=origin.get("status", "provided_csv"),
                    source=origin.get("source", str(csv_path)),
                ))
    for method in variants:
        for step in STEPS:
            path = output / "eval" / method / f"update-{step}" / "metrics.json"
            rows.append(metric_record(path, method, step))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)
    save_json(output / "all_results_sources.json", rows)
    save_json(output / "curve_progress.json", {
        method: {
            "completed": [r["optimizer_update"] for r in rows
                          if r["method"] == method and r["status"] == "measured"],
            "pending": [r["optimizer_update"] for r in rows
                        if r["method"] == method and r["status"] == "pending"],
        } for method in variants
    })
    return rows


def plot_results(rows, output):
    """Use gaps for missing checkpoints; never interpolate new ablations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "ours": ("Ours", "#D55E00", "*", "-"),
        "no_certainty": ("w/o Certainty", "#0072B2", "o", "--"),
        "no_similarity": ("w/o Similarity", "#009E73", "D", "-."),
        "random20": ("Random 20%", "#777777", "X", ":"),
        "last_layer_certainty": ("Last layer + certainty", "#0072B2", "s", "--"),
        "last_layer_no_certainty": ("Last layer w/o certainty", "#009E73", "^", "-."),
        "sample_1_1_js": ("1/1 sampling + JS", "#0072B2", "o", "--"),
        "sample_5_5_js": ("5/5 sampling + JS", "#CC79A7", "D", "-."),
        "sample_1_5_hard_flip": ("1/5 sampling + hard flip", "#009E73", "^", ":"),
    }
    panels = (
        ("(a) Component ablation", ("ours", "no_certainty", "no_similarity", "random20")),
        ("(b) Retrieval layers", ("ours", "last_layer_certainty", "last_layer_no_certainty")),
        ("(c) Belief estimation and selection", ("ours", "sample_1_1_js", "sample_5_5_js", "sample_1_5_hard_flip")),
    )
    lookup = {(r["method"], r["optimizer_update"]): r for r in rows}
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.5), sharey=True)
    baseline = lookup.get(("original", 0), {}).get("avg_percent", "")
    for ax, (title, methods) in zip(axes, panels):
        missing = []
        for method in methods:
            label, color, marker, linestyle = styles[method]
            values = [lookup.get((method, step), {}) for step in STEPS]
            if not any(v.get("avg_percent", "") != "" for v in values):
                missing.append(label)
                continue
            y = [float(v["avg_percent"]) if v.get("avg_percent", "") != "" else math.nan
                 for v in values]
            sd = [float(v["sd_percent"]) if v.get("sd_percent", "") != "" else math.nan
                  for v in values]
            if any(not math.isfinite(v) for v in y):
                missing.append(label)
            ax.plot(STEPS, y, label=label, color=color, marker=marker, linestyle=linestyle,
                    linewidth=2.3 if method == "ours" else 1.7,
                    markersize=8 if method == "ours" else 4.5)
            ax.fill_between(STEPS, [a-b for a, b in zip(y, sd)],
                            [a+b for a, b in zip(y, sd)], color=color, alpha=.09, linewidth=0)
        if baseline != "":
            ax.axhline(float(baseline), color="#333333", linestyle="--", linewidth=1.2,
                       label="Original")
        ax.set(title=title, xlabel="Optimizer update", xticks=STEPS)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.22), frameon=False, fontsize=9)
        if missing:
            ax.text(.02, .02, "Incomplete checkpoints", transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("Avg@3 accuracy (%)")
    fig.subplots_adjust(left=.055, right=.99, top=.88, bottom=.34, wspace=.12)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"supplementary_ablation_curves.{extension}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def run_command(*arguments):
    print("[stage] " + " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, "-u", "-m", *map(str, arguments)], check=True)


def execute(args):
    previous = args.previous_run.resolve()
    output = (args.output or previous / "curve_500").resolve()
    if output == previous:
        raise ValueError("Use a separate output directory to preserve the old 250-update run")
    variants = tuple(args.variants)
    if args.action in {"report", "plot"}:
        rows = collect_results(previous, output, variants, args.existing_results)
        if args.action == "plot":
            plot_results(rows, output)
        return

    original = read_json(previous / "experiment.json")
    if args.config:
        from .io import load_config
        config = load_config(args.config)
        if config != original["config"]:
            raise ValueError("Config differs from the original subset experiment")
    else:
        config = original["config"]
    model = args.model or original["model"]
    data = Path(args.data or original["data"])
    source = Path(args.source_run or original["source"])
    subsets = args.subsets_dir or previous / "subsets"
    candidates = source / "candidates/shared.jsonl.gz"
    if int(original.get("training_updates", 500)) != 500:
        raise ValueError("Expected the original 500-update learning-rate horizon")
    if int(config["evaluation"]["num_rollouts"]) != 3:
        raise ValueError("Use the original three-rollout evaluation config")
    for path in (data, candidates, *(subsets / f"{v}.jsonl.gz" for v in variants)):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing artifact not found: {path}; nothing will be regenerated")
    pool = {row["uid"] for row in read_jsonl(candidates)}
    for variant in variants:
        selected = list(read_jsonl(subsets / f"{variant}.jsonl.gz"))
        if not selected or len({r["uid"] for r in selected}) != len(selected):
            raise ValueError(f"Empty subset or duplicate IDs: {variant}")
        if any(r.get("split") != "train" or not r.get("rollout")
               or r["uid"] not in pool or not set(r.get("demo_ids", [])) <= pool for r in selected):
            raise ValueError(f"Invalid existing training rows/demonstrations: {variant}")
    batch = int(config["training"]["batch_size"])
    if batch < 1:
        raise ValueError("Training batch size must be positive")
    identity = dict(previous_run=str(previous), source_run=str(source.resolve()),
                    subsets_dir=str(subsets.resolve()), model=model, data=str(data.resolve()),
                    config=config, updates=500, eval_updates=list(STEPS), variants=list(variants))
    manifest = output / "curve_experiment.json"
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Curve configuration changed; use a new output directory")
    if args.action == "plan":
        print(json.dumps(identity, indent=2))
        return
    save_json(manifest, identity)
    config_path = output / "original_config.json"
    save_json(config_path, config)
    collect_results(previous, output, variants, args.existing_results)
    for variant in variants:
        target = output / "checkpoints" / variant
        complete = target / "training_metrics.json"
        if not complete.exists():
            run_command("rig.opsd_train", "--config", config_path, "--model", model,
                        "--selected", subsets / f"{variant}.jsonl.gz",
                        "--candidates", candidates, "--output-dir", target,
                        "--trajectories", 500 * batch,
                        "--checkpoint-every-trajectories", 50 * batch)
        trained = read_json(complete)
        if trained["optimizer_updates"] != 500 or trained["scheduler_horizon_updates"] != 500:
            raise ValueError(f"Expected a full 500-update curve in {target}")
        for step in STEPS:
            adapter = target / f"checkpoint-{step * batch}"
            if not adapter_exists(adapter):
                raise FileNotFoundError(f"Missing saved adapter at update {step}: {adapter}")
            evaluation = output / "eval" / variant / f"update-{step}"
            if not (evaluation / "metrics.json").is_file():
                run_command("rig.opsd_evaluate", "--config", config_path, "--model", model,
                            "--data", data, "--adapter", adapter, "--output", evaluation)
            collect_results(previous, output, variants, args.existing_results)
    rows = collect_results(previous, output, variants, args.existing_results)
    plot_results(rows, output)
    print(f"Completed {len(variants)} variants x 10 checkpoints. Results: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--subsets-dir", type=Path)
    parser.add_argument("--existing-results", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--action", choices=("run", "plan", "report", "plot"), default="run")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
