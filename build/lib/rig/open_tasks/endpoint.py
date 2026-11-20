"""Endpoint and measured-budget tables for the three-task experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .data import dump
from .paper_baselines import PAPER_METHODS, family
from .pipeline import records, teacher_path
from .training import update_checkpoint


METHODS = ("original", *PAPER_METHODS)
DISPLAY = ("Original", "Full", "Redundancy", "Consistency", "NEURON", "Mutual", "INFLOW")
TASKS = (("code", "MBPP"), ("coqa", "CoQA"), ("squad", "SQuAD"), ("overall", "Overall"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def initialize(output, seconds):
    import torch

    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("A positive shared wall-clock budget is required")
    config = read(output / "resolved_config.json")
    if config["training"]["updates"] != 250 or config["evaluation"]["num_samples"] != 3:
        raise ValueError("The endpoint table requires 250 updates and three evaluation rollouts")
    hardware = {"gpu": torch.cuda.get_device_name(0), "pytorch": str(torch.__version__),
                "cuda": torch.version.cuda}
    policy = {"budget_seconds": seconds, "update_cap": 250, "hardware": hardware,
              "accounting": "Standalone adaptation time reconstructed from measured stages on the same device. Shared preparation is charged to each method. Dataset construction, evaluation, idle time and orchestration outside the measured stages are excluded."}
    path = output / "wall_clock_policy.json"
    if path.exists() and read(path) != policy:
        raise ValueError("Budget or hardware changed; use a new output directory")
    if not path.exists():
        dump(path, policy)


def preparation_cost(output, method, initial, selections):
    name = family(method)
    teacher = records(teacher_path(output, name))
    if not initial or set(initial) != set(teacher):
        raise ValueError(f"Incomplete timing data for {method}")
    first = [row["timing"] for row in initial.values()]
    contextual = [row["timing"] for row in teacher.values()]
    both = first + contextual
    return {
        "preparation_loading_seconds": read(output / "preparation_loading.json")["seconds"],
        "candidate_generation_seconds": sum(row["generation_seconds"] for row in first),
        "retrieval_seconds": read(output / f"retrieval.{name}.timing.json")["seconds"]
        + sum(row["feature_seconds"] for row in both),
        "teacher_generation_seconds": sum(row["generation_seconds"] for row in contextual),
        "belief_estimation_seconds": sum(row["belief_seconds"] for row in both),
        "selection_seconds": selections[method]["seconds"],
    }


def budget_plan(output):
    policy = read(output / "wall_clock_policy.json")
    cap = policy["update_cap"]
    initial = records(output / "initial.jsonl")
    selections = read(output / "selection_manifest.json")
    result = {"original": {"update": 0, "adaptation_seconds": 0.0, "status": "unadapted"}}
    costs = {"original": {"method": "original", "adaptation_seconds": 0.0}}
    for method in PAPER_METHODS:
        prep = preparation_cost(output, method, initial, selections)
        ready = []
        for path in (output / "checkpoints" / method / "updates").glob("step-*/ready_time.json"):
            update = int(path.parent.name.split("-")[-1])
            if update <= cap:
                marker = read(update_checkpoint(output, method, update) / "checkpoint.json")
                if marker != {"method": method, "optimizer_update": update}:
                    raise ValueError(f"Unexpected timed checkpoint: {path}")
                timing = read(path)
                ready.append((update, sum(prep.values()) + sum(timing.values()), timing))
        final = next((row for row in ready if row[0] == cap), None)
        if final is None:
            raise ValueError(f"Missing timed {cap}-update checkpoint for {method}")
        eligible = [row for row in ready if row[1] <= policy["budget_seconds"]]
        if eligible:
            update, elapsed, _ = max(eligible, key=lambda row: row[0])
            result[method] = {"update": update, "adaptation_seconds": elapsed, "status": "adapted"}
        else:
            result[method] = {"update": 0, "adaptation_seconds": None, "status": "no checkpoint within budget"}
        costs[method] = {"method": method, **prep, **final[2], "adaptation_seconds": final[1]}
    dump(output / "wall_clock_plan.json", {"policy": policy, "methods": result})
    dump(output / "endpoint_costs.json", costs)
    return result


def csv_table(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def tex_table(path, header, rows, notes, text_columns=1):
    alignment = "l" * text_columns + "r" * (len(header) - text_columns)
    lines = [f"% {notes}", r"\begin{tabular}{" + alignment + "}", r"\toprule"]
    lines.extend([" & ".join(header) + r" \\", r"\midrule"])
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def accuracy(item, task):
    if task == "overall":
        return item["overall_accuracy"], item["rollout_accuracy_sd"]
    row = item["by_task"][task]
    return row["pass_at_1"] if task == "code" else row["exact_match"], row["rollout_accuracy_sd"]


def accuracy_table(output, name, metrics, extra_rows, note):
    header = ["Model", "Task", *DISPLAY]
    csv_rows, tex_rows = [], []
    model = Path(metrics["original"]["model"]).name
    for task, label in TASKS:
        values = [accuracy(metrics[m], task) if m in metrics else None for m in METHODS]
        ranks = sorted({round(value[0], 1) for value in values if value is not None}, reverse=True)
        csv_row = [model, label]
        tex_row = [model.replace("_", r"\_") if not tex_rows else "", r"\textbf{Overall}" if task == "overall" else label]
        for value in values:
            if value is None:
                csv_row.append("")
                tex_row.append("--")
                continue
            mean, sd = value
            csv_row.append(f"{mean:.1f} +/- {sd:.2f}")
            cell = f"{mean:.1f} $\\pm$ {sd:.2f}"
            rank = ranks.index(round(mean, 1))
            if rank == 0:
                cell = r"\textbf{" + cell + "}"
            elif rank == 1:
                cell = r"\underline{" + cell + "}"
            tex_row.append(cell)
        csv_rows.append(csv_row)
        tex_rows.append(tex_row)
    csv_table(output / f"{name}.csv", header, csv_rows + [[model, *row] for row in extra_rows])
    tex_table(output / f"{name}.tex", header, tex_rows + [["", *row] for row in extra_rows], note, text_columns=2)


def write_tables(output):
    plan = read(output / "wall_clock_plan.json")
    costs = read(output / "endpoint_costs.json")
    end = {method: read(output / "metrics" / f"{method}.json") for method in METHODS}
    if any(set(row["by_task"]) != {"code", "coqa", "squad"} for row in end.values()):
        raise ValueError("Expected MBPP, CoQA and SQuAD endpoint metrics")
    matched = {"original": end["original"]}
    for method in PAPER_METHODS:
        update = plan["methods"][method]["update"]
        if update == plan["policy"]["update_cap"]:
            matched[method] = end[method]
        elif update:
            row = read(output / "curve" / f"step-{update:06d}" / "metrics" / f"{method}.json")
            if row["optimizer_update"] != update:
                raise ValueError(f"Wrong budget checkpoint for {method}")
            matched[method] = row
    if len({row["evaluation_id"] for row in [*end.values(), *matched.values()]}) != 1:
        raise ValueError("All table entries must use the same held-out questions")
    accuracy_table(output, "endpoint_table", end,
                   [["Adaptation (min)", *[f"{costs[m]['adaptation_seconds']/60:.2f}" for m in METHODS]]],
                   "250 updates; Avg@3 +/- rollout-index SD; MBPP pass@1, CoQA/SQuAD EM; Overall question weighted.")
    seconds = plan["policy"]["budget_seconds"]
    steps = ["Updates", *[str(plan["methods"][m]["update"]) if m in matched else "--" for m in METHODS]]
    elapsed = ["Adaptation (min)", *[f"{plan['methods'][m]['adaptation_seconds']/60:.2f}" if m in matched else "--" for m in METHODS]]
    accuracy_table(output, "wall_clock_table", matched, [steps, elapsed],
                   f"Shared adaptation budget: {seconds/60:g} minutes, capped at 250 updates. Dash: no saved adapter within budget. Evaluation excluded.")
    fields = [("preparation_loading_seconds", "Preparation loading"),
              ("candidate_generation_seconds", "Candidate generation"), ("retrieval_seconds", "Retrieval and features"),
              ("teacher_generation_seconds", "Teacher generation"), ("belief_estimation_seconds", "Belief estimation"),
              ("selection_seconds", "Selection"), ("startup_seconds", "Training startup"),
              ("distillation_seconds", "Distillation and checkpointing"), ("adaptation_seconds", "Total adaptation")]
    rows = [[label, *[f"{costs[m].get(key, 0)/60:.2f}" for m in METHODS]] for key, label in fields]
    rows.append(["Evaluation generation", *[f"{end[m]['generation_seconds']/60:.2f}" for m in METHODS]])
    csv_table(output / "cost_table.csv", ["Stage (min)", *DISPLAY], rows)
    tex_table(output / "cost_table.tex", ["Stage (min)", *DISPLAY], rows,
              "Measured times. Shared preparation is charged in full to each method; evaluation generation is separate.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("init", "plan", "report"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.action == "init":
        if args.seconds is None:
            parser.error("init requires --seconds")
        initialize(args.output, args.seconds)
    elif args.action == "plan":
        for method, item in budget_plan(args.output).items():
            if 0 < item["update"] < 250:
                print(method, item["update"])
    else:
        write_tables(args.output)


if __name__ == "__main__":
    main()
