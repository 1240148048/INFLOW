from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from .data import dump


def paired_interval(left, right, seed=42, repetitions=2000, groups=None):
    ids = sorted(left)
    if not ids or set(ids) != set(right):
        raise ValueError("Paired intervals require identical question IDs")
    differences = [100 * (left[uid] - right[uid]) for uid in ids]
    rng = random.Random(seed)
    clusters = {}
    for uid, difference in zip(ids, differences):
        clusters.setdefault(groups.get(uid, uid) if groups else uid, []).append(difference)
    values = list(clusters.values())
    samples = []
    for _ in range(repetitions):
        chosen = rng.choices(values, k=len(values))
        samples.append(sum(sum(group) for group in chosen) / sum(len(group) for group in chosen))
    samples.sort()
    return {"difference_pp": sum(differences) / len(ids), "ci95_low": samples[int(0.025 * repetitions)], "ci95_high": samples[min(repetitions - 1, int(0.975 * repetitions))], "resampling_unit": "conversation for CoQA, question otherwise; candidates stay grouped; not training-seed uncertainty"}


def report(output):
    output = Path(output)
    files = sorted((output / "metrics").glob("*.json"))
    if not files:
        raise ValueError("No scored results to summarize")
    results = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    task = results[0]["task"]
    if any(row["task"] != task for row in results):
        raise ValueError("One report must contain one task")
    fields = ["method", "task", "questions", "accuracy", "surface_exact_match", "strict_exact_match",
              "token_f1", "intent_accuracy", "hierarchical_slot_f1", "valid_rate",
              "overall_contribution_pp", "rollout_accuracy_sd"]
    fields += ["training_seconds", "evaluation_seconds"]
    rows = []
    for item in results:
        training_path = output / "checkpoints" / item["method"] / "training_metrics.json"
        training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.exists() else {}
        for name, values in item["by_task"].items():
            row = {key: values.get(key) for key in fields}
            row.update({"method": item["method"], "task": name, "accuracy": values.get("exact_match", values.get("pass_at_1")), "valid_rate": values.get("valid_parse_rate", values.get("syntax_valid_rate", values.get("valid_answer_rate")))})
            row["overall_contribution_pp"] = item.get("overall_task_contributions_pp", {}).get(name)
            rows.append(row)
        if item["task"] == "mixed":
            rows.append({"method": item["method"], "task": "Overall", "questions": item["questions"], "accuracy": item["overall_accuracy"], "rollout_accuracy_sd": item.get("rollout_accuracy_sd"), "training_seconds": training.get("seconds", 0 if item["method"] == "original" else None), "evaluation_seconds": item["generation_seconds"]})
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    (output / "results.csv").write_text(stream.getvalue(), encoding="utf-8")
    rendered = lambda value: "--" if value is None else str(value) if isinstance(value, int) else f"{value:.3f}" if isinstance(value, float) else str(value)
    lines = ["# Open-task experiment results", "", "Accuracy metrics are percentages. Times are seconds. Training and evaluation are separate; shared preprocessing is reported in preparation_metrics.json and must be included for end-to-end cost claims.", "", "| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines += ["| " + " | ".join(rendered(row.get(field)) for field in fields) + " |" for row in rows]
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    escaped = lambda value: rendered(value).replace("_", r"\_")
    tex_fields = ["method", "task", "questions", "accuracy", "strict_exact_match", "token_f1", "valid_rate"]
    tex = [r"% TOPv2 main EM compares intent and nested slot labels/values; strict EM compares the full tree.", r"\begin{tabular}{llrrrrr}", r"\hline", r"Method & Task & N & EM/pass@1 & Strict EM & Token F1 & Valid (\%) \\", r"\hline"]
    tex += [" & ".join(escaped(row.get(field)) for field in tex_fields) + r" \\" for row in rows]
    tex += [r"\hline", r"\end{tabular}"]
    (output / "results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    indexed = {row["method"]: row for row in results}
    comparisons = {}
    if "js" in indexed:
        for method in indexed.keys() - {"js"}:
            comparisons[f"js_minus_{method}"] = paired_interval(indexed["js"]["per_question"], indexed[method]["per_question"], groups=indexed["js"].get("bootstrap_groups"))
    dump(output / "paired_comparisons.json", comparisons)
    print(f"Saved results.md, results.csv, results.tex and paired_comparisons.json in {output}")
