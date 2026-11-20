#!/usr/bin/env python3
"""Summarize the direction of retrieval-induced semantic belief shifts.

The initial and retrieval-conditioned decisions are the argmax of the same
legal-answer distributions used to compute INFLOW's JS score:
  P0 = neuron_opsd.samples[0].choice_probabilities
  PS = opsd_teacher_generation.mean_choice_probabilities
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INITIAL_KEY = "neuron_opsd"
TEACHER_KEY = "opsd_teacher_generation"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def labels_for(row: dict[str, Any]) -> list[str]:
    labels = row.get("choice_labels") or []
    if labels:
        return [str(label) for label in labels]
    return [chr(ord("A") + i) for i in range(len(row.get("choices", [])))]


def argmax_label(probabilities: dict[str, Any], labels: list[str]) -> str:
    missing = [label for label in labels if label not in probabilities]
    if missing:
        raise KeyError(f"probability distribution misses labels {missing}")
    # Label order provides a deterministic tie break.
    return max(labels, key=lambda label: float(probabilities[label]))


def direction(initial_correct: bool, retrieved_correct: bool) -> str:
    if not initial_correct and retrieved_correct:
        return "Fix"
    if initial_correct and not retrieved_correct:
        return "Break"
    if initial_correct and retrieved_correct:
        return "Right-to-Right"
    return "Wrong-to-Wrong"


def resolve_inputs(candidates_dir: Path, teacher: Path | None):
    candidates = candidates_dir / "shared.jsonl.gz"
    run_root = candidates_dir.parent
    teacher = teacher or run_root / "teacher" / "ours_all.jsonl.gz"
    for path in (candidates, teacher):
        if not path.exists():
            raise FileNotFoundError(path)
    return candidates, teacher


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["direction"] for row in rows)
    total = len(rows)
    return {
        **{f"{name}_pct": 100.0 * counts[name] / total if total else 0.0 for name in (
            "Fix", "Break", "Wrong-to-Wrong", "Right-to-Right"
        )},
        "Fix_minus_Break_pp": (
            100.0 * (counts["Fix"] - counts["Break"]) / total if total else 0.0
        ),
        "initial_accuracy_pct": (
            100.0 * sum(row["initial_correct"] for row in rows) / total
            if total else 0.0
        ),
        "teacher_accuracy_pct": (
            100.0 * sum(row["retrieved_correct"] for row in rows) / total
            if total else 0.0
        ),
        "mean_js": sum(row["belief_shift_js"] for row in rows) / total if total else 0.0,
    }


def split_descending_js_deciles(
    rows: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Return exclusive 10% rank intervals after sorting JS high to low."""
    ordered = sorted(
        rows,
        key=lambda row: (-row["belief_shift_js"], row["uid"]),
    )
    n = len(ordered)
    groups = []
    for decile in range(10):
        start = decile * n // 10
        end = (decile + 1) * n // 10
        groups.append((f"{10 * decile}-{10 * (decile + 1)}%", ordered[start:end]))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts/opsd_selection_t2000_u500/candidates",
    )
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    candidates_path, teacher_path = resolve_inputs(args.candidates_dir, args.teacher)
    output_dir = args.output_dir or args.candidates_dir.parent / "belief_shift_direction"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = {row["uid"]: row for row in read_jsonl(candidates_path)}
    records: list[dict[str, Any]] = []

    for teacher_row in read_jsonl(teacher_path):
        uid = teacher_row["uid"]
        candidate = candidates.get(uid)
        if candidate is None:
            raise KeyError(f"{uid} occurs in teacher data but not in {candidates_path}")
        labels = labels_for(candidate)
        gold = candidate.get("answer")
        if not labels or gold not in labels:
            raise ValueError(f"{uid} has invalid labels or gold answer: {gold!r}")

        initial_meta = candidate.get("metadata", {}).get(INITIAL_KEY, {})
        samples = initial_meta.get("samples") or []
        if not samples:
            raise KeyError(f"{uid} has no {INITIAL_KEY}.samples[0]")
        initial_probs = samples[0].get("choice_probabilities")
        teacher_meta = teacher_row.get("metadata", {}).get(TEACHER_KEY, {})
        retrieved_probs = teacher_meta.get("mean_choice_probabilities")
        if not isinstance(initial_probs, dict) or not isinstance(retrieved_probs, dict):
            raise KeyError(f"{uid} lacks one of the two semantic belief distributions")

        initial_answer = argmax_label(initial_probs, labels)
        retrieved_answer = argmax_label(retrieved_probs, labels)
        initial_correct = initial_answer == gold
        retrieved_correct = retrieved_answer == gold
        records.append({
            "uid": uid,
            "dataset": candidate.get("dataset"),
            "domain": candidate.get("domain"),
            "gold_answer": gold,
            "initial_answer": initial_answer,
            "retrieved_answer": retrieved_answer,
            "initial_correct": initial_correct,
            "retrieved_correct": retrieved_correct,
            "direction": direction(initial_correct, retrieved_correct),
            "belief_shift_js": float(teacher_meta["distribution_shift_js"]),
        })

    if not records:
        raise ValueError(f"No records were read from {teacher_path}")

    summary = [
        {"js_rank_interval": name, **summarize(group)}
        for name, group in split_descending_js_deciles(records)
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print(
        "JS rank\tFix (%)\tBreak (%)\tWrong->Wrong (%)\tRight->Right (%)\t"
        "Fix-Break (pp)\tInitial acc. (%)\tTeacher acc. (%)\tMean JS"
    )
    for values in summary:
        print(
            f"{values['js_rank_interval']}\t{values['Fix_pct']:.2f}%\t"
            f"{values['Break_pct']:.2f}%\t"
            f"{values['Wrong-to-Wrong_pct']:.2f}%\t"
            f"{values['Right-to-Right_pct']:.2f}%\t"
            f"{values['Fix_minus_Break_pp']:.2f} pp\t"
            f"{values['initial_accuracy_pct']:.2f}%\t"
            f"{values['teacher_accuracy_pct']:.2f}%\t"
            f"{values['mean_js']:.6f}"
        )
    print(f"Wrote {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
