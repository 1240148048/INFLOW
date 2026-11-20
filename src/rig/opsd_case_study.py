from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import Example, read_jsonl, write_jsonl


INITIAL_KEY = "neuron_opsd"
RETRIEVAL_KEY = "opsd_retrieval"
TEACHER_KEY = "opsd_teacher_generation"
SELECTION_KEY = "opsd_selection"


def labels_for(example: Example) -> list[str]:
    return example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= max(float(left.sum()), 1e-12)
    right /= max(float(right.sum()), 1e-12)
    middle = 0.5 * (left + right)
    left_kl = np.sum(
        left * np.log(np.maximum(left, 1e-12) / np.maximum(middle, 1e-12))
    )
    right_kl = np.sum(
        right * np.log(np.maximum(right, 1e-12) / np.maximum(middle, 1e-12))
    )
    return float(0.5 * (left_kl + right_kl))


def _sample_record(sample: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "sample_index": index,
        "response": sample.get("text"),
        "parsed_answer": sample.get("answer"),
        "choice_probabilities": sample.get("choice_probabilities"),
        "choice_entropy": sample.get("choice_entropy"),
        "normalized_choice_entropy": sample.get("normalized_choice_entropy"),
        "mean_token_entropy": sample.get("mean_token_entropy"),
        "answer_token_entropy": sample.get("answer_token_entropy"),
        "token_count": sample.get("token_count"),
    }


def build_case_record(
    example: Example,
    candidates: dict[str, Example],
    selected: dict[str, Example],
) -> dict[str, Any]:
    labels = labels_for(example)
    initial_samples = example.metadata[INITIAL_KEY]["samples"]
    if not initial_samples:
        raise ValueError(f"{example.uid} has no initial samples")
    initial = initial_samples[0]

    retrieval = example.metadata[RETRIEVAL_KEY]
    context_uids = retrieval["context_uids"]
    context_scores = retrieval["context_scores_in_prompt_order"]
    if len(context_uids) != len(context_scores):
        raise ValueError(f"{example.uid} has mismatched context IDs and scores")

    contexts = []
    context_count = len(context_uids)
    for prompt_position, (uid, score) in enumerate(
        zip(context_uids, context_scores), start=1
    ):
        source = candidates[uid]
        source_info = source.metadata[INITIAL_KEY]
        source_sample = source_info["samples"][source_info["demonstration_index"]]
        source_uncertainty = float(source_sample["normalized_choice_entropy"])
        contexts.append(
            {
                "prompt_position": prompt_position,
                "retrieval_rank": context_count - prompt_position + 1,
                "uid": uid,
                "score": float(score),
                "question": source.question,
                "choices": dict(zip(labels_for(source), source.choices)),
                "gold_answer": source.answer,
                "demonstration_response": source.rollout,
                "demonstration_answer": source_sample.get("answer"),
                "source_uncertainty": source_uncertainty,
                "source_certainty": 1.0 - source_uncertainty,
            }
        )

    teacher = example.metadata[TEACHER_KEY]
    teacher_samples = teacher["samples"]
    vote_counts = Counter(
        sample.get("answer") if sample.get("answer") in labels else "invalid"
        for sample in teacher_samples
    )
    initial_probability = np.asarray(
        [initial["choice_probabilities"][label] for label in labels],
        dtype=np.float64,
    )
    teacher_probability = np.asarray(
        [teacher["mean_choice_probabilities"][label] for label in labels],
        dtype=np.float64,
    )
    recomputed_js = jensen_shannon(initial_probability, teacher_probability)
    stored_js = float(teacher["distribution_shift_js"])
    if not np.isclose(recomputed_js, stored_js, rtol=1e-7, atol=1e-10):
        raise ValueError(
            f"{example.uid} has inconsistent JS values: {stored_js} vs {recomputed_js}"
        )

    selected_example = selected.get(example.uid)
    selection = (
        selected_example.metadata.get(SELECTION_KEY, {}) if selected_example else {}
    )
    best_index = int(teacher["best_sample_index"])
    return {
        "uid": example.uid,
        "dataset": example.dataset,
        "domain": example.domain,
        "question": example.question,
        "choices": dict(zip(labels, example.choices)),
        "gold_answer": example.answer,
        "initial": {
            "sample_index_used_for_js": 0,
            **_sample_record(initial, 0),
            "initial_record_response": example.rollout,
        },
        "retrieval": {
            "method": retrieval["method"],
            "definition": retrieval["definition"],
            "contexts_in_prompt_order": contexts,
            "note": "The strongest retrieved context is placed closest to the target.",
        },
        "teacher_generation": {
            "samples": [
                _sample_record(sample, index)
                for index, sample in enumerate(teacher_samples)
            ],
            "hard_vote_counts": dict(vote_counts),
            "modal_answer": teacher["modal_answer"],
            "vote_agreement": float(teacher["vote_agreement"]),
            "mean_choice_probabilities": teacher["mean_choice_probabilities"],
            "best_sample_index": best_index,
            "chosen_teacher_response": teacher_samples[best_index]["text"],
            "chosen_teacher_answer": teacher["best_answer"],
        },
        "data_selection": {
            "distribution_shift_js": stored_js,
            "selected": selected_example is not None,
            "selection_rank": selection.get("rank"),
            "selection_score": selection.get("score"),
            "selection_criterion": selection.get("criterion"),
        },
    }


def write_index(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "uid",
        "dataset",
        "domain",
        "gold_answer",
        "initial_answer",
        "retrieved_uids",
        "retrieval_scores",
        "teacher_answers",
        "vote_counts",
        "modal_answer",
        "vote_agreement",
        "distribution_shift_js",
        "selected",
        "selection_rank",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            contexts = record["retrieval"]["contexts_in_prompt_order"]
            teacher = record["teacher_generation"]
            selection = record["data_selection"]
            writer.writerow(
                {
                    "uid": record["uid"],
                    "dataset": record["dataset"],
                    "domain": record["domain"],
                    "gold_answer": record["gold_answer"],
                    "initial_answer": record["initial"]["parsed_answer"],
                    "retrieved_uids": json.dumps(
                        [item["uid"] for item in contexts], ensure_ascii=False
                    ),
                    "retrieval_scores": json.dumps(
                        [item["score"] for item in contexts]
                    ),
                    "teacher_answers": json.dumps(
                        [item["parsed_answer"] for item in teacher["samples"]]
                    ),
                    "vote_counts": json.dumps(teacher["hard_vote_counts"]),
                    "modal_answer": teacher["modal_answer"],
                    "vote_agreement": teacher["vote_agreement"],
                    "distribution_shift_js": selection["distribution_shift_js"],
                    "selected": selection["selected"],
                    "selection_rank": selection["selection_rank"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index-csv", required=True)
    args = parser.parse_args()

    candidates = {
        example.uid: example
        for example in (Example.from_dict(row) for row in read_jsonl(args.candidates))
    }
    selected = {
        example.uid: example
        for example in (Example.from_dict(row) for row in read_jsonl(args.selected))
    }
    examples = [Example.from_dict(row) for row in read_jsonl(args.teacher)]
    records = [build_case_record(example, candidates, selected) for example in examples]
    write_jsonl(args.output, records)
    write_index(Path(args.index_csv), records)
    print(f"Saved {len(records)} complete case-study records to {args.output}")
    print(f"Saved searchable case-study index to {args.index_csv}")


if __name__ == "__main__":
    main()
