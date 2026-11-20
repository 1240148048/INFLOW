from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .io import Example, read_jsonl, write_jsonl


METHODS = ("ours", "activation", "uncertainty", "difficulty")


def rig_info(example: Example) -> dict[str, Any]:
    return example.metadata.get("rig", {})


def vote_answer(example: Example) -> str:
    summaries = rig_info(example).get("sample_summaries", [])
    labels = example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]
    counts = Counter(
        str(sample.get("answer"))
        for sample in summaries
        if sample.get("answer") in labels
    )
    if not counts:
        fallback = rig_info(example).get("standard_answer") or example.rollout
        if fallback not in labels:
            raise ValueError(f"{example.uid} has no valid sampled answer.")
        return str(fallback)
    largest = max(counts.values())
    return next(label for label in labels if counts.get(label, 0) == largest)


def vote_disagreement(example: Example) -> float:
    summaries = rig_info(example).get("sample_summaries", [])
    answers = [sample.get("answer") for sample in summaries]
    if len(answers) < 2:
        raise ValueError(
            f"{example.uid} needs at least two rollouts for vote disagreement."
        )
    largest = max(Counter(answers).values())
    return 1.0 - largest / len(answers)


def normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite two-dimensional feature matrix.")
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def cosine_contexts(
    embeddings: np.ndarray,
    queries: list[int],
    pool: list[int],
    count: int,
) -> dict[int, list[int]]:
    matrix = normalized_rows(embeddings)
    similarity = matrix @ matrix.T
    pool_set = set(pool)
    output: dict[int, list[int]] = {}
    for query in queries:
        candidates = [index for index in pool_set if index != query]
        ranked = sorted(candidates, key=lambda index: (-similarity[query, index], index))
        # Put the strongest demonstration last so left truncation preserves it.
        output[query] = list(reversed(ranked[:count]))
    return output


def activation_contexts(
    neuron_sets: list[set[int]], queries: list[int], count: int
) -> dict[int, list[int]]:
    output: dict[int, list[int]] = {}
    for query in queries:
        source = neuron_sets[query]
        scored = []
        for index, target in enumerate(neuron_sets):
            if index == query:
                continue
            union = len(source | target)
            similarity = len(source & target) / union if union else 0.0
            scored.append((similarity, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        output[query] = list(reversed([index for _score, index in scored[:count]]))
    return output


def information_graph(
    examples: list[Example], ridge: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sampled = np.asarray(
        [example.metadata["qair"]["sampled_gradient_features"] for example in examples],
        dtype=np.float64,
    )
    questions = np.asarray(
        [example.metadata["qair"]["question_change_feature"] for example in examples],
        dtype=np.float64,
    )
    if sampled.ndim != 3 or not np.isfinite(sampled).all():
        raise ValueError(
            "Ours requires sampled qair features. Run `python -m rig.qa_information "
            "extract` first."
        )
    sample_count = sampled.shape[1]
    centered = sampled - sampled.mean(axis=1, keepdims=True)
    active_rank = max(min(sample_count - 1, sampled.shape[2]), 1)
    entropy = np.empty(len(examples), dtype=np.float64)
    for index in range(len(examples)):
        gram = centered[index] @ centered[index].T / max(sample_count - 1, 1)
        sign, logdet = np.linalg.slogdet(np.eye(sample_count) + gram)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Invalid rollout covariance in qair features.")
        entropy[index] = 0.5 * logdet / active_rank
    # Differential entropy of the rollout-direction covariance is the amount
    # of unresolved representation information for this question.
    need = entropy
    teaching_noise = np.maximum(np.expm1(entropy), 1e-10)

    centered_questions = questions - questions.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(centered_questions), axis=0))
    keep = scale > max(float(scale.max()) * 1e-8, 1e-12)
    standardized = (
        centered_questions[:, keep] / scale[keep]
        if np.any(keep)
        else centered_questions
    )
    norms = np.einsum("ij,ij->i", standardized, standardized)
    distances = np.maximum(
        norms[:, None] + norms[None, :] - 2.0 * standardized @ standardized.T,
        0.0,
    )
    upper = distances[np.triu_indices(len(examples), k=1)]
    positive = upper[upper > 1e-12]
    bandwidth = float(np.median(positive)) if positive.size else 1.0
    transfer = np.exp(-distances / max(2.0 * bandwidth, 1e-12))
    transfer += np.eye(len(examples)) * max(ridge, 1e-8)

    target_variance = np.maximum(np.diag(transfer), 1e-12)
    source_variance = target_variance + teaching_noise
    rho2 = np.square(transfer) / np.maximum(
        target_variance[:, None] * source_variance[None, :], 1e-12
    )
    rho2 = np.clip(rho2, 0.0, 1.0 - 1e-12)
    pair_information = -0.5 * np.log1p(-rho2)
    np.fill_diagonal(pair_information, -np.inf)
    return need, pair_information, {
        "mean_information_need_nats": float(need.mean()),
        "mean_teaching_noise": float(teaching_noise.mean()),
        "rbf_median_squared_distance": bandwidth,
    }


def build_method(
    examples: list[Example],
    method: str,
    fraction: float,
    context_count: int,
    activation_context_count: int,
    ridge: float,
) -> tuple[list[int], dict[int, list[int]], dict[str, Any]]:
    total = len(examples)
    selected_count = max(1, int(round(total * fraction)))
    embeddings = np.asarray(
        [rig_info(example)["standard_embedding"] for example in examples],
        dtype=np.float64,
    )
    diagnostics: dict[str, Any] = {}

    if method == "ours":
        need, pair_information, diagnostics = information_graph(examples, ridge)
        selected = sorted(range(total), key=lambda index: (-need[index], index))[
            :selected_count
        ]
        contexts = {}
        for query in selected:
            ranked = sorted(
                (index for index in range(total) if index != query),
                key=lambda index: (-pair_information[query, index], index),
            )[:context_count]
            contexts[query] = list(reversed(ranked))
        diagnostics["selection_rule"] = "largest_rollout_representation_information_need"
        diagnostics["retrieval_rule"] = "largest_pairwise_gaussian_information"
    elif method == "activation":
        counts = np.asarray(
            [float(rig_info(example)["standard_activation_count"]) for example in examples]
        )
        selected = sorted(range(total), key=lambda index: (counts[index], index))[
            :selected_count
        ]
        neuron_sets = []
        for example in examples:
            identifiers = rig_info(example).get("activation_neuron_ids")
            if identifiers is None:
                raise ValueError(
                    "Activation retrieval requires `activation_neuron_ids`. "
                    "Regenerate candidates with the current rig.selection."
                )
            neuron_sets.append(set(int(value) for value in identifiers))
        contexts = activation_contexts(
            neuron_sets, selected, activation_context_count
        )
        diagnostics = {
            "selection_rule": "bottom_activation_union_size",
            "retrieval_rule": "largest_activation_set_jaccard",
        }
    elif method == "uncertainty":
        entropy = np.asarray(
            [float(rig_info(example)["standard_predictive_entropy"]) for example in examples]
        )
        selected = sorted(range(total), key=lambda index: (entropy[index], index))[
            :selected_count
        ]
        contexts = cosine_contexts(
            embeddings, selected, list(range(total)), context_count
        )
        diagnostics = {
            "selection_rule": "lowest_answer_sequence_predictive_entropy",
            "retrieval_rule": "largest_answer_representation_cosine",
        }
    elif method == "difficulty":
        disagreement = np.asarray([vote_disagreement(example) for example in examples])
        ranked = sorted(range(total), key=lambda index: (disagreement[index], index))
        start = int(math.floor(0.4 * total))
        selected = ranked[start : start + selected_count]
        reliable_pool = ranked[: max(1, int(math.floor(0.4 * total)))]
        contexts = cosine_contexts(
            embeddings, selected, reliable_pool, context_count
        )
        diagnostics = {
            "selection_rule": "vote_disagreement_percentile_40_to_60",
            "retrieval_rule": "largest_answer_representation_cosine_within_bottom_40pct_disagreement",
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    return selected, contexts, diagnostics


def write_method_subset(
    examples: list[Example],
    method: str,
    selected: list[int],
    contexts: dict[int, list[int]],
    output: Path,
) -> dict[str, Any]:
    rows = []
    for rank, index in enumerate(selected, start=1):
        example = Example.from_dict(examples[index].to_dict())
        example.rollout = vote_answer(example)
        example.demo_ids = [examples[item].uid for item in contexts[index]]
        example.metadata.setdefault("context_selection", {}).update(
            {
                "method": method,
                "selection_rank": rank,
                "context_uids": list(example.demo_ids),
            }
        )
        rows.append(example.to_dict())
    write_jsonl(output, rows)
    return {
        "method": method,
        "count": len(rows),
        "path": str(output.resolve()),
        "mean_context_count": float(
            np.mean([len(contexts[index]) for index in selected])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build four retrieval-conditioned self-distillation subsets."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--activation-context-count", type=int, default=33)
    parser.add_argument("--ridge", type=float, default=0.1)
    args = parser.parse_args()
    if not 0.0 < args.fraction < 1.0:
        raise ValueError("--fraction must lie strictly between zero and one.")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown or not methods:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    if not examples:
        raise ValueError("Candidate file is empty.")
    # All demonstrations use the five-sample majority label.
    for example in examples:
        example.rollout = vote_answer(example)

    output_dir = Path(args.output_dir)
    manifest = []
    diagnostics = {}
    for method in methods:
        selected, contexts, details = build_method(
            examples,
            method,
            args.fraction,
            args.context_count,
            args.activation_context_count,
            args.ridge,
        )
        output = output_dir / method / "p20.jsonl.gz"
        manifest.append(
            write_method_subset(examples, method, selected, contexts, output)
        )
        diagnostics[method] = details
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_manifest.json").write_text(
        json.dumps(
            {
                "candidate_path": str(Path(args.input).resolve()),
                "fraction": args.fraction,
                "subsets": manifest,
                "diagnostics": diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} context-conditioned subsets to {output_dir}")


if __name__ == "__main__":
    main()
