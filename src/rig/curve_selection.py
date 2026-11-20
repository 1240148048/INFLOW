from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import Example, read_jsonl, write_jsonl
from .selection import metadata
from .qa_information import has_features, question_answer_information_order


METHODS = (
    "random",
    "difficulty",
    "uncertainty",
    "redundancy",
    "activation",
    "ours",
    "ours-inverse",
)


def parse_fractions(text: str) -> list[float]:
    values = sorted({float(item) for item in text.split(",") if item.strip()})
    if not values or any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("Fractions must be unique values strictly between 0 and 1.")
    return values


def difficulty_agreement(example: Example) -> float:
    info = metadata(example)
    summaries = info["sample_summaries"]
    standard_index = int(info["standard_sample_index"])
    target = str(info["standard_answer"])
    others = [
        sample.get("answer")
        for index, sample in enumerate(summaries)
        if index != standard_index
    ]
    if not others:
        raise ValueError("Difficulty selection requires at least two rollouts.")
    return sum(answer == target for answer in others) / len(others)


def redundancy_elimination_order(embeddings: np.ndarray) -> list[int]:
    matrix = np.asarray(embeddings, dtype=np.float64)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    similarity = matrix @ matrix.T
    active = list(range(len(matrix)))
    removed: list[int] = []
    while len(active) > 1:
        submatrix = similarity[np.ix_(active, active)].copy()
        np.fill_diagonal(submatrix, -np.inf)
        nearest = submatrix.max(axis=1)
        candidate = int(np.argmax(nearest))
        removed.append(active.pop(candidate))
    return removed + active


def build_orders(
    examples: list[Example],
    seed: int,
    rank: int,
    ridge: float,
    noise: float,
    balance_domains: bool,
    methods: list[str] | None = None,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    del rank, balance_domains
    requested = set(METHODS if methods is None else methods)
    count = len(examples)
    orders: dict[str, list[int]] = {}
    diagnostics: dict[str, Any] = {"methods": sorted(requested)}
    if "random" in requested:
        orders["random"] = list(range(count))
        random.Random(seed).shuffle(orders["random"])
    if "difficulty" in requested:
        agreements = np.asarray([difficulty_agreement(item) for item in examples])
        orders["difficulty"] = sorted(
            range(count), key=lambda index: (abs(agreements[index] - 0.5), index)
        )
        diagnostics["difficulty_agreement_histogram"] = dict(
            sorted(Counter(float(value) for value in agreements).items())
        )
    if "uncertainty" in requested:
        entropy = np.asarray(
            [float(metadata(item)["standard_predictive_entropy"]) for item in examples]
        )
        orders["uncertainty"] = sorted(
            range(count), key=lambda index: (entropy[index], index)
        )
        diagnostics["mean_standard_entropy"] = float(entropy.mean())
    if "activation" in requested:
        activations = np.asarray(
            [float(metadata(item)["standard_activation_count"]) for item in examples]
        )
        orders["activation"] = sorted(
            range(count), key=lambda index: (activations[index], index)
        )
        diagnostics["mean_activation_count"] = float(activations.mean())
    if "redundancy" in requested:
        embeddings = np.asarray(
            [metadata(item)["standard_embedding"] for item in examples],
            dtype=np.float32,
        )
        orders["redundancy"] = redundancy_elimination_order(embeddings)
        diagnostics["redundancy_rule"] = (
            "remove_largest_nearest_neighbor_cosine"
        )
    if requested & {"ours", "ours-inverse"}:
        if not all(has_features(example) for example in examples):
            raise ValueError(
                "Question-answer information features are required for "
                "ours/ours-inverse. Run `python -m rig.qa_information extract` "
                "on the candidate file first."
            )
        if "ours" in requested:
            orders["ours"], diagnostics["ours"] = (
                question_answer_information_order(
                    examples, ridge=ridge, maximize=True
                )
            )
        if "ours-inverse" in requested:
            orders["ours-inverse"], diagnostics["ours_inverse"] = (
                question_answer_information_order(
                    examples, ridge=ridge, maximize=False
                )
            )
    return orders, diagnostics


def selected_indices(method: str, order: list[int], count: int) -> list[int]:
    if method == "redundancy":
        return sorted(order[len(order) - count :])
    return order[:count]


def create_subsets(
    examples: list[Example],
    orders: dict[str, list[int]],
    fractions: list[float],
    output_dir: Path,
    methods: list[str] | None = None,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for method in list(METHODS) if methods is None else methods:
        for fraction in fractions:
            count = max(1, int(round(len(examples) * fraction)))
            indices = selected_indices(method, orders[method], count)
            output = output_dir / method / f"p{int(round(100 * fraction)):02d}.jsonl.gz"
            rows = []
            domains: defaultdict[str, int] = defaultdict(int)
            for selection_order, index in enumerate(indices, start=1):
                row = examples[index].to_dict()
                rig = row.setdefault("metadata", {}).setdefault("rig", {})
                for heavy_key in (
                    "standard_embedding",
                    "projected_embedding",
                    "need_operator",
                    "sample_summaries",
                ):
                    rig.pop(heavy_key, None)
                qair = row.setdefault("metadata", {}).setdefault("qair", {})
                qair.pop("gradient_feature", None)
                qair.pop("sampled_gradient_features", None)
                qair.pop("question_change_feature", None)
                row.setdefault("metadata", {}).setdefault("curve", {}).update(
                    {
                        "method": method,
                        "fraction": fraction,
                        "selection_order": selection_order,
                    }
                )
                rows.append(row)
                domains[examples[index].domain] += 1
            write_jsonl(output, rows)
            manifest.append(
                {
                    "method": method,
                    "fraction": fraction,
                    "count": count,
                    "path": str(output.resolve()),
                    "domain_counts": dict(sorted(domains.items())),
                }
            )
    full = output_dir / "full" / "p100.jsonl.gz"
    full_rows = []
    for example in examples:
        row = example.to_dict()
        rig = row.setdefault("metadata", {}).setdefault("rig", {})
        for heavy_key in (
            "standard_embedding",
            "projected_embedding",
            "need_operator",
            "sample_summaries",
        ):
            rig.pop(heavy_key, None)
        qair = row.setdefault("metadata", {}).setdefault("qair", {})
        qair.pop("gradient_feature", None)
        qair.pop("sampled_gradient_features", None)
        qair.pop("question_change_feature", None)
        full_rows.append(row)
    write_jsonl(full, full_rows)
    manifest.append(
        {
            "method": "full",
            "fraction": 1.0,
            "count": len(examples),
            "path": str(full.resolve()),
            "domain_counts": dict(sorted(Counter(x.domain for x in examples).items())),
        }
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build nested selection curves")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fractions", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--balance-domains", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    if not examples:
        raise ValueError("The candidate file is empty.")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    orders, diagnostics = build_orders(
        examples,
        args.seed,
        args.rank,
        args.ridge,
        args.noise,
        args.balance_domains,
        methods,
    )
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"subsets": [], "diagnostics": {}}
    )
    generated = create_subsets(
        examples, orders, parse_fractions(args.fractions), output_dir, methods
    )
    by_key = {
        (row["method"], round(float(row["fraction"]), 12)): row
        for row in previous.get("subsets", [])
    }
    for row in generated:
        by_key[(row["method"], round(float(row["fraction"]), 12))] = row
    previous_diagnostics = previous.get("diagnostics", {})
    previous_methods = set(previous_diagnostics.get("methods", []))
    previous_diagnostics.update(diagnostics)
    previous_diagnostics["methods"] = sorted(previous_methods | set(methods))
    manifest = sorted(
        by_key.values(), key=lambda row: (row["method"], row["fraction"])
    )
    manifest_path.write_text(
        json.dumps(
            {"subsets": manifest, "diagnostics": previous_diagnostics}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"Updated {len(generated)} subsets; manifest contains {len(manifest)}.")


if __name__ == "__main__":
    main()
