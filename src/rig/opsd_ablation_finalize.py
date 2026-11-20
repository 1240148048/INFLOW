from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

from .io import Example, read_jsonl, write_jsonl


KEY = "opsd_teacher_generation"


def attach_selection(
    example: Example,
    method: str,
    rank: int,
    criterion: str,
    score: float | None,
) -> Example:
    result = copy.deepcopy(example)
    result.metadata["opsd_selection"] = {
        "method": method,
        "rank": rank,
        "criterion": criterion,
        "score": score,
        "context_uids": result.demo_ids,
    }
    return result


def js_selection(
    examples: list[Example], method: str, fraction: float
) -> tuple[list[Example], str]:
    count = max(1, int(round(len(examples) * fraction)))
    criterion = "top Jensen-Shannon semantic belief shift"
    ordered = sorted(
        examples,
        key=lambda example: (
            -float(example.metadata[KEY]["distribution_shift_js"]),
            example.uid,
        ),
    )
    return [
        attach_selection(
            example,
            method,
            rank,
            criterion,
            float(example.metadata[KEY]["distribution_shift_js"]),
        )
        for rank, example in enumerate(ordered[:count], start=1)
    ], criterion


def random_selection(
    examples: list[Example], fraction: float, seed: int
) -> tuple[list[Example], str]:
    count = max(1, int(round(len(examples) * fraction)))
    criterion = f"uniform random selection with seed {seed}"
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    return [
        attach_selection(examples[index], "random20", rank, criterion, None)
        for rank, index in enumerate(indices[:count], start=1)
    ], criterion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-certainty", required=True)
    parser.add_argument("--no-similarity", required=True)
    parser.add_argument("--ours-teacher", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")

    sources = {
        "no_certainty": [
            Example.from_dict(row) for row in read_jsonl(args.no_certainty)
        ],
        "no_similarity": [
            Example.from_dict(row) for row in read_jsonl(args.no_similarity)
        ],
        "random20": [
            Example.from_dict(row) for row in read_jsonl(args.ours_teacher)
        ],
    }
    selected = {}
    selected["no_certainty"], criterion_no_certainty = js_selection(
        sources["no_certainty"], "no_certainty", args.fraction
    )
    selected["no_similarity"], criterion_no_similarity = js_selection(
        sources["no_similarity"], "no_similarity", args.fraction
    )
    selected["random20"], criterion_random = random_selection(
        sources["random20"], args.fraction, args.seed
    )
    criteria = {
        "no_certainty": criterion_no_certainty,
        "no_similarity": criterion_no_similarity,
        "random20": criterion_random,
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for method, examples in selected.items():
        destination = output / f"{method}.jsonl.gz"
        write_jsonl(destination, [example.to_dict() for example in examples])
        manifest[method] = {
            "path": str(destination.resolve()),
            "pool": len(sources[method]),
            "selected": len(examples),
            "criterion": criteria[method],
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
