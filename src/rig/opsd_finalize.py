from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io import Example, read_jsonl, write_jsonl


KEY = "opsd_teacher_generation"


def select(examples: list[Example], method: str, fraction: float):
    count = max(1, int(round(len(examples) * fraction)))
    if method == "full":
        score = None
        criterion = "all generated teacher data"
        ordered = examples
        count = len(examples)
    elif method == "ours":
        score = "distribution_shift_js"
        criterion = "top observed Jensen-Shannon shift after reliable decision-state retrieval"
    elif method == "mutual":
        score = "theoretical_retrieval_information"
        criterion = "top theoretical joint information from layerwise mutual retrieval"
    elif method == "consistency":
        score = "vote_agreement"
        criterion = "top five-sample answer consistency"
    elif method == "neuron":
        score = "mean_activation_count"
        criterion = "lowest mean activated-neuron count"
    elif method == "redundancy":
        score = None
        criterion = "greedy removal of the most redundant teacher representations"
        vectors = []
        for example in examples:
            vector = np.asarray(example.metadata[KEY]["best_last_layer_embedding"], dtype=np.float64)
            vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
        similarity = np.asarray(vectors) @ np.asarray(vectors).T
        np.fill_diagonal(similarity, -np.inf)
        retained = list(range(len(examples)))
        while len(retained) > count:
            local = similarity[np.ix_(retained, retained)]
            nearest = local.max(axis=1)
            remove_at = max(range(len(retained)), key=lambda i: (nearest[i], -retained[i]))
            retained.pop(remove_at)
        ordered = [examples[index] for index in retained]
    else:
        raise ValueError(f"Unsupported method: {method}")
    if score is not None:
        reverse = method != "neuron"
        ordered = sorted(
            examples,
            key=lambda example: (
                (-1.0 if reverse else 1.0) * float(example.metadata[KEY][score]),
                example.uid,
            ),
        )
    selected = []
    for rank, source in enumerate(ordered[:count], start=1):
        example = Example.from_dict(source.to_dict())
        example.metadata["opsd_selection"] = {
            "method": method,
            "rank": rank,
            "criterion": criterion,
            "score": None if score is None else float(example.metadata[KEY][score]),
            "context_uids": example.demo_ids,
        }
        selected.append(example)
    return selected, criterion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", required=True)
    parser.add_argument("--redundancy", required=True)
    parser.add_argument("--consistency", required=True)
    parser.add_argument("--neuron", required=True)
    parser.add_argument("--ours", required=True)
    parser.add_argument("--mutual", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for method in ("full", "redundancy", "consistency", "neuron", "ours", "mutual"):
        source = getattr(args, method)
        examples = [Example.from_dict(row) for row in read_jsonl(source)]
        selected, criterion = select(examples, method, args.fraction)
        destination = output / f"{method}.jsonl.gz"
        write_jsonl(destination, [example.to_dict() for example in selected])
        manifest[method] = {
            "path": str(destination.resolve()),
            "pool": len(examples),
            "selected": len(selected),
            "criterion": criterion,
        }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
