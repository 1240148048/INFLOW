from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from tqdm import tqdm

from .io import Example, read_jsonl, write_jsonl


KEY = "neuron_opsd"
DEFINITIONS = {
    "last_layer_certainty": "rcri_final_layer_v1",
    "last_layer_no_certainty": "positive_final_layer_similarity_squared_v1",
    "full": "final_layer_cosine_v1",
    "redundancy": "final_layer_cosine_v1",
    "consistency": "no_retrieval_v1",
    "neuron": "neuron_jaccard_v1",
    "ours": "rcri_final_layer_v1",
    "mutual": "layerwise_gaussian_information_v1",
    "no_certainty": "positive_final_layer_similarity_squared_v1",
    "no_similarity": "source_certainty_only_v1",
}


def info(example: Example) -> dict[str, Any]:
    return example.metadata[KEY]


def representation_tensor(examples: list[Example]) -> np.ndarray:
    matrices = np.asarray(
        [info(example)["samples"][0]["layer_embeddings"] for example in examples],
        dtype=np.float32,
    )
    if matrices.ndim != 3 or matrices.shape[1] == 0:
        raise ValueError(f"Expected [examples, layers, dimensions], got {matrices.shape}")
    return matrices / np.maximum(
        np.linalg.norm(matrices, axis=2, keepdims=True), 1e-12
    )


def ranked_contexts(score_matrix: np.ndarray, count: int, description: str):
    score_matrix = np.asarray(score_matrix, dtype=np.float64).copy()
    np.fill_diagonal(score_matrix, -np.inf)
    contexts: dict[int, list[int]] = {}
    scores: dict[int, list[float]] = {}
    for target in tqdm(
        range(len(score_matrix)), desc=description, unit="query", dynamic_ncols=True
    ):
        ranked = sorted(
            range(len(score_matrix)),
            key=lambda source: (score_matrix[target, source], source),
            reverse=True,
        )[:count]
        # Put the strongest demonstration closest to the target prompt.
        prompt_order = list(reversed(ranked))
        contexts[target] = prompt_order
        scores[target] = [float(score_matrix[target, source]) for source in prompt_order]
    return contexts, scores


def information_scores(similarity: np.ndarray, examples: list[Example]) -> np.ndarray:
    """Gaussian channel information with source reliability as signal fraction."""
    uncertainty = np.asarray(
        [
            info(example)["samples"][0]["normalized_choice_entropy"]
            for example in examples
        ],
        dtype=np.float32,
    )
    effective_correlation2 = np.clip(similarity, 0.0, 1.0) ** 2 * (
        1.0 - np.clip(uncertainty, 0.0, 1.0)[None, :]
    )
    return -0.5 * np.log1p(
        -np.clip(effective_correlation2, 0.0, 1.0 - 1e-8)
    )


def mutual_contexts(examples: list[Example], count: int):
    matrices = representation_tensor(examples)
    layers = matrices.shape[1]
    flattened = matrices.reshape(len(examples), -1) / np.sqrt(layers)
    similarity = np.clip(flattened @ flattened.T, 0.0, 1.0)
    return ranked_contexts(
        information_scores(similarity, examples), count, "retrieving mutual"
    )


def ours_contexts(examples: list[Example], count: int):
    final_vectors = representation_tensor(examples)[:, -1, :]
    similarity = np.clip(final_vectors @ final_vectors.T, -1.0, 1.0)
    return ranked_contexts(
        information_scores(similarity, examples), count, "retrieving ours"
    )


def similarity_only_contexts(examples: list[Example], count: int):
    final_vectors = representation_tensor(examples)[:, -1, :]
    similarity = np.clip(final_vectors @ final_vectors.T, 0.0, 1.0)
    return ranked_contexts(
        similarity**2, count, "retrieving without certainty"
    )


def certainty_only_contexts(examples: list[Example], count: int):
    uncertainty = np.asarray(
        [
            info(example)["samples"][0]["normalized_choice_entropy"]
            for example in examples
        ],
        dtype=np.float32,
    )
    certainty = 1.0 - np.clip(uncertainty, 0.0, 1.0)
    scores = np.broadcast_to(certainty[None, :], (len(examples), len(examples)))
    return ranked_contexts(
        scores, count, "retrieving without similarity"
    )


def cosine_contexts(examples: list[Example], count: int):
    final_vectors = representation_tensor(examples)[:, -1, :]
    similarity = np.clip(final_vectors @ final_vectors.T, -1.0, 1.0)
    return ranked_contexts(similarity, count, "retrieving embedding baselines")


def neuron_contexts(examples: list[Example], count: int):
    sets = [set(info(example).get("neuron_ids", [])) for example in examples]
    if not all(sets):
        raise ValueError(
            "NEURON retrieval requires non-empty candidate neuron sets; "
            "remove the stale shared-candidate cache and regenerate it."
        )
    # A Python pairwise set-intersection loop is prohibitively slow for 500
    # examples with thousands of active neurons each.  Sparse incidence-matrix
    # multiplication computes every intersection count in one native operation.
    identifiers = sorted(set().union(*sets))
    columns = {identifier: index for index, identifier in enumerate(identifiers)}
    rows = np.concatenate([
        np.full(len(values), index, dtype=np.int32)
        for index, values in enumerate(sets)
    ])
    cols = np.fromiter(
        (columns[value] for values in sets for value in values),
        dtype=np.int32,
        count=sum(map(len, sets)),
    )
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(sets), len(identifiers)),
    )
    intersections = (incidence @ incidence.T).toarray()
    sizes = np.asarray([len(values) for values in sets], dtype=np.float32)
    unions = sizes[:, None] + sizes[None, :] - intersections
    scores = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    return ranked_contexts(scores, count, "retrieving neuron contexts")


def write_retrieval(
    examples: list[Example],
    contexts: dict[int, list[int]],
    scores: dict[int, list[float]],
    method: str,
    output: Path,
) -> None:
    rows = []
    for index, source in enumerate(examples):
        # Layer matrices and neuron IDs are consumed while constructing the
        # retrieval indices. Teacher generation only needs the initial answer
        # distribution, so do not gzip the heavy retrieval features four times.
        source_initial = info(source)
        initial = {
            key: value
            for key, value in source_initial.items()
            if key not in ("samples", "neuron_ids")
        }
        initial["samples"] = [
            {
                key: value
                for key, value in sample.items()
                if key not in ("layer_embeddings", "neuron_ids")
            }
            for sample in source_initial.get("samples", [])
        ]
        example = copy.copy(source)
        example.metadata = dict(source.metadata)
        example.metadata[KEY] = initial
        example.demo_ids = [examples[item].uid for item in contexts[index]]
        example.metadata["opsd_retrieval"] = {
            "method": method,
            "definition": DEFINITIONS[method],
            "context_uids": example.demo_ids,
            "context_scores_in_prompt_order": scores[index],
        }
        rows.append(example.to_dict())
    write_jsonl(output, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-count", type=int, default=3)
    args = parser.parse_args()
    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cosine, cosine_scores = cosine_contexts(examples, args.context_count)
    neuron, neuron_scores = neuron_contexts(examples, args.context_count)
    ours, ours_scores = ours_contexts(examples, args.context_count)
    mutual, mutual_scores = mutual_contexts(examples, args.context_count)
    for method in ("full", "redundancy"):
        write_retrieval(
            examples, cosine, cosine_scores, method, output / f"{method}_all.jsonl.gz"
        )
    empty = {index: [] for index in range(len(examples))}
    write_retrieval(
        examples, empty, empty, "consistency", output / "consistency_all.jsonl.gz"
    )
    write_retrieval(
        examples, neuron, neuron_scores, "neuron", output / "neuron_all.jsonl.gz"
    )
    write_retrieval(
        examples, ours, ours_scores, "ours", output / "ours_all.jsonl.gz"
    )
    write_retrieval(
        examples, mutual, mutual_scores, "mutual", output / "mutual_all.jsonl.gz"
    )
    manifest = {
        "full": {"examples": len(examples), "context_count": args.context_count,
                 "retrieval": "final-layer cosine"},
        "redundancy": {"examples": len(examples), "context_count": args.context_count,
                       "retrieval": "final-layer cosine"},
        "consistency": {"examples": len(examples), "context_count": 0,
                        "retrieval": "none"},
        "neuron": {"examples": len(examples), "context_count": args.context_count,
                   "retrieval": "Jaccard overlap of activation sets"},
        "ours": {
            "examples": len(examples),
            "context_count": args.context_count,
            "retrieval": "reliability-calibrated Gaussian information from final-layer decision-state cosine",
        },
        "mutual": {
            "examples": len(examples),
            "context_count": args.context_count,
            "retrieval": "Gaussian information from mean layerwise cosine and source reliability",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
