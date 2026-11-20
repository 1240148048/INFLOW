from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import Example, read_jsonl


KEY = "neuron_opsd"
METHOD_FILES = {
    "ours": "ours_all.jsonl.gz",
    "mutual": "mutual_all.jsonl.gz",
    "neuron": "neuron_all.jsonl.gz",
    "cosine": "full_all.jsonl.gz",
}
LABELS = {
    "ours": "Ours",
    "mutual": "Mutual information",
    "neuron": "Neuron overlap",
    "cosine": "Cosine similarity",
}
METHOD_COLORS = {
    "ours": "#d43d2f",
    "mutual": "#7a6f9b",
    "neuron": "#4c78a8",
    "cosine": "#555555",
}
GROUP_COLORS = {
    "Mathematics": "#e69f00",
    "Natural sciences": "#2474b5",
    "Humanities and social sciences": "#1b9e77",
    "Other": "#999999",
}


def broad_domain(domain: str) -> str:
    if domain == "Math":
        return "Mathematics"
    if domain in {"Biology", "Chemistry", "Material", "Physics"}:
        return "Natural sciences"
    if domain in {"Psychology", "Global Facts", "History", "Economics", "Law"}:
        return "Humanities and social sciences"
    return "Other"


def final_vectors(examples: list[Example]) -> np.ndarray:
    vectors = np.asarray(
        [example.metadata[KEY]["samples"][0]["layer_embeddings"][-1]
         for example in examples],
        dtype=np.float64,
    )
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def pca(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    _left, singular, right = np.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ right[:2].T
    variance = singular**2
    explained = variance[:2] / max(float(variance.sum()), 1e-12)
    return coordinates, explained


def round_robin_targets(
    candidates: list[Example], target_uids: set[str], maximum: int
) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for example in candidates:
        if example.uid in target_uids:
            groups[broad_domain(example.domain)].append(example.uid)
    for values in groups.values():
        values.sort()
    selected: list[str] = []
    ordered_groups = sorted(groups)
    cursor = 0
    while len(selected) < maximum and any(groups.values()):
        group = ordered_groups[cursor % len(ordered_groups)]
        if groups[group]:
            selected.append(groups[group].pop(0))
        cursor += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA representation-retrieval graph for the Qwen k=3 experiment."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--retrieval-dir", required=True)
    parser.add_argument("--subsets-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-count", type=int, default=3)
    parser.add_argument("--max-targets", type=int, default=60)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    candidates = [Example.from_dict(row) for row in read_jsonl(args.candidates)]
    by_uid = {example.uid: example for example in candidates}
    uid_to_index = {example.uid: index for index, example in enumerate(candidates)}
    vectors = final_vectors(candidates)
    coordinates, explained = pca(vectors)
    groups = [broad_domain(example.domain) for example in candidates]

    ours_subset = Path(args.subsets_dir) / "ours.jsonl.gz"
    if not ours_subset.exists():
        raise FileNotFoundError(
            f"The fixed target subset is missing: {ours_subset}. Run the main experiment first."
        )
    target_pool = {row["uid"] for row in read_jsonl(ours_subset)}
    target_uids = round_robin_targets(candidates, target_pool, args.max_targets)
    target_set = set(target_uids)

    retrieval: dict[str, dict[str, list[str]]] = {}
    for method, filename in METHOD_FILES.items():
        path = Path(args.retrieval_dir) / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rows = {row["uid"]: row.get("demo_ids", []) for row in read_jsonl(path)}
        for uid in target_uids:
            if len(rows.get(uid, [])) != args.context_count:
                raise ValueError(
                    f"{method}/{uid} does not have exactly {args.context_count} contexts."
                )
        retrieval[method] = rows

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pca_coordinates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["uid", "dataset", "domain", "broad_domain", "pc1", "pc2"]
        )
        writer.writeheader()
        for example, group, point in zip(candidates, groups, coordinates):
            writer.writerow({
                "uid": example.uid,
                "dataset": example.dataset,
                "domain": example.domain,
                "broad_domain": group,
                "pc1": float(point[0]),
                "pc2": float(point[1]),
            })

    edge_rows = []
    statistics = []
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), constrained_layout=True)
    for axis, method in zip(axes.flat, METHOD_FILES):
        for group, color in GROUP_COLORS.items():
            indices = [index for index, value in enumerate(groups) if value == group]
            if indices:
                axis.scatter(
                    coordinates[indices, 0], coordinates[indices, 1], s=18,
                    color=color, alpha=0.66, edgecolors="none", label=group,
                )

        source_indices: set[int] = set()
        same_domain = 0
        same_group = 0
        similarities = []
        edge_count = 0
        for target_uid in target_uids:
            target_index = uid_to_index[target_uid]
            for rank, source_uid in enumerate(retrieval[method][target_uid], start=1):
                source_index = uid_to_index[source_uid]
                source_indices.add(source_index)
                similarity = float(vectors[target_index] @ vectors[source_index])
                similarities.append(similarity)
                same_domain += int(by_uid[target_uid].domain == by_uid[source_uid].domain)
                same_group += int(groups[target_index] == groups[source_index])
                edge_count += 1
                edge_rows.append({
                    "method": method,
                    "target_uid": target_uid,
                    "source_uid": source_uid,
                    "rank": rank,
                    "target_domain": by_uid[target_uid].domain,
                    "source_domain": by_uid[source_uid].domain,
                    "same_broad_domain": groups[target_index] == groups[source_index],
                    "final_layer_cosine": similarity,
                })
                axis.plot(
                    [coordinates[source_index, 0], coordinates[target_index, 0]],
                    [coordinates[source_index, 1], coordinates[target_index, 1]],
                    color=METHOD_COLORS[method], alpha=0.12, linewidth=0.65, zorder=1,
                )
        target_indices = [uid_to_index[uid] for uid in target_uids]
        axis.scatter(
            coordinates[target_indices, 0], coordinates[target_indices, 1],
            s=42, facecolors="none", edgecolors="#111111", linewidths=0.75,
            label="Fixed target queries", zorder=4,
        )
        source_indices_array = np.asarray(sorted(source_indices), dtype=int)
        axis.scatter(
            coordinates[source_indices_array, 0], coordinates[source_indices_array, 1],
            s=19, marker="s", color=METHOD_COLORS[method], alpha=0.72,
            label="Retrieved contexts", zorder=3,
        )
        statistics.append({
            "method": method,
            "targets": len(target_uids),
            "edges": edge_count,
            "unique_contexts": len(source_indices),
            "same_domain_rate": same_domain / edge_count,
            "same_broad_domain_rate": same_group / edge_count,
            "mean_final_layer_cosine": float(np.mean(similarities)),
        })
        axis.set_title(LABELS[method], fontsize=12, fontweight="semibold")
        axis.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% variance)")
        axis.grid(color="#e4e7ea", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(), unique.keys(), ncol=5, loc="upper center",
        bbox_to_anchor=(0.5, 1.035), frameon=False,
    )
    figure.suptitle(
        "Representation Retrieval Graphs (fixed queries, three contexts)",
        y=1.075, fontsize=14, fontweight="semibold",
    )
    figure.savefig(output / "retrieval_pca_graph.png", dpi=300, bbox_inches="tight")
    figure.savefig(output / "retrieval_pca_graph.pdf", bbox_inches="tight")
    plt.close(figure)

    for filename, rows in (
        ("retrieval_edges.csv", edge_rows),
        ("retrieval_graph_stats.csv", statistics),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved PCA retrieval graph and tables to {output}")


if __name__ == "__main__":
    main()
