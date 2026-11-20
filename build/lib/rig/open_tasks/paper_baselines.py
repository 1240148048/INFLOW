"""Open-output instantiations of the retrieval and selection rules in Table 5."""

import numpy as np
from .runtime import output_contract


PAPER_METHODS = ("full", "redundancy", "consistency", "neuron", "mutual", "js")
METHODS = ("original", *PAPER_METHODS, "random", "hard_flip", "entropy", "agreement")
FAMILIES = {"full": "full", "redundancy": "full", "consistency": "consistency",
            "neuron": "neuron", "mutual": "mutual"}


def family(method):
    return FAMILIES.get(method, "js")


def unit_vectors(values):
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def paper_contexts(rows, count, method, neurons=None):
    if method == "consistency":
        return {row["uid"]: ([], 0.0) for row in rows}
    matrices = unit_vectors([row["trajectory"] for row in rows])
    if method == "full":
        gram = matrices[:, -1] @ matrices[:, -1].T
    elif method == "mutual":
        vectors = unit_vectors(matrices.reshape(len(rows), -1))
        similarity = np.clip(vectors @ vectors.T, 0, 1)
        certainty = np.asarray([row["certainty"] for row in rows])
        gram = -.5 * np.log1p(-np.minimum(similarity**2 * certainty[None, :], 1 - 1e-6))
    elif method != "neuron":
        raise ValueError(f"Unknown baseline retrieval: {method}")
    pools = {}
    for i, row in enumerate(rows):
        pools.setdefault(output_contract(row), []).append(i)
    result = {}
    for pool in pools.values():
        if len(pool) <= count:
            raise ValueError("Each task must have more training rows than retrieval depth")
        if method == "neuron":
            from scipy.sparse import csr_matrix
            neuron_sets = [sorted(set(neurons[rows[i]["uid"]])) for i in pool]
            if any(not s for s in neuron_sets):
                raise ValueError("NEURON requires nonempty contributing-neuron sets")
            ii = [i for i, s in enumerate(neuron_sets) for _ in s]
            jj = [j for s in neuron_sets for j in s]
            incidence = csr_matrix((np.ones(len(ii), dtype=np.int32), (ii, jj)))
            intersections = (incidence @ incidence.T).toarray()
            sizes = np.asarray([len(s) for s in neuron_sets])
            overlap = intersections / (sizes[:, None] + sizes[None, :] - intersections)
        for local, i in enumerate(pool):
            eligible = [j for j in range(len(pool)) if j != local]
            scores = overlap[local] if method == "neuron" else gram[i, pool]
            indices = sorted(eligible, key=lambda j: (-scores[j], rows[pool[j]]["uid"]))[:count]
            value = float(.5 * np.log1p(np.expm1(2 * np.maximum(scores[indices], 0)).sum())) if method == "mutual" else 0.0
            result[rows[i]["uid"]] = ([rows[pool[j]]["uid"] for j in reversed(indices)], value)
    return result


def redundancy_subset(order, teacher, count):
    vectors = unit_vectors([teacher[uid]["final_representation"] for uid in order])
    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -np.inf)
    retained = list(range(len(order)))
    while len(retained) > count:
        scores = similarity[np.ix_(retained, retained)].max(axis=1)
        retained.pop(int(np.argmax(scores)))
    return [order[i] for i in retained]
