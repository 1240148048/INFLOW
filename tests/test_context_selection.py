import numpy as np

from rig.context_selection import (
    activation_contexts,
    cosine_contexts,
    vote_answer,
    vote_disagreement,
)
from rig.io import Example


def example(answers):
    return Example(
        uid="x",
        dataset="d",
        domain="d",
        split="train",
        question="q",
        choices=["one", "two", "three"],
        choice_labels=["A", "B", "C"],
        metadata={"rig": {"sample_summaries": [{"answer": x} for x in answers]}},
    )


def test_vote_statistics_use_all_rollouts():
    item = example(["B", "A", "B", "C", "B"])
    assert vote_answer(item) == "B"
    assert vote_disagreement(item) == 0.4


def test_cosine_retrieval_excludes_query_and_places_best_last():
    features = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    contexts = cosine_contexts(features, [0], [0, 1, 2], 2)
    assert contexts[0] == [2, 1]


def test_activation_retrieval_uses_jaccard_overlap():
    contexts = activation_contexts(
        [{1, 2, 3}, {1, 2}, {8, 9}, {1, 2, 3, 4}], [0], 2
    )
    assert contexts[0] == [1, 3]
