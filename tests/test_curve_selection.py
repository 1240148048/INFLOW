import unittest

import numpy as np

from rig.curve_selection import (
    difficulty_agreement,
    redundancy_elimination_order,
    selected_indices,
)
from rig.io import Example
from rig.qa_information import question_answer_information_order


def example_with_answers(answers, standard_index=0):
    return Example(
        uid="x",
        dataset="test",
        domain="test",
        split="train",
        question="q",
        metadata={
            "rig": {
                "sample_summaries": [{"answer": answer} for answer in answers],
                "standard_sample_index": standard_index,
                "standard_answer": answers[standard_index],
            }
        },
    )


class CurveSelectionTests(unittest.TestCase):
    def test_difficulty_excludes_the_canonical_sample(self):
        example = example_with_answers(["A", "A", "A", "B", "B"])
        self.assertEqual(difficulty_agreement(example), 0.5)

    def test_nearest_max_removes_a_duplicate_before_isolated_points(self):
        embeddings = np.asarray(
            [[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], dtype=np.float64
        )
        order = redundancy_elimination_order(embeddings)
        self.assertIn(order[0], {0, 1})
        retained = selected_indices("redundancy", order, count=2)
        self.assertIn(2, retained)
        self.assertEqual(len(set(retained) & {0, 1}), 1)

    def test_redundancy_subsets_follow_one_nested_elimination_path(self):
        order = [3, 1, 0, 2]
        small = set(selected_indices("redundancy", order, 2))
        large = set(selected_indices("redundancy", order, 3))
        self.assertTrue(small < large)

    def test_information_retrieval_prefers_reliable_source_near_uncertain_target(self):
        gradients = ([1.0, 0.0], [1.0, 0.0], [0.0, 1.0])
        question_features = ([0.0, 0.0], [0.05, 0.0], [4.0, 4.0])
        examples = []
        for index, (gradient, question_feature) in enumerate(
            zip(gradients, question_features)
        ):
            sampled = [gradient] * 5
            if index == 1:
                sampled = [
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [0.0, 1.0],
                ]
            examples.append(
                Example(
                    uid=str(index),
                    dataset="test",
                    domain="test",
                    split="train",
                    question="q",
                    metadata={
                        "qair": {
                            "version": "qair_uncertainty_retrieval_v4",
                            "gradient_feature": gradient,
                            "sampled_gradient_features": sampled,
                            "question_change_feature": question_feature,
                            "gradient_noise": 1e-6,
                        }
                    },
                )
            )
        order, diagnostics = question_answer_information_order(
            examples, ridge=1.0, maximize=True
        )
        self.assertEqual(order[0], 0)
        self.assertEqual(
            diagnostics["rule"],
            "uncertainty_directed_pairwise_information_retrieval",
        )


if __name__ == "__main__":
    unittest.main()
