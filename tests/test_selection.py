import unittest

import numpy as np
from rig.selection import (
    curriculum_coverage,
    effective_rank,
    greedy_conditional_information_gain,
    spectral_whiten_operators,
)


class SelectionTests(unittest.TestCase):
    def test_information_gain_prefers_a_new_direction(self):
        operators = [
            np.asarray([[4.0, 0.0]]),
            np.asarray([[3.0, 0.0]]),
            np.asarray([[0.0, 2.0]]),
        ]
        selected, gains, scores = greedy_conditional_information_gain(
            operators, count=2, noise=1.0
        )
        self.assertEqual(selected, [0, 2])
        self.assertTrue(all(gain > 0 for gain in gains))
        self.assertEqual(gains, scores)

    def test_spectral_whitening_is_finite(self):
        operators = [
            np.asarray([[1.0, 0.0], [2.0, 0.0]]),
            np.asarray([[0.0, 1.0]]),
        ]
        whitened, eigenvalues = spectral_whiten_operators(
            operators, rank=2, ridge=0.1
        )
        self.assertEqual([item.shape for item in whitened], [(2, 2), (1, 2)])
        self.assertTrue(all(np.isfinite(item).all() for item in whitened))
        self.assertTrue((eigenvalues >= 0).all())

    def test_effective_rank_detects_identical_and_orthogonal_traces(self):
        identical = [np.asarray([1.0, 0.0])] * 5
        orthogonal = [np.eye(5)[index] for index in range(5)]
        self.assertAlmostEqual(effective_rank(identical), 1.0)
        self.assertAlmostEqual(effective_rank(orthogonal), 5.0)

    def test_reliability_weights_conditional_gain(self):
        operators = [
            np.asarray([[2.0, 0.0]]),
            np.asarray([[0.0, 2.0]]),
        ]
        selected, _, _ = greedy_conditional_information_gain(
            operators,
            count=1,
            noise=1.0,
            reliability=[-0.5, 0.5],
        )
        self.assertEqual(selected, [1])

    def test_curriculum_coverage_is_normalized(self):
        operators = [
            np.asarray([[2.0, 0.0]]),
            np.asarray([[0.0, 2.0]]),
        ]
        selected, full, ratio = curriculum_coverage(
            operators, selected=[0], noise=1.0
        )
        self.assertGreater(full, selected)
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
