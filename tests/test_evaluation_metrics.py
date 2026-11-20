import unittest

from rig.evaluate_single_gpu import summarize


class EvaluationMetricTests(unittest.TestCase):
    def test_rollout_columns_have_requested_mean_and_sample_std(self):
        rows = [
            {
                "answer": "A",
                "predictions": ["A", "A", "B"],
                "majority_prediction": "A",
                "confidence": 2 / 3,
                "completion_lengths": [1, 1, 1],
                "finished": [True, True, True],
            },
            {
                "answer": "B",
                "predictions": ["B", "A", "B"],
                "majority_prediction": "B",
                "confidence": 2 / 3,
                "completion_lengths": [1, 1, 1],
                "finished": [True, True, True],
            },
        ]
        metrics = summarize(rows, bins=15, num_rollouts=3)
        self.assertEqual(metrics["rollout_index_accuracies"], [1.0, 0.5, 0.5])
        self.assertAlmostEqual(metrics["rollout_accuracy_mean"], 2 / 3)
        self.assertAlmostEqual(metrics["avg_at_3"], 2 / 3)
        self.assertAlmostEqual(metrics["rollout_accuracy_std"], 0.28867513459481287)
        self.assertEqual(metrics["maj_at_3"], 1.0)


if __name__ == "__main__":
    unittest.main()
