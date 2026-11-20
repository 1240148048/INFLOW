import unittest

from rig.io import Example
from rig.opsd_ablation_finalize import js_selection, random_selection


def example(uid: str, score: float) -> Example:
    return Example(
        uid=uid,
        dataset="toy",
        domain="toy",
        split="train",
        question=uid,
        metadata={
            "opsd_teacher_generation": {"distribution_shift_js": score},
            "opsd_retrieval": {"definition": "test"},
        },
    )


class AblationFinalizeTests(unittest.TestCase):
    def test_js_selection_keeps_highest_scores(self):
        selected, _ = js_selection(
            [example("low", 0.1), example("high", 0.9)],
            "no_certainty",
            0.5,
        )
        self.assertEqual([item.uid for item in selected], ["high"])
        self.assertEqual(
            selected[0].metadata["opsd_selection"]["method"], "no_certainty"
        )

    def test_random_selection_is_reproducible(self):
        examples = [example(str(index), index) for index in range(10)]
        left, _ = random_selection(examples, 0.2, 42)
        right, _ = random_selection(examples, 0.2, 42)
        self.assertEqual([item.uid for item in left], [item.uid for item in right])
        self.assertEqual(len(left), 2)


if __name__ == "__main__":
    unittest.main()
