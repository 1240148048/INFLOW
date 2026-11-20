import unittest

from rig.prepare_comprehensive import (
    MMLU_SPECS,
    SCIKNOW_COUNTS,
    mmlu_examples,
    sciknow_examples,
    split_ordered,
)


class ComprehensivePreparationTests(unittest.TestCase):
    def test_requested_training_total_is_500(self):
        science = sum(SCIKNOW_COUNTS.values())
        mmlu = sum(spec[2] for spec in MMLU_SPECS.values())
        self.assertEqual(science, 200)
        self.assertEqual(mmlu, 300)
        self.assertEqual(science + mmlu, 500)

    def test_test_split_immediately_follows_train(self):
        train, test = split_ordered(list(range(12)), count=5, name="sample")
        self.assertEqual(train, [0, 1, 2, 3, 4])
        self.assertEqual(test, [5, 6, 7, 8, 9])
        self.assertFalse(set(train) & set(test))

    def test_complete_synthetic_build_has_disjoint_500_500_split(self):
        science_rows = []
        source_index = 0
        for domain, count in SCIKNOW_COUNTS.items():
            for _ in range(2 * count):
                science_rows.append(
                    {
                        "_source_index": source_index,
                        "domain": domain,
                        "question": f"science-{source_index}",
                        "choices": ["one", "two", "three", "four"],
                        "answerKey": "A",
                    }
                )
                source_index += 1

        mmlu_rows = []
        question_id = 0
        for _domain, (field, value, count) in MMLU_SPECS.items():
            for _ in range(2 * count):
                row = {
                    "question_id": question_id,
                    "question": f"mmlu-{question_id}",
                    "options": [str(index) for index in range(10)],
                    "answer": "A",
                    "category": "other",
                    "src": "synthetic",
                }
                row[field] = value
                mmlu_rows.append(row)
                question_id += 1

        examples = sciknow_examples(science_rows) + mmlu_examples(mmlu_rows)
        train = [example.uid for example in examples if example.split == "train"]
        test = [example.uid for example in examples if example.split == "test"]
        self.assertEqual(len(train), 500)
        self.assertEqual(len(test), 500)
        self.assertFalse(set(train) & set(test))


if __name__ == "__main__":
    unittest.main()
