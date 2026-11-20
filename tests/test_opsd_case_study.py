import unittest

from rig.io import Example
from rig.opsd_case_study import build_case_record


def sample(text, answer, probabilities, uncertainty=0.2):
    return {
        "text": text,
        "answer": answer,
        "choice_probabilities": probabilities,
        "choice_entropy": 0.5,
        "normalized_choice_entropy": uncertainty,
        "mean_token_entropy": 1.0,
        "answer_token_entropy": 0.4,
        "token_count": 4,
    }


class CaseStudyTests(unittest.TestCase):
    def test_builds_complete_trace_and_reverses_prompt_order_for_rank(self):
        probabilities = {"A": 0.7, "B": 0.3}
        candidate = Example(
            uid="source",
            dataset="toy",
            domain="science",
            split="train",
            question="Source?",
            choices=["yes", "no"],
            choice_labels=["A", "B"],
            answer="A",
            rollout="source response",
            metadata={
                "neuron_opsd": {
                    "samples": [sample("source response", "A", probabilities)],
                    "demonstration_index": 0,
                }
            },
        )
        teacher_samples = [
            sample(f"teacher {index}", "B" if index < 4 else "A", {"A": 0.2, "B": 0.8})
            for index in range(5)
        ]
        target = Example(
            uid="target",
            dataset="toy",
            domain="science",
            split="train",
            question="Target?",
            choices=["yes", "no"],
            choice_labels=["A", "B"],
            answer="B",
            rollout="initial response",
            demo_ids=["source"],
            metadata={
                "neuron_opsd": {
                    "samples": [sample("initial response", "A", probabilities)]
                },
                "opsd_retrieval": {
                    "method": "ours",
                    "definition": "rcri_final_layer_v1",
                    "context_uids": ["source"],
                    "context_scores_in_prompt_order": [0.9],
                },
                "opsd_teacher_generation": {
                    "samples": teacher_samples,
                    "modal_answer": "B",
                    "vote_agreement": 0.8,
                    "mean_choice_probabilities": {"A": 0.2, "B": 0.8},
                    "best_sample_index": 0,
                    "best_answer": "B",
                    "distribution_shift_js": 0.1325054509170478,
                },
            },
        )
        selected = Example.from_dict(target.to_dict())
        selected.metadata["opsd_selection"] = {
            "rank": 1,
            "score": 0.1325054509170478,
            "criterion": "top JS",
        }

        record = build_case_record(target, {"source": candidate}, {"target": selected})

        self.assertEqual(record["initial"]["response"], "initial response")
        self.assertEqual(record["retrieval"]["contexts_in_prompt_order"][0]["retrieval_rank"], 1)
        self.assertEqual(record["teacher_generation"]["hard_vote_counts"], {"B": 4, "A": 1})
        self.assertTrue(record["data_selection"]["selected"])


if __name__ == "__main__":
    unittest.main()
