import unittest

import torch

from rig.chat_encoding import encode_messages
from rig.io import Example
from rig.prompts import answer_only_messages, question_text


class MappingTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        assert "Which element" in messages[0]["content"]
        assert "B. Helium" in messages[0]["content"]
        return {"input_ids": torch.tensor([[11, 12, 13]])}


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.example = Example(
            uid="x",
            dataset="test",
            domain="Chemistry",
            split="test",
            question="Which element has atomic number 2?",
            choices=["Hydrogen", "Helium"],
            choice_labels=["A", "B"],
        )

    def test_question_and_options_are_present(self):
        prompt = question_text(self.example)
        self.assertIn(self.example.question, prompt)
        self.assertIn("A. Hydrogen", prompt)
        self.assertIn("B. Helium", prompt)

    def test_messages_reach_chat_template(self):
        encoded = encode_messages(
            MappingTokenizer(),
            answer_only_messages(self.example),
            max_length=32,
            device=torch.device("cpu"),
        )
        self.assertEqual(encoded["input_ids"].tolist(), [[11, 12, 13]])

    def test_empty_content_is_rejected_before_tokenization(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            encode_messages(
                MappingTokenizer(),
                [{"role": "user", "content": None}],
                max_length=32,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
