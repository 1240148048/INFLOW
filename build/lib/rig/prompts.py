from __future__ import annotations

import re
from typing import Iterable

from .io import Example


ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer|choice)\s*(?:is|:)?\s*[\(\[]?([A-J])[\)\]]?",
    re.IGNORECASE,
)
JSON_ANSWER_RE = re.compile(
    r"""["']answer["']\s*:\s*["']?([A-J])["']?""",
    re.IGNORECASE,
)
BARE_CHOICE_RE = re.compile(r"(?:^|\s)[\(\[]?([A-J])[\)\]]?(?:\s|[.!]?$)")

def question_text(example: Example) -> str:
    if not example.choices:
        return example.question
    labels = example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]
    options = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, example.choices)
    )
    return (
        f"{example.question}\n\n{options}\n\n"
        "Return only the letter of the best answer and no other text."
    )


def answer_only_messages(example: Example) -> list[dict[str, str]]:
    return [{"role": "user", "content": question_text(example)}]


def training_messages(
    example: Example, demos: Iterable[Example] = ()
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for demo in demos:
        if not demo.rollout:
            continue
        messages.append({"role": "user", "content": question_text(demo)})
        messages.append({"role": "assistant", "content": demo.rollout})
    messages.extend(answer_only_messages(example))
    return messages


def extract_choice(text: str, valid_labels: list[str]) -> str | None:
    valid = {label.upper() for label in valid_labels}
    matches = JSON_ANSWER_RE.findall(text)
    for candidate in reversed(matches):
        if candidate.upper() in valid:
            return candidate.upper()
    matches = ANSWER_RE.findall(text)
    for candidate in reversed(matches):
        if candidate.upper() in valid:
            return candidate.upper()
    matches = BARE_CHOICE_RE.findall(text[-200:])
    for candidate in reversed(matches):
        if candidate.upper() in valid:
            return candidate.upper()
    return None
