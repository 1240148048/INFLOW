from __future__ import annotations

from collections import Counter
from typing import Iterable

import torch

from .chat_encoding import encode_messages
from .io import Example
from .model_inputs import trim_response
from .modeling import model_device
from .prompts import extract_choice


def labels_for(example: Example) -> list[str]:
    return example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]


def reasoning_question(example: Example) -> str:
    labels = labels_for(example)
    options = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, example.choices)
    )
    return (
        f"{example.question}\n\n{options}\n\n"
        "Solve the problem internally, then return exactly one valid JSON object "
        "and no other text. Use this schema: "
        '{"knowledge":["keyword 1","keyword 2"],'
        '"method":"one concise sentence describing the solution method",'
        '"result":"the complete text of the selected option",'
        '"answer":"X"}. '
        "Knowledge must contain only a few short keywords, method must be exactly "
        "one concise sentence, result must reproduce the selected option content, "
        "and answer must be its option letter."
    )


def reasoning_messages(
    example: Example, demonstrations: Iterable[Example] = ()
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for demonstration in demonstrations:
        if not demonstration.rollout:
            continue
        messages.append({"role": "user", "content": reasoning_question(demonstration)})
        messages.append({"role": "assistant", "content": demonstration.rollout})
    messages.append({"role": "user", "content": reasoning_question(example)})
    return messages


@torch.inference_mode()
def generate_reasoning(
    model,
    tokenizer,
    example: Example,
    demonstrations: Iterable[Example],
    *,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int = 1,
) -> tuple[list[int], list[list[int]]]:
    encoded = encode_messages(
        tokenizer,
        reasoning_messages(example, demonstrations),
        max_prompt_tokens,
        model_device(model),
        enable_thinking=False,
    )
    prompt_width = encoded["input_ids"].shape[1]
    generated = model.generate(
        **encoded,
        use_cache=True,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    responses = [
        trim_response(
            row[prompt_width:].tolist(),
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        for row in generated
    ]
    return encoded["input_ids"][0].tolist(), responses


def decoded_answer(tokenizer, response: list[int], example: Example) -> str | None:
    text = tokenizer.decode(response, skip_special_tokens=True)
    return extract_choice(text, labels_for(example))


def modal_answer(samples: list[dict], labels: list[str]) -> tuple[str | None, float]:
    counts = Counter(
        sample.get("answer") for sample in samples if sample.get("answer") in labels
    )
    if not counts:
        return None, 0.0
    largest = max(counts.values())
    winner = next(label for label in labels if counts.get(label, 0) == largest)
    return winner, largest / len(samples)
