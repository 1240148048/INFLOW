from __future__ import annotations

import torch


def trim_response(
    tokens: list[int], eos_token_id: int | list[int] | None, pad_token_id: int
) -> list[int]:
    eos = (
        set(eos_token_id)
        if isinstance(eos_token_id, list)
        else {eos_token_id}
        if eos_token_id is not None
        else set()
    )
    result = []
    for token in tokens:
        if token == pad_token_id:
            break
        result.append(token)
        if token in eos:
            break
    return result or [pad_token_id]


def distillation_inputs(
    tokenizer,
    prompt: str,
    response: list[int],
    max_sequence_tokens: int,
    device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    max_prompt = max_sequence_tokens - len(response) + 1
    if max_prompt < 1:
        raise ValueError("Response is longer than max_sequence_tokens.")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_prompt:]
    sequence = prompt_ids + response[:-1]
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    mask = torch.ones((1, len(response)), dtype=torch.float32, device=device)
    return {"input_ids": input_ids, "attention_mask": attention}, mask


def distillation_inputs_from_prompt_ids(
    prompt_ids: list[int],
    response: list[int],
    max_sequence_tokens: int,
    device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    max_prompt = max_sequence_tokens - len(response) + 1
    if max_prompt < 1:
        raise ValueError("Response is longer than max_sequence_tokens.")
    sequence = prompt_ids[-max_prompt:] + response[:-1]
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    mask = torch.ones((1, len(response)), dtype=torch.float32, device=device)
    return {"input_ids": input_ids, "attention_mask": attention}, mask
