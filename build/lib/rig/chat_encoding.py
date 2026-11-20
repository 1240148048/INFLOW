from __future__ import annotations

from collections.abc import Mapping

import torch


def encode_messages(
    tokenizer,
    messages: list[dict[str, str]],
    max_length: int,
    device,
    enable_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    if not messages or any(
        not isinstance(message.get("content"), str)
        or not message["content"].strip()
        for message in messages
    ):
        raise ValueError("Chat messages must contain non-empty text content.")
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    try:
        encoded = tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(encoded, torch.Tensor):
        input_ids = encoded
    elif isinstance(encoded, Mapping):
        input_ids = encoded["input_ids"]
    elif isinstance(encoded, str):
        input_ids = tokenizer(
            encoded,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
    elif (
        isinstance(encoded, (list, tuple))
        and encoded
        and isinstance(encoded[0], str)
    ):
        input_ids = tokenizer(
            encoded[0],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
    else:
        input_ids = torch.tensor([encoded], dtype=torch.long)
    input_ids = input_ids[:, -max_length:]
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("Chat template produced an empty token sequence.")
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": torch.ones_like(input_ids, device=device),
    }
