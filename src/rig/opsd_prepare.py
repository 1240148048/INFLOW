from __future__ import annotations

import argparse
import copy
import gc
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import set_seed

from .io import Example, append_jsonl, load_config, read_jsonl
from .modeling import (
    hidden_size,
    lm_head,
    load_causal_model,
    load_tokenizer,
    mlp_activation_modules,
    mlp_down_projections,
    model_device,
    torch_dtype,
)
from .opsd_common import decoded_answer, generate_reasoning, labels_for, modal_answer


KEY = "neuron_opsd"


def completion_marker(path: Path) -> Path:
    return Path(str(path) + ".done")


class ResponseActivationCapture:
    """Capture post-nonlinearity MLP activations at response prediction positions."""

    def __init__(self, model, positions: list[int]):
        self.positions = torch.tensor(positions, dtype=torch.long)
        self.values: list[torch.Tensor | None] = [
            None for _ in mlp_activation_modules(model)
        ]
        self.handles = []
        for index, module in enumerate(mlp_activation_modules(model)):
            self.handles.append(module.register_forward_hook(self._hook(index)))

    def _hook(self, index: int):
        def save(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            positions = self.positions.to(value.device)
            self.values[index] = value[0].index_select(0, positions).detach()

        return save

    def close(self):
        for handle in self.handles:
            handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def rademacher(input_dim: int, output_dim: int, seed: int, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randint(
        0, 2, (input_dim, output_dim), generator=generator, dtype=torch.int8
    )
    return matrix.float().mul_(2).sub_(1).div_(math.sqrt(output_dim)).to(device)


def neuron_set(
    activations: list[torch.Tensor | None],
    down_projections,
    head,
    target_tokens: list[int],
    per_layer_topk: int,
    global_topk: int,
) -> list[int]:
    token_candidates: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
        [] for _ in target_tokens
    ]
    offset = 0
    target = torch.tensor(target_tokens, device=head.weight.device)
    token_embeddings = head.weight.index_select(0, target)
    for activation, projection in zip(activations, down_projections):
        if activation is None:
            raise RuntimeError("An MLP activation hook did not fire.")
        count = projection.weight.shape[1]
        direction = token_embeddings.to(projection.weight.dtype) @ projection.weight
        score = (activation.to(direction.dtype) * direction).float()
        keep = min(per_layer_topk, score.shape[1])
        values, indices = torch.topk(score, keep, dim=1)
        indices = indices.add(offset)
        for position in range(len(target_tokens)):
            token_candidates[position].append(
                (values[position].cpu(), indices[position].cpu())
            )
        offset += count
    # NEURON retains global Top-K contributors for every response token and
    # then takes their union. The resulting set size, unlike a single Top-K,
    # is a meaningful response-level consensus statistic.
    activated: set[int] = set()
    for candidates in token_candidates:
        values = torch.cat([item[0] for item in candidates])
        identifiers = torch.cat([item[1] for item in candidates])
        keep = min(global_topk, values.numel())
        top_values, top_indices = torch.topk(values, keep)
        activated.update(
            int(identifier)
            for value, identifier in zip(
                top_values.tolist(), identifiers[top_indices].tolist()
            )
            if value > 0.0
        )
    return sorted(activated)


def layer_embedding_matrix(
    hidden_states: tuple[torch.Tensor, ...],
    positions: list[int],
    projection: torch.Tensor,
) -> np.ndarray:
    index = torch.tensor(positions, device=projection.device)
    # hidden_states[0] is the embedding lookup; the remaining entries are
    # transformer-layer outputs. A shared projection preserves cosine geometry
    # while keeping the cached per-layer matrices compact.
    summaries = torch.stack([
        hidden_states[layer][0].index_select(0, index).float().mean(dim=0)
        for layer in range(1, len(hidden_states))
    ])
    projected = summaries @ projection
    return F.normalize(projected, dim=1).cpu().numpy().astype(np.float32)


def choice_token_id(tokenizer, label: str) -> int:
    token_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
    if len(token_ids) == 1:
        return int(token_ids[0])
    spaced = tokenizer(" " + label, add_special_tokens=False)["input_ids"]
    if not spaced:
        raise ValueError(f"Could not tokenize choice label {label!r}.")
    return int(spaced[-1])


def choice_distribution(logits, tokenizer, labels: list[str]) -> tuple[list[float], float]:
    identifiers = torch.tensor(
        [choice_token_id(tokenizer, label) for label in labels],
        dtype=torch.long,
        device=logits.device,
    )
    probabilities = logits.index_select(0, identifiers).softmax(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return probabilities.cpu().tolist(), float(entropy.item())


@torch.inference_mode()
def analyze_response(
    model,
    tokenizer,
    example: Example,
    prompt_ids: list[int],
    response: list[int],
    projection: torch.Tensor,
    capture_neurons: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    response = [int(token) for token in response]
    if response[-1] not in eos_ids and eos_ids:
        response.append(next(iter(eos_ids)))
    positions = [len(prompt_ids) - 1 + index for index in range(len(response))]
    input_ids = torch.tensor(
        [prompt_ids + response], dtype=torch.long, device=model_device(model)
    )
    attention_mask = torch.ones_like(input_ids)
    capture = ResponseActivationCapture(model, positions) if capture_neurons else None
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    finally:
        if capture is not None:
            capture.close()
    prediction_hidden = output.hidden_states[-1][0].index_select(
        0, torch.tensor(positions, device=input_ids.device)
    )
    logits = lm_head(model)(prediction_hidden.to(lm_head(model).weight.dtype)).float()
    log_probability = logits.log_softmax(dim=-1)
    probability = log_probability.exp()
    token_entropy = -(probability * log_probability).sum(dim=-1)
    entropy = float(token_entropy.mean().item())
    layer_embeddings = layer_embedding_matrix(
        output.hidden_states, positions, projection
    )
    identifiers: list[int] = []
    if capture is not None:
        identifiers = neuron_set(
            capture.values,
            mlp_down_projections(model),
            lm_head(model),
            response,
            int(config["neuron"]["per_layer_topk"]),
            int(config["neuron"]["global_topk"]),
        )
    text = tokenizer.decode(response, skip_special_tokens=True)
    answer_match = re.search(
        r'''["']answer["']\s*:\s*["']?([A-J])''', text, re.IGNORECASE
    )
    if answer_match is None:
        answer_position = len(response) - 1
    else:
        answer_position = len(
            tokenizer(
                text[: answer_match.start(1)], add_special_tokens=False
            )["input_ids"]
        )
        answer_position = min(max(answer_position, 0), len(response) - 1)
    labels = labels_for(example)
    option_probabilities, option_entropy = choice_distribution(
        logits[answer_position], tokenizer, labels
    )
    entropy_denominator = math.log(max(len(labels), 2))
    return {
        "text": text,
        "answer": decoded_answer(tokenizer, response, example),
        "mean_token_entropy": entropy,
        "answer_token_entropy": float(token_entropy[answer_position].item()),
        "choice_probabilities": dict(zip(labels, option_probabilities)),
        "choice_entropy": option_entropy,
        "normalized_choice_entropy": option_entropy / entropy_denominator,
        "token_count": len(response),
        "layer_embeddings": layer_embeddings.tolist(),
        "neuron_ids": identifiers,
    }


def choose_demonstration(samples: list[dict], modal: str | None) -> int:
    eligible = [
        index for index, sample in enumerate(samples) if sample["answer"] == modal
    ]
    if not eligible:
        eligible = list(range(len(samples)))
    return min(
        eligible,
        key=lambda index: (
            samples[index]["normalized_choice_entropy"],
            samples[index]["mean_token_entropy"],
            index,
        ),
    )


@torch.inference_mode()
def prepare_one(model, tokenizer, example, projection, config):
    prompt_ids, responses = generate_reasoning(
        model,
        tokenizer,
        example,
        (),
        max_prompt_tokens=int(config["generation"]["max_prompt_tokens"]),
        max_new_tokens=int(config["generation"]["max_new_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        num_return_sequences=int(config["generation"]["num_samples"]),
    )
    samples = [
        analyze_response(
            model,
            tokenizer,
            example,
            prompt_ids,
            response,
            projection,
            True,
            config,
        )
        for index, response in enumerate(responses)
    ]
    modal, agreement = modal_answer(samples, labels_for(example))
    demo_index = choose_demonstration(samples, modal)
    result = copy.deepcopy(example)
    result.rollout = samples[demo_index]["text"]
    result.metadata[KEY] = {
        "samples": samples,
        "modal_answer": modal,
        "vote_agreement": agreement,
        "demonstration_index": demo_index,
        "neuron_ids": samples[demo_index]["neuron_ids"],
        "activation_count": len(samples[demo_index]["neuron_ids"]),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Candidate preparation requires CUDA.")
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    name = args.model or config["model"]["name"]
    tokenizer = load_tokenizer(name, config["model"]["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_model(
        name,
        torch_dtype(config["model"]["dtype"]),
        config["model"]["trust_remote_code"],
    ).to("cuda:0").eval()
    projection = rademacher(
        hidden_size(model),
        int(config["representation"]["projection_dim"]),
        int(config["seed"]),
        model_device(model),
    )
    examples = [
        Example.from_dict(row) for row in read_jsonl(args.input) if row["split"] == "train"
    ]
    output = Path(args.output)
    completed = (
        {row["uid"] for row in read_jsonl(output)}
        if args.resume and output.exists()
        else set()
    )
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; pass --resume to reuse it.")
    if int(config["generation"]["num_samples"]) != 1:
        raise ValueError("Initial candidate preparation must use exactly one sample.")
    for example in tqdm(examples, desc="preparing one-shot retrieval statistics"):
        if example.uid in completed:
            continue
        append_jsonl(output, prepare_one(model, tokenizer, example, projection, config).to_dict())
        gc.collect()
        torch.cuda.empty_cache()
    completion_marker(output).write_text(str(len(examples)), encoding="utf-8")
    print(f"Prepared {len(examples)} shared candidates at {output}")


if __name__ == "__main__":
    main()
