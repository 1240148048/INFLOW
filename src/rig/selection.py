from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import set_seed

from .activation import NeuronActivationCapture
from .chat_encoding import encode_messages
from .io import Example, append_jsonl, load_config, read_jsonl, write_jsonl
from .model_inputs import trim_response
from .modeling import (
    hidden_size,
    language_backbone,
    lm_head,
    load_causal_model,
    load_tokenizer,
    model_device,
    torch_dtype,
)
from .prompts import ANSWER_RE, answer_only_messages, extract_choice


METHOD_KEY = "rig"


def metadata(example: Example) -> dict[str, Any]:
    return example.metadata.setdefault(METHOD_KEY, {})


def load_model(config: dict[str, Any], override: str | None):
    spec = config["model"]
    name = override or spec["name"]
    tokenizer = load_tokenizer(name, spec["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_model(
        name,
        torch_dtype(spec["dtype"]),
        spec["trust_remote_code"],
    ).to("cuda:0")
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model.eval(), tokenizer


def rademacher(
    input_dim: int, output_dim: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    signs = torch.randint(
        0, 2, (input_dim, output_dim), generator=generator, dtype=torch.int8
    )
    return signs.float().mul_(2).sub_(1).div_(math.sqrt(output_dim)).to(device)


def option_token_ids(tokenizer, labels: list[str]) -> list[list[int]]:
    output = []
    for label in labels:
        variants = set()
        for text in (label, " " + label):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                variants.add(int(ids[-1]))
        if not variants:
            raise ValueError(f"No token id found for choice {label!r}.")
        output.append(sorted(variants))
    return output


def option_distribution(
    logits: torch.Tensor, token_ids: list[list[int]]
) -> torch.Tensor:
    scores = []
    for variants in token_ids:
        values = logits[torch.tensor(variants, device=logits.device)]
        scores.append(torch.logsumexp(values.float(), dim=0))
    return torch.softmax(torch.stack(scores), dim=0)


def answer_character_position(text: str, labels: list[str]) -> tuple[str | None, int | None]:
    valid = {label.upper() for label in labels}
    matches = list(ANSWER_RE.finditer(text))
    for match in reversed(matches):
        answer = match.group(1).upper()
        if answer in valid:
            return answer, match.start(1)
    answer = extract_choice(text, labels)
    if answer is None:
        return None, None
    bare = list(re.finditer(rf"\b{re.escape(answer)}\b", text, re.IGNORECASE))
    return (answer, bare[-1].start() if bare else None)


def effective_rank(embeddings: list[np.ndarray]) -> float:
    matrix = np.stack(embeddings).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    singular = np.linalg.svd(matrix, compute_uv=False)
    total = float(singular.sum())
    if total <= 1e-12:
        return 1.0
    probability = singular / total
    entropy = -float(
        np.sum(probability * np.log(np.maximum(probability, 1e-12)))
    )
    return float(np.clip(np.exp(entropy), 1.0, len(embeddings)))


@torch.inference_mode()
def generate_traces(model, tokenizer, example: Example, config: dict[str, Any]):
    generation = config["generation"]
    encoded = encode_messages(
        tokenizer,
        answer_only_messages(example),
        int(generation["max_prompt_tokens"]),
        model_device(model),
        enable_thinking=False,
    )
    count = int(generation.get("num_samples", 5))
    labels = example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]
    eos = tokenizer.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    if eos is None:
        raise ValueError("Answer-only generation requires an EOS token.")
    allowed_sequences = []
    for label in labels:
        for variant in (label, " " + label):
            tokens = tokenizer(variant, add_special_tokens=False)["input_ids"]
            if tokens:
                allowed_sequences.append([int(token) for token in tokens] + [eos])
    prompt_width = encoded["input_ids"].shape[1]

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prefix = input_ids[prompt_width:].tolist()
        allowed = {
            sequence[len(prefix)]
            for sequence in allowed_sequences
            if len(prefix) < len(sequence) and sequence[: len(prefix)] == prefix
        }
        return sorted(allowed) if allowed else [eos]

    output = model.generate(
        **encoded,
        do_sample=True,
        num_return_sequences=count,
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        prefix_allowed_tokens_fn=allowed_tokens,
    )
    traces = []
    for sequence in output:
        response = trim_response(
            sequence[prompt_width:].tolist(),
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
        )
        traces.append(response)
    del output
    return encoded["input_ids"][0].tolist(), traces


@torch.inference_mode()
def analyze_trace(
    model,
    tokenizer,
    prompt_ids: list[int],
    response: list[int],
    labels: list[str],
    token_ids: list[list[int]],
) -> dict[str, Any]:
    device = model_device(model)
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    content_response = []
    target_response = []
    for token in response:
        if token == tokenizer.pad_token_id:
            break
        target_response.append(int(token))
        if token in eos_ids:
            break
        content_response.append(int(token))
    if not content_response:
        content_response = [int(response[0])]
    if not target_response:
        target_response = list(content_response)
    text = tokenizer.decode(content_response, skip_special_tokens=True)
    answer, character_position = answer_character_position(text, labels)
    if character_position is None:
        response_position = max(len(content_response) - 1, 0)
    else:
        response_position = len(
            tokenizer(
                text[:character_position], add_special_tokens=False
            )["input_ids"]
        )
        response_position = min(
            max(response_position, 0), max(len(content_response) - 1, 0)
        )
    input_ids = torch.tensor(
        [prompt_ids + target_response], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    prediction_positions = [
        len(prompt_ids) - 1 + index for index in range(len(target_response))
    ]
    answer_prediction_position = len(prompt_ids) - 1 + response_position
    with NeuronActivationCapture(model, [answer_prediction_position]) as capture:
        output = language_backbone(model)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    all_hidden = output.last_hidden_state[0]
    index = torch.tensor(prediction_positions, device=device)
    prediction_hidden = all_hidden.index_select(0, index)
    logits = lm_head(model)(prediction_hidden).float()
    log_probabilities = logits.log_softmax(dim=-1)
    probabilities = log_probabilities.exp()
    entropy = float((-(probabilities * log_probabilities).sum(dim=-1)).mean().item())
    answer_hidden = all_hidden[len(prompt_ids) + response_position]
    option_probabilities = option_distribution(logits[0], token_ids)
    parsed = answer is not None
    if answer is None:
        sampled = int(torch.multinomial(option_probabilities, 1).item())
        answer = labels[sampled]
    neuron_ids = capture.activation_set([content_response[response_position]])
    result = {
        "answer": answer,
        "predictive_entropy": entropy,
        "parsed_answer": parsed,
        "generated_tokens": len(target_response),
        "embedding": answer_hidden.float().cpu().numpy().astype(np.float32),
        "activation_count": len(neuron_ids),
        "neuron_ids": neuron_ids,
    }
    del input_ids, attention_mask, output, all_hidden, logits, probabilities
    return result


@torch.inference_mode()
def prepare_example(
    model,
    tokenizer,
    example: Example,
    config: dict[str, Any],
    projection: torch.Tensor,
) -> tuple[Example, Example]:
    prompt_ids, responses = generate_traces(model, tokenizer, example, config)
    labels = example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]
    label_token_ids = option_token_ids(tokenizer, labels)
    samples = [
        analyze_trace(
            model, tokenizer, prompt_ids, response, labels, label_token_ids
        )
        for response in responses
    ]
    rank = effective_rank([sample["embedding"] for sample in samples])
    reliability = float(np.clip(2.0 / rank - 1.0, -1.0, 1.0))
    standard_index = int(
        np.argmin([sample["predictive_entropy"] for sample in samples])
    )
    activation_union = set()
    for sample in samples:
        activation_union.update(sample["neuron_ids"])
    standard_embedding = torch.from_numpy(
        samples[standard_index]["embedding"]
    ).to(projection.device)
    normalized = F.normalize(standard_embedding, dim=0)
    projected = F.normalize(normalized @ projection, dim=0)
    summaries = [
        {
            key: sample[key]
            for key in (
                "answer",
                "predictive_entropy",
                "parsed_answer",
                "generated_tokens",
                "activation_count",
                "neuron_ids",
            )
        }
        for sample in samples
    ]
    standard = copy.deepcopy(example)
    standard.rollout = samples[standard_index]["answer"]
    metadata(standard).update(
        {
            "sample_summaries": summaries,
            "first_answer": samples[0]["answer"],
            "standard_answer": standard.rollout,
            "standard_sample_index": standard_index,
            "effective_rank": rank,
            "reliability": reliability,
            "standard_predictive_entropy": samples[standard_index][
                "predictive_entropy"
            ],
            "standard_activation_count": len(activation_union),
            "activation_neuron_ids": sorted(activation_union),
            "standard_trace_activation_count": samples[standard_index]["activation_count"],
            "standard_embedding": normalized.cpu().tolist(),
            "projected_embedding": projected.cpu().tolist(),
            "need_operator": [projected.cpu().tolist()],
        }
    )
    first = copy.deepcopy(standard)
    first.rollout = samples[0]["answer"]
    metadata(first)["training_variant"] = "first_sample"
    metadata(standard)["training_variant"] = "minimum_entropy_standard"
    return first, standard


def prepare_command(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Candidate preparation requires one CUDA GPU.")
    config = load_config(args.config)
    set_seed(config["seed"])
    model, tokenizer = load_model(config, args.model)
    projection_dim = int(config["selection"]["embedding_projection_dim"])
    projection = rademacher(
        hidden_size(model),
        projection_dim,
        config["seed"],
        torch.device("cuda:0"),
    )
    examples = [
        Example.from_dict(row)
        for row in read_jsonl(args.input)
        if row["split"] == "train"
    ]
    standard_output = Path(args.output)
    first_output = Path(args.first_output)
    standard_done = (
        {row["uid"] for row in read_jsonl(standard_output)}
        if args.resume and standard_output.exists()
        else set()
    )
    first_done = (
        {row["uid"] for row in read_jsonl(first_output)}
        if args.resume and first_output.exists()
        else set()
    )
    if standard_done != first_done:
        raise ValueError("First-sample and standard candidate files are out of sync.")
    if not args.resume and (standard_output.exists() or first_output.exists()):
        raise FileExistsError("Candidate output exists. Use --resume.")
    pending = [example for example in examples if example.uid not in standard_done]
    for example in tqdm(pending, desc="sampling five self-distillation traces"):
        first, standard = prepare_example(
            model, tokenizer, example, config, projection
        )
        append_jsonl(first_output, first.to_dict())
        append_jsonl(standard_output, standard.to_dict())
        gc.collect()
        torch.cuda.empty_cache()
    print(
        f"Prepared {len(examples)} first-sample and minimum-entropy targets."
    )


def spectral_whiten_operators(
    operators: list[np.ndarray], rank: int, ridge: float
) -> tuple[list[np.ndarray], np.ndarray]:
    feature_dim = operators[0].shape[1]
    second_moment = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    for operator in operators:
        if not np.isfinite(operator).all():
            raise ValueError("Embedding operator contains non-finite values.")
        second_moment += operator.T.astype(np.float64) @ operator
    second_moment /= max(len(operators), 1)
    eigenvalues, basis = np.linalg.eigh(second_moment)
    order = np.argsort(eigenvalues)[::-1]
    keep = min(max(int(rank), 1), feature_dim)
    eigenvalues = eigenvalues[order[:keep]].clip(min=0.0)
    basis = basis[:, order[:keep]]
    scale = max(float(eigenvalues[0]), 1e-12)
    transform = basis / np.sqrt(eigenvalues[None, :] + ridge * scale)
    whitened = [
        (operator @ transform).astype(np.float64) for operator in operators
    ]
    return whitened, eigenvalues.astype(np.float32)


def greedy_conditional_information_gain(
    operators: list[np.ndarray],
    count: int,
    noise: float,
    reliability: list[float] | None = None,
    eligible: list[bool] | None = None,
    maximize: bool = True,
) -> tuple[list[int], list[float], list[float]]:
    feature_dim = operators[0].shape[1]
    inverse = np.eye(feature_dim, dtype=np.float64) / max(noise, 1e-12)
    available = np.asarray(
        eligible if eligible is not None else [True] * len(operators),
        dtype=bool,
    )
    weights = np.asarray(
        reliability if reliability is not None else [1.0] * len(operators),
        dtype=np.float64,
    )
    selected, gains, scores = [], [], []
    for _ in range(min(count, int(available.sum()))):
        candidates = np.flatnonzero(available)
        if all(operators[int(index)].shape[0] == 1 for index in candidates):
            rows = np.vstack([operators[int(index)][0] for index in candidates])
            quadratic = np.einsum(
                "ij,jk,ik->i", rows, inverse, rows, optimize=True
            ).clip(min=0.0)
            candidate_gains = 0.5 * np.log1p(quadratic)
        else:
            candidate_gains = []
            for index in candidates:
                operator = operators[int(index)]
                middle = np.eye(operator.shape[0]) + operator @ inverse @ operator.T
                sign, logdet = np.linalg.slogdet(middle)
                candidate_gains.append(0.5 * logdet if sign > 0 else -np.inf)
            candidate_gains = np.asarray(candidate_gains)
        candidate_scores = weights[candidates] * candidate_gains
        local = int(np.argmax(candidate_scores) if maximize else np.argmin(candidate_scores))
        index = int(candidates[local])
        gain = float(candidate_gains[local])
        score = float(candidate_scores[local])
        operator = operators[index]
        if operator.shape[0] == 1:
            vector = operator[0]
            inverse_vector = inverse @ vector
            denominator = 1.0 + float(vector @ inverse_vector)
            inverse -= np.outer(inverse_vector, inverse_vector) / denominator
        else:
            middle = np.eye(operator.shape[0]) + operator @ inverse @ operator.T
            solved = np.linalg.solve(middle, operator @ inverse)
            inverse -= inverse @ operator.T @ solved
        inverse = 0.5 * (inverse + inverse.T)
        available[index] = False
        selected.append(index)
        gains.append(gain)
        scores.append(score)
    return selected, gains, scores


def curriculum_coverage(
    operators: list[np.ndarray], selected: list[int], noise: float
) -> tuple[float, float, float]:
    feature_dim = operators[0].shape[1]

    def information(indices) -> float:
        precision = np.eye(feature_dim, dtype=np.float64)
        for index in indices:
            operator = operators[int(index)]
            precision += operator.T @ operator / max(noise, 1e-12)
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0:
            raise ValueError("Information matrix is not positive definite.")
        return 0.5 * float(logdet)

    selected_information = information(selected)
    full_information = information(range(len(operators)))
    ratio = selected_information / max(full_information, 1e-12)
    return selected_information, full_information, float(np.clip(ratio, 0.0, 1.0))


def select_command(args) -> None:
    config = load_config(args.config)
    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    operators = [
        np.asarray(metadata(example)["need_operator"], dtype=np.float32)
        for example in examples
    ]
    reliability = [float(metadata(example)["reliability"]) for example in examples]
    whitened, eigenvalues = spectral_whiten_operators(
        operators, args.rank, args.ridge
    )
    if config["selection"].get("balance_domains", False):
        domain_counts = defaultdict(int)
        for example in examples:
            domain_counts[example.domain] += 1
        domains = sorted(domain_counts)
        domain_index = {domain: index for index, domain in enumerate(domains)}
        base_dim = whitened[0].shape[1]
        total = len(examples)
        conditioned = []
        for operator, example in zip(whitened, examples):
            block = np.zeros(
                (operator.shape[0], base_dim * len(domains)), dtype=np.float64
            )
            start = domain_index[example.domain] * base_dim
            weight = total / (len(domains) * domain_counts[example.domain])
            block[:, start : start + base_dim] = operator * math.sqrt(weight)
            conditioned.append(block)
        whitened = conditioned
    fraction = args.fraction or float(config["selection"]["fraction"])
    eligible = [value > 0.0 for value in reliability] if args.positive_only else None
    selection_pool_size = sum(eligible) if eligible is not None else len(examples)
    count = max(1, int(selection_pool_size * fraction))
    indices, gains, scores = greedy_conditional_information_gain(
        whitened,
        count,
        args.noise,
        reliability=reliability,
        eligible=eligible,
    )
    selected_information, full_information, coverage = curriculum_coverage(
        whitened, indices, args.noise
    )
    output = []
    domain_selection = defaultdict(int)
    for order, (index, gain, score) in enumerate(
        zip(indices, gains, scores), start=1
    ):
        example = examples[index]
        metadata(example).update(
            {
                "selection_order": order,
                "conditional_information_gain": gain,
                "reliability_weighted_gain": score,
                "curriculum_coverage": coverage,
                "positive_only": bool(args.positive_only),
            }
        )
        domain_selection[example.domain] += 1
        output.append(example)
    write_jsonl(args.output, (example.to_dict() for example in output))
    metrics = {
        "candidate_count": len(examples),
        "positive_candidate_count": int(sum(value > 0 for value in reliability)),
        "selected_count": len(output),
        "target_count": count,
        "positive_only": bool(args.positive_only),
        "mean_candidate_reliability": float(np.mean(reliability)),
        "mean_selected_reliability": float(
            np.mean([reliability[index] for index in indices])
        ),
        "spectral_rank": len(eigenvalues),
        "selected_information": selected_information,
        "full_information": full_information,
        "curriculum_coverage": coverage,
        "domain_counts": dict(sorted(domain_selection.items())),
    }
    Path(str(args.output) + ".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RIG reliability and selection")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--model")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--first-output", required=True)
    prepare.add_argument("--resume", action="store_true")
    prepare.set_defaults(function=prepare_command)
    select = commands.add_parser("select")
    select.add_argument("--config", required=True)
    select.add_argument("--input", required=True)
    select.add_argument("--output", required=True)
    select.add_argument("--fraction", type=float)
    select.add_argument("--rank", type=int, default=64)
    select.add_argument("--ridge", type=float, default=0.1)
    select.add_argument("--noise", type=float, default=1.0)
    select.add_argument("--positive-only", action="store_true")
    select.set_defaults(function=select_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
