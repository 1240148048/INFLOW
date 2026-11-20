from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .chat_encoding import encode_messages
from .io import Example, load_config, read_jsonl, write_jsonl
from .modeling import hidden_size, language_backbone, lm_head, model_device
from .prompts import answer_only_messages
from .selection import load_model, option_distribution, option_token_ids, rademacher


METHOD_KEY = "scfg"
VERSION = "scfg_v1"


def metadata(example: Example) -> dict[str, Any]:
    return example.metadata.setdefault(METHOD_KEY, {})


def has_signature(example: Example) -> bool:
    info = example.metadata.get(METHOD_KEY, {})
    return info.get("version") == VERSION and bool(info.get("signature"))


def _hidden_states(output) -> tuple[torch.Tensor, ...]:
    states = getattr(output, "hidden_states", None)
    if states is None:
        raise RuntimeError(
            "The language backbone did not return hidden states. "
            "Check model compatibility with output_hidden_states=True."
        )
    return tuple(states)


@torch.inference_mode()
def forward_signature(
    model,
    tokenizer,
    example: Example,
    config: dict[str, Any],
    hidden_projection: torch.Tensor,
    signature_projection: torch.Tensor,
    num_layers: int,
    max_choices: int,
) -> tuple[np.ndarray, list[float], list[int]]:
    device = model_device(model)
    encoded = encode_messages(
        tokenizer,
        answer_only_messages(example),
        int(config["generation"]["max_prompt_tokens"]),
        device,
        enable_thinking=False,
    )
    output = language_backbone(model)(
        **encoded,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    states = _hidden_states(output)
    if len(states) < num_layers:
        raise ValueError(
            f"Requested {num_layers} hidden layers, model returned {len(states)}."
        )
    layer_indices = list(range(len(states) - num_layers, len(states)))
    rows = torch.stack(
        [states[index][0, -1].float() for index in layer_indices], dim=0
    )
    rows = F.normalize(rows, dim=1)
    projected_hidden = rows @ hidden_projection

    labels = example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]
    if not labels or len(labels) > max_choices:
        raise ValueError(
            f"{example.uid} has {len(labels)} choices; expected 1..{max_choices}."
        )
    probabilities = option_distribution(
        lm_head(model)(output.last_hidden_state[0, -1]).float(),
        option_token_ids(tokenizer, labels),
    )
    if example.rollout not in labels:
        raise ValueError(
            f"Standard answer {example.rollout!r} is invalid for {example.uid}."
        )
    residual = torch.zeros(max_choices, device=device, dtype=torch.float32)
    residual[: len(labels)] = probabilities
    residual[labels.index(example.rollout)] -= 1.0

    raw_signature = torch.einsum(
        "lr,c->lrc", projected_hidden, residual
    ).reshape(-1)
    signature = raw_signature @ signature_projection
    signature = signature.float().cpu().numpy().astype(np.float32)
    probability_values = probabilities.float().cpu().tolist()
    del output, states, rows, projected_hidden, probabilities, residual
    return signature, probability_values, layer_indices


@torch.inference_mode()
def extract_command(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("SCFG extraction requires one CUDA GPU.")
    config = load_config(args.config)
    model, tokenizer = load_model(config, args.model)
    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    if not examples:
        raise ValueError("Candidate file is empty.")
    max_choices = max(
        len(example.choice_labels or example.choices) for example in examples
    )
    if max_choices < 2:
        raise ValueError("SCFG currently requires multiple-choice candidates.")
    device = model_device(model)
    hidden_projection = rademacher(
        hidden_size(model), args.hidden_dim, config["seed"] + 701, device
    )
    raw_dim = args.num_layers * args.hidden_dim * max_choices
    signature_projection = rademacher(
        raw_dim, args.signature_dim, config["seed"] + 709, device
    )
    rows = []
    for example in tqdm(examples, desc="extracting forward-only SCFG signatures"):
        if has_signature(example) and len(metadata(example)["signature"]) == args.signature_dim:
            rows.append(example.to_dict())
            continue
        signature, probabilities, layer_indices = forward_signature(
            model,
            tokenizer,
            example,
            config,
            hidden_projection,
            signature_projection,
            args.num_layers,
            max_choices,
        )
        metadata(example).update(
            {
                "version": VERSION,
                "signature": signature.tolist(),
                "signature_norm": float(np.linalg.norm(signature)),
                "option_probabilities": probabilities,
                "standard_answer": example.rollout,
                "hidden_projection_dim": args.hidden_dim,
                "signature_dim": args.signature_dim,
                "layer_indices": layer_indices,
                "max_choices": max_choices,
                "uses_backward": False,
            }
        )
        rows.append(example.to_dict())
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} forward-only SCFG records to {args.output}")


def _scaled_signatures(examples: list[Example]) -> tuple[np.ndarray, float]:
    signatures = np.asarray(
        [example.metadata[METHOD_KEY]["signature"] for example in examples],
        dtype=np.float64,
    )
    if signatures.ndim != 2 or not np.isfinite(signatures).all():
        raise ValueError("Invalid SCFG signature matrix.")
    norms = np.linalg.norm(signatures, axis=1)
    scale = float(np.median(norms[norms > 1e-12])) if np.any(norms > 1e-12) else 1.0
    return signatures / max(scale, 1e-12), scale


def signed_conditional_fisher_order(
    examples: list[Example],
    ridge: float,
    noise: float,
    maximize: bool,
) -> tuple[list[int], dict[str, Any]]:
    x, scale = _scaled_signatures(examples)
    count, dimension = x.shape
    noise = max(float(noise), 1e-12)
    ridge = max(float(ridge), 1e-8)
    pool_gradient = x.mean(axis=0)
    pool_fisher = ridge * np.eye(dimension) + (x.T @ x) / (count * noise)
    pool_update = -np.linalg.solve(pool_fisher, pool_gradient)
    pool_norm = float(np.sqrt(max(pool_update @ pool_fisher @ pool_update, 0.0)))
    if pool_norm <= 1e-12:
        raise ValueError("The full-pool SCFG reference direction is numerically zero.")

    covariance = np.eye(dimension, dtype=np.float64) / ridge
    available = np.ones(count, dtype=bool)
    order: list[int] = []
    chosen_scores: list[float] = []
    chosen_information: list[float] = []
    chosen_alignment: list[float] = []
    for _ in range(count):
        candidates = np.flatnonzero(available)
        rows = x[candidates]
        impacts = -(rows @ covariance.T)
        leverage = np.einsum("ij,ij->i", rows @ covariance, rows) / noise
        information = 0.5 * np.log1p(np.maximum(leverage, 0.0))
        pool_impacts = impacts @ pool_fisher
        numerator = pool_impacts @ pool_update
        impact_norm = np.sqrt(
            np.maximum(np.einsum("ij,ij->i", pool_impacts, impacts), 0.0)
        )
        alignment = numerator / np.maximum(impact_norm * pool_norm, 1e-12)
        alignment = np.clip(alignment, -1.0, 1.0)
        scores = alignment * information
        local = int(np.argmax(scores) if maximize else np.argmin(scores))
        index = int(candidates[local])

        vector = x[index]
        cv = covariance @ vector
        denominator = noise + float(vector @ cv)
        covariance -= np.outer(cv, cv) / denominator
        covariance = 0.5 * (covariance + covariance.T)
        available[index] = False
        order.append(index)
        chosen_scores.append(float(scores[local]))
        chosen_information.append(float(information[local]))
        chosen_alignment.append(float(alignment[local]))

    diagnostics = {
        "version": VERSION,
        "rule": "signed_conditional_fisher_gain",
        "maximize": maximize,
        "signature_scale": scale,
        "pool_update_norm": pool_norm,
        "selected_scores": chosen_scores,
        "selected_information_gains": chosen_information,
        "selected_direction_alignments": chosen_alignment,
    }
    return order, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Forward-only SCFG utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model")
    extract.add_argument("--num-layers", type=int, default=8)
    extract.add_argument("--hidden-dim", type=int, default=32)
    extract.add_argument("--signature-dim", type=int, default=128)
    extract.set_defaults(function=extract_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
