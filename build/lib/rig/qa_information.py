from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .chat_encoding import encode_messages
from .io import Example, load_config, read_jsonl, write_jsonl
from .modeling import hidden_size, language_backbone, lm_head, model_device
from .prompts import answer_only_messages
from .selection import load_model, option_distribution, option_token_ids, rademacher


METHOD_KEY = "qair"
VERSION = "qair_uncertainty_retrieval_v4"
ORDER_VERSION = "uncertainty_directed_retrieval_v4"


def metadata(example: Example) -> dict[str, Any]:
    return example.metadata.setdefault(METHOD_KEY, {})


def has_features(example: Example) -> bool:
    info = example.metadata.get(METHOD_KEY, {})
    return (
        info.get("version") == VERSION
        and bool(info.get("gradient_feature"))
        and bool(info.get("sampled_gradient_features"))
        and bool(info.get("question_change_feature"))
        and float(info.get("gradient_noise", 0.0)) > 0.0
    )


def _question_change_signature(
    hidden_states: tuple[torch.Tensor, ...] | None,
    last_hidden_state: torch.Tensor,
    hidden_projection: torch.Tensor,
) -> torch.Tensor:
    """Represent how the prompt state evolves through depth, not its endpoint."""
    if hidden_states and len(hidden_states) >= 2:
        point_count = min(5, len(hidden_states))
        indices = torch.linspace(
            0, len(hidden_states) - 1, point_count, device="cpu"
        ).round().to(torch.long).unique(sorted=True)
        points = torch.stack(
            [hidden_states[int(index)][0, -1].float() for index in indices]
        )
    else:
        sequence = last_hidden_state[0].float()
        point_count = min(5, int(sequence.shape[0]))
        indices = torch.linspace(
            0, sequence.shape[0] - 1, point_count, device="cpu"
        ).round().to(torch.long).unique(sorted=True)
        points = sequence[indices.to(sequence.device)]
    projected = points @ hidden_projection
    if projected.shape[0] == 1:
        return projected.reshape(-1)
    changes = projected[1:] - projected[:-1]
    return changes.reshape(-1) / math.sqrt(changes.shape[0])


def _mode_answer(
    answers: list[str], labels: list[str], probabilities: torch.Tensor
) -> str:
    counts = Counter(answer for answer in answers if answer in labels)
    if not counts:
        return labels[int(torch.argmax(probabilities).item())]
    largest = max(counts.values())
    tied = [label for label in labels if counts.get(label, 0) == largest]
    return max(tied, key=lambda label: float(probabilities[labels.index(label)]))


def _answer_ids(tokenizer, answer: str, device: torch.device) -> torch.Tensor:
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"The standard answer {answer!r} has no tokens.")
    return torch.tensor(ids, dtype=torch.long, device=device)


@torch.inference_mode()
def _gradient_signature(
    backbone,
    head,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    hidden_projection: torch.Tensor,
    residual_projection: torch.Tensor,
) -> torch.Tensor:
    """Sketch the exact answer-token CE gradient with respect to the LM head."""
    device = prompt_ids.device
    prefix = target_ids[:-1]
    input_ids = torch.cat([prompt_ids, prefix], dim=0).unsqueeze(0)
    output = backbone(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        return_dict=True,
    )
    width = int(target_ids.numel())
    start = int(prompt_ids.numel()) - 1
    states = output.last_hidden_state[0, start : start + width]
    if states.shape[0] != width:
        raise RuntimeError("Could not align answer targets with hidden states.")

    logits = head(states).float()
    probabilities = torch.softmax(logits, dim=-1)
    # (p - e_y) R_v without materializing the V x H gradient matrix.
    projected_residual = probabilities @ residual_projection
    projected_residual -= residual_projection[target_ids]
    projected_hidden = states.float() @ hidden_projection
    sketch = torch.einsum(
        "tr,th->rh", projected_residual, projected_hidden
    ).reshape(-1)
    sketch /= math.sqrt(max(width, 1))
    del output, states, logits, probabilities, projected_residual, projected_hidden
    return sketch


@torch.inference_mode()
def extract_features(
    model,
    tokenizer,
    example: Example,
    config: dict[str, Any],
    hidden_projection: torch.Tensor,
    residual_projection: torch.Tensor,
    noise_floor: float,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, float, str, list[float], list[str]
]:
    device = model_device(model)
    labels = example.choice_labels or [
        chr(ord("A") + index) for index in range(len(example.choices))
    ]
    summaries = example.metadata.get("rig", {}).get("sample_summaries", [])
    answers = [str(item.get("answer")) for item in summaries]
    if not answers:
        raise ValueError(f"{example.uid} has no sampled answers in RIG metadata.")

    target_ids = {
        label: _answer_ids(tokenizer, label, device) for label in labels
    }
    max_answer_tokens = max(int(ids.numel()) for ids in target_ids.values())
    prompt_limit = max(
        1,
        int(config["generation"]["max_prompt_tokens"]) - max_answer_tokens + 1,
    )
    encoded = encode_messages(
        tokenizer,
        answer_only_messages(example),
        prompt_limit,
        device,
        enable_thinking=False,
    )
    prompt_ids = encoded["input_ids"][0]
    backbone = language_backbone(model)
    head = lm_head(model)

    prompt_output = backbone(
        input_ids=prompt_ids.unsqueeze(0),
        attention_mask=torch.ones(1, prompt_ids.numel(), dtype=torch.long, device=device),
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    question_change = _question_change_signature(
        getattr(prompt_output, "hidden_states", None),
        prompt_output.last_hidden_state,
        hidden_projection,
    )
    prompt_logits = head(prompt_output.last_hidden_state[0, -1]).float()
    choice_probabilities = option_distribution(
        prompt_logits, option_token_ids(tokenizer, labels)
    )
    standard = _mode_answer(answers, labels, choice_probabilities)

    signatures = {
        answer: _gradient_signature(
            backbone,
            head,
            prompt_ids,
            target_ids[answer],
            hidden_projection,
            residual_projection,
        )
        for answer in sorted(set(answers) | {standard})
        if answer in target_ids
    }
    standard_signature = signatures[standard]
    sampled = torch.stack(
        [signatures.get(answer, standard_signature) for answer in answers]
    )
    center = sampled.mean(dim=0)
    # Mean coordinate-wise variance of five forward-gradient directions.
    noise = float(
        ((sampled - center) ** 2).mean().clamp_min(noise_floor).item()
    )
    probabilities = choice_probabilities.float().cpu().tolist()
    sampled_answers = [answer if answer in labels else standard for answer in answers]
    sampled_array = sampled.float().cpu().numpy().astype(np.float32)
    del prompt_output, prompt_logits, choice_probabilities, sampled, center
    return (
        standard_signature.float().cpu().numpy().astype(np.float32),
        sampled_array,
        question_change.float().cpu().numpy().astype(np.float32),
        noise,
        standard,
        probabilities,
        sampled_answers,
    )


@torch.inference_mode()
def extract_command(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Forward-gradient extraction requires one CUDA GPU.")
    config = load_config(args.config)
    model, tokenizer = load_model(config, args.model)
    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    if not examples:
        raise ValueError("Candidate file is empty.")
    device = model_device(model)
    head = lm_head(model)
    vocab_size = int(head.weight.shape[0])
    hidden_projection = rademacher(
        hidden_size(model), args.hidden_dim, config["seed"] + 811, device
    )
    residual_projection = rademacher(
        vocab_size, args.residual_dim, config["seed"] + 821, device
    )
    rows = []
    for example in tqdm(
        examples, desc="extracting forward-gradient information features"
    ):
        if has_features(example):
            rows.append(example.to_dict())
            continue
        (
            gradient,
            sampled_gradients,
            question_change,
            noise,
            standard,
            probabilities,
            sampled_answers,
        ) = extract_features(
            model,
            tokenizer,
            example,
            config,
            hidden_projection,
            residual_projection,
            args.noise_floor,
        )
        example.rollout = standard
        rig = example.metadata.setdefault("rig", {})
        summaries = rig.get("sample_summaries", [])
        matching = [
            index
            for index, item in enumerate(summaries)
            if item.get("answer") == standard
        ]
        if matching:
            best = min(
                matching,
                key=lambda index: float(
                    summaries[index].get("predictive_entropy", float("inf"))
                ),
            )
            rig["standard_sample_index"] = best
            rig["standard_predictive_entropy"] = float(
                summaries[best].get("predictive_entropy", float("inf"))
            )
        rig["standard_answer"] = standard
        rig["training_variant"] = "five_sample_mode"
        metadata(example).update(
            {
                "version": VERSION,
                "gradient_feature": gradient.tolist(),
                "sampled_gradient_features": sampled_gradients.tolist(),
                "question_change_feature": question_change.tolist(),
                "gradient_noise": noise,
                "sampled_answers": sampled_answers,
                "standard_answer": standard,
                "option_probabilities": probabilities,
                "hidden_projection_dim": args.hidden_dim,
                "residual_projection_dim": args.residual_dim,
                "gradient_dim": args.hidden_dim * args.residual_dim,
                "gradient_target": "full_vocab_answer_token_cross_entropy",
                "similarity_target": "standardized_prompt_layerwise_state_changes",
                "noise_floor": args.noise_floor,
                "uses_backward": False,
            }
        )
        rows.append(example.to_dict())
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} forward-gradient records to {args.output}")


def question_answer_information_order(
    examples: list[Example], ridge: float, maximize: bool
) -> tuple[list[int], dict[str, Any]]:
    gradients = np.asarray(
        [example.metadata[METHOD_KEY]["gradient_feature"] for example in examples],
        dtype=np.float64,
    )
    question_features = np.asarray(
        [
            example.metadata[METHOD_KEY]["question_change_feature"]
            for example in examples
        ],
        dtype=np.float64,
    )
    if gradients.ndim != 2 or not np.isfinite(gradients).all():
        raise ValueError("Invalid forward-gradient feature matrix.")
    sampled_gradients = np.asarray(
        [
            example.metadata[METHOD_KEY]["sampled_gradient_features"]
            for example in examples
        ],
        dtype=np.float64,
    )
    if (
        sampled_gradients.ndim != 3
        or sampled_gradients.shape[0] != gradients.shape[0]
        or sampled_gradients.shape[2] != gradients.shape[1]
        or not np.isfinite(sampled_gradients).all()
    ):
        raise ValueError("Invalid sampled forward-gradient feature tensor.")
    if question_features.ndim != 2 or not np.isfinite(question_features).all():
        raise ValueError("Invalid question-change feature matrix.")

    norms = np.linalg.norm(gradients, axis=1)
    positive = norms[norms > 1e-12]
    gradient_scale = float(np.median(positive)) if positive.size else 1.0
    gradients = gradients / max(gradient_scale, 1e-12)
    sampled_gradients = sampled_gradients / max(gradient_scale, 1e-12)
    count, dimension = gradients.shape

    # A rollout covariance is a Gaussian representation-entropy surrogate.
    # The determinant lemma keeps the calculation in the K x K sample space.
    centered_rollouts = sampled_gradients - sampled_gradients.mean(
        axis=1, keepdims=True
    )
    sample_count = sampled_gradients.shape[1]
    active_rank = max(min(sample_count - 1, dimension), 1)
    representation_entropy = np.empty(count, dtype=np.float64)
    for index in range(count):
        gram = (
            centered_rollouts[index] @ centered_rollouts[index].T
        ) / max(sample_count - 1, 1)
        sign, logdet = np.linalg.slogdet(np.eye(sample_count) + gram)
        if sign <= 0 or not np.isfinite(logdet):
            raise ValueError("Invalid rollout representation covariance.")
        representation_entropy[index] = 0.5 * logdet / active_rank
    reliability = np.exp(-representation_entropy)
    target_need = 1.0 - reliability
    teaching_noise = np.maximum(np.expm1(representation_entropy), 1e-10)

    # Standardize prompt trajectories before constructing a PSD RBF kernel.
    # Full covariance whitening can make small, high-dimensional sets nearly
    # equidistant and erase the local similarity that retrieval needs.
    centered_questions = question_features - question_features.mean(
        axis=0, keepdims=True
    )
    coordinate_scale = np.sqrt(
        np.mean(np.square(centered_questions), axis=0)
    )
    threshold = max(float(np.max(coordinate_scale)) * 1e-8, 1e-12)
    keep = coordinate_scale > threshold
    if np.any(keep):
        whitened = centered_questions[:, keep] / coordinate_scale[keep]
    else:
        whitened = centered_questions
    squared_norms = np.einsum("ij,ij->i", whitened, whitened)
    squared_distances = np.maximum(
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (whitened @ whitened.T),
        0.0,
    )
    positive_distances = squared_distances[
        np.triu_indices(count, k=1)
    ]
    positive_distances = positive_distances[positive_distances > 1e-12]
    distance_scale = float(np.median(positive_distances)) if positive_distances.size else 1.0
    transfer_kernel = np.exp(
        -squared_distances / max(2.0 * distance_scale, 1e-12)
    )
    transfer_kernel += np.eye(count) * max(float(ridge), 1e-8)

    # The GP posterior is updated after every selected teaching example. A
    # candidate is scored only by information transferred to other uncertain
    # questions, so an uncertain question cannot rank highly by explaining itself.
    posterior = transfer_kernel.copy()
    posterior = 0.5 * (posterior + posterior.T)

    available = np.ones(count, dtype=bool)
    order: list[int] = []
    chosen_gains: list[float] = []
    chosen_reliability: list[float] = []
    chosen_target_need: list[float] = []
    chosen_teaching_noise: list[float] = []
    chosen_mean_target_information: list[float] = []

    for _ in range(count):
        candidates = np.flatnonzero(available)
        latent_variance = np.maximum(np.diag(posterior), 1e-12)
        source_variance = latent_variance + teaching_noise
        squared_correlation = np.square(posterior) / np.maximum(
            source_variance[:, None] * latent_variance[None, :], 1e-12
        )
        squared_correlation = np.clip(
            squared_correlation, 0.0, 1.0 - 1e-12
        )
        pairwise_information = -0.5 * np.log1p(-squared_correlation)
        np.fill_diagonal(pairwise_information, 0.0)
        # Source reliability already enters as heteroscedastic observation
        # noise; target need is the only explicit information weight.
        all_gains = pairwise_information @ target_need
        gains = all_gains[candidates]
        local = int(np.argmax(gains) if maximize else np.argmin(gains))
        index = int(candidates[local])

        posterior_column = posterior[:, index].copy()
        posterior_denominator = teaching_noise[index] + posterior[index, index]
        posterior -= np.outer(posterior_column, posterior_column) / max(
            posterior_denominator, 1e-12
        )
        posterior = 0.5 * (posterior + posterior.T)
        available[index] = False
        order.append(index)
        chosen_gains.append(float(gains[local]))
        chosen_reliability.append(float(reliability[index]))
        chosen_target_need.append(float(target_need[index]))
        chosen_teaching_noise.append(float(teaching_noise[index]))
        chosen_mean_target_information.append(
            float(
                np.sum(pairwise_information[index] * target_need)
                / max(np.sum(target_need), 1e-12)
            )
        )

    return order, {
        "version": ORDER_VERSION,
        "rule": "uncertainty_directed_pairwise_information_retrieval",
        "maximize": maximize,
        "ridge": float(ridge),
        "gradient_scale": gradient_scale,
        "rbf_median_squared_distance": distance_scale,
        "mean_representation_entropy": float(representation_entropy.mean()),
        "mean_teaching_reliability": float(reliability.mean()),
        "mean_target_need": float(target_need.mean()),
        "selected_gains": chosen_gains,
        "selected_reliability": chosen_reliability,
        "selected_target_need": chosen_target_need,
        "selected_teaching_noise": chosen_teaching_noise,
        "selected_mean_target_information": chosen_mean_target_information,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forward-only gradient information features"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--config", required=True)
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--model")
    extract.add_argument("--hidden-dim", type=int, default=16)
    extract.add_argument("--residual-dim", type=int, default=8)
    extract.add_argument("--noise-floor", type=float, default=1e-6)
    extract.set_defaults(function=extract_command)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
