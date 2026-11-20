from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import set_seed

from .io import Example, append_jsonl, load_config, read_jsonl
from .modeling import (
    hidden_size,
    load_causal_model,
    load_tokenizer,
    model_device,
    torch_dtype,
)
from .opsd_common import generate_reasoning, labels_for, modal_answer
from .opsd_prepare import (
    analyze_response,
    choose_demonstration,
    completion_marker,
    rademacher,
)


KEY = "neuron_opsd"
TEACHER_KEY = "opsd_teacher_generation"


def distribution_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities /= max(float(probabilities.sum()), 1e-12)
    return float(
        -(probabilities * np.log(np.maximum(probabilities, 1e-12))).sum()
    )


def jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left /= max(float(left.sum()), 1e-12)
    right /= max(float(right.sum()), 1e-12)
    middle = 0.5 * (left + right)
    left_kl = np.sum(left * np.log(np.maximum(left, 1e-12) / np.maximum(middle, 1e-12)))
    right_kl = np.sum(right * np.log(np.maximum(right, 1e-12) / np.maximum(middle, 1e-12)))
    return float(0.5 * (left_kl + right_kl))


@torch.inference_mode()
def prepare_one(
    model,
    tokenizer,
    example: Example,
    demonstrations: list[Example],
    projection: torch.Tensor,
    config: dict,
    method: str,
) -> Example:
    generation = config["teacher_generation"]
    prompt_ids, responses = generate_reasoning(
        model,
        tokenizer,
        example,
        demonstrations,
        max_prompt_tokens=int(generation["max_prompt_tokens"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        num_return_sequences=int(generation["num_samples"]),
    )
    samples = []
    for response in responses:
        sample = analyze_response(
            model,
            tokenizer,
            example,
            prompt_ids,
            response,
            projection,
            method == "neuron",
            config,
        )
        sample["activation_count"] = len(sample["neuron_ids"])
        sample.pop("neuron_ids", None)
        samples.append(sample)

    labels = labels_for(example)
    modal, agreement = modal_answer(samples, labels)
    best_index = choose_demonstration(samples, modal)
    probability_matrix = np.asarray(
        [
            [sample["choice_probabilities"][label] for label in labels]
            for sample in samples
        ],
        dtype=np.float64,
    )
    mean_probability = probability_matrix.mean(axis=0)
    mean_probability /= max(float(mean_probability.sum()), 1e-12)
    teacher_entropy = distribution_entropy(mean_probability)

    initial = example.metadata[KEY]["samples"][0]
    initial_entropy = float(initial["choice_entropy"])
    actual_gain = initial_entropy - teacher_entropy
    initial_probability = np.asarray(
        [initial["choice_probabilities"][label] for label in labels],
        dtype=np.float64,
    )
    distribution_shift = jensen_shannon(initial_probability, mean_probability)
    mean_activation = float(
        np.mean([sample["activation_count"] for sample in samples])
    )
    best_last_layer = samples[best_index]["layer_embeddings"][-1]
    edge_information = np.asarray(
        example.metadata["opsd_retrieval"]["context_scores_in_prompt_order"],
        dtype=np.float64,
    )
    # For independent noisy observations of one latent decision state,
    # exp(2 I_j)-1 is the per-source signal-to-noise ratio and SNRs add.
    theoretical_information = 0.5 * np.log1p(
        np.expm1(2.0 * np.maximum(edge_information, 0.0)).sum()
    )

    # Keep selection records compact after retaining the teacher representation
    # needed by the redundancy baseline.
    for sample in samples:
        sample.pop("layer_embeddings", None)

    result = copy.deepcopy(example)
    result.rollout = samples[best_index]["text"]
    result.metadata[TEACHER_KEY] = {
        "method": method,
        "samples": samples,
        "modal_answer": modal,
        "vote_agreement": agreement,
        "best_sample_index": best_index,
        "best_answer": samples[best_index]["answer"],
        "mean_choice_probabilities": dict(zip(labels, mean_probability.tolist())),
        "initial_choice_entropy": initial_entropy,
        "teacher_mean_choice_entropy": teacher_entropy,
        "actual_information_gain": actual_gain,
        "distribution_shift_js": distribution_shift,
        "theoretical_retrieval_information": float(theoretical_information),
        "mean_activation_count": mean_activation,
        "best_last_layer_embedding": best_last_layer,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--retrieved", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method",
        choices=(
            "full",
            "redundancy",
            "consistency",
            "neuron",
            "ours",
            "mutual",
            "no_certainty",
            "no_similarity",
        ),
        required=True,
    )
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Teacher-data preparation requires CUDA.")
    config = load_config(args.config)
    generation = config["teacher_generation"]
    if int(generation["num_samples"]) != 5:
        raise ValueError("Teacher-data preparation requires exactly five samples.")

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

    candidates = {
        example.uid: example
        for example in (Example.from_dict(row) for row in read_jsonl(args.candidates))
    }
    examples = [Example.from_dict(row) for row in read_jsonl(args.retrieved)]
    output = Path(args.output)
    completed = (
        {row["uid"] for row in read_jsonl(output)}
        if args.resume and output.exists()
        else set()
    )
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} exists; pass --resume to reuse it.")

    method_offset = {
        "full": 2_000_000,
        "redundancy": 3_000_000,
        "consistency": 4_000_000,
        "neuron": 5_000_000,
        "ours": 0,
        "mutual": 1_000_000,
        "no_certainty": 6_000_000,
        "no_similarity": 7_000_000,
    }[args.method]
    pending = [
        (index, example)
        for index, example in enumerate(examples)
        if example.uid not in completed
    ]
    progress = tqdm(
        pending,
        total=len(examples),
        initial=len(examples) - len(pending),
        desc=f"generating {args.method} teacher data",
        unit="example",
        dynamic_ncols=True,
    )
    for index, example in progress:
        missing = set(example.demo_ids) - set(candidates)
        if missing:
            raise KeyError(f"Missing demonstrations: {sorted(missing)}")
        demonstrations = [copy.deepcopy(candidates[uid]) for uid in example.demo_ids]
        set_seed(int(config["seed"]) + method_offset + index)
        prepared = prepare_one(
            model,
            tokenizer,
            example,
            demonstrations,
            projection,
            config,
            args.method,
        )
        append_jsonl(output, prepared.to_dict())
        gc.collect()
        torch.cuda.empty_cache()
    completion_marker(output).write_text(str(len(examples)), encoding="utf-8")
    print(f"Prepared {len(examples)} {args.method} teacher-data records at {output}")


if __name__ == "__main__":
    main()
