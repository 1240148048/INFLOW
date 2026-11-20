from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import set_seed

from .chat_encoding import encode_messages
from .io import Example, load_config, read_jsonl, write_jsonl
from .modeling import load_causal_model, load_tokenizer, torch_dtype
from .prompts import answer_only_messages, extract_choice


def choice_constraint(tokenizer, labels: list[str], prompt_width: int):
    eos = tokenizer.eos_token_id
    if isinstance(eos, list):
        eos = eos[0]
    if eos is None:
        raise ValueError("Constrained evaluation requires an EOS token.")
    sequences = [
        tokenizer(label, add_special_tokens=False)["input_ids"]
        + [eos]
        for label in labels
    ]

    def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
        prefix = input_ids[prompt_width:].tolist()
        allowed = {
            sequence[len(prefix)]
            for sequence in sequences
            if len(prefix) < len(sequence)
            and sequence[: len(prefix)] == prefix
        }
        return sorted(allowed) if allowed else [eos]

    return allowed_tokens


def majority_vote(predictions: list[str | None]) -> tuple[str | None, float]:
    valid = [value for value in predictions if value is not None]
    if not valid:
        return None, 0.0
    winner, count = sorted(
        Counter(valid).items(), key=lambda item: (-item[1], item[0])
    )[0]
    return winner, count / len(predictions)


def ece(confidence: list[float], correct: list[bool], bins: int) -> float:
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (conf >= edges[index]) & (conf <= edges[index + 1])
        else:
            mask = (conf >= edges[index]) & (conf < edges[index + 1])
        if mask.any():
            total += mask.mean() * abs(corr[mask].mean() - conf[mask].mean())
    return float(total)


def summarize(
    rows: list[dict[str, Any]], bins: int, num_rollouts: int
) -> dict[str, Any]:
    rollout_correct = [
        prediction == row["answer"]
        for row in rows
        for prediction in row["predictions"]
    ]
    majority_correct = [
        row["majority_prediction"] == row["answer"] for row in rows
    ]
    parsed_predictions = [
        prediction
        for row in rows
        for prediction in row["predictions"]
        if prediction is not None
    ]
    parsed_correct = [
        prediction == row["answer"]
        for row in rows
        for prediction in row["predictions"]
        if prediction is not None
    ]
    covered_rows = [
        row for row in rows if row["majority_prediction"] is not None
    ]
    covered_correct = [
        row["majority_prediction"] == row["answer"] for row in covered_rows
    ]
    total_rollouts = len(rows) * num_rollouts
    output_lengths = [
        length for row in rows for length in row["completion_lengths"]
    ]
    rollout_index_accuracies = [
        float(
            np.mean(
                [row["predictions"][index] == row["answer"] for row in rows]
            )
        )
        for index in range(num_rollouts)
    ] if rows else [0.0] * num_rollouts
    return {
        "count": len(rows),
        "total_output_tokens": int(sum(output_lengths)),
        "mean_output_tokens": (
            float(np.mean(output_lengths)) if output_lengths else 0.0
        ),
        "median_output_tokens": (
            float(np.median(output_lengths)) if output_lengths else 0.0
        ),
        "max_output_tokens": int(max(output_lengths, default=0)),
        "parse_rate": (
            len(parsed_predictions) / total_rollouts if total_rollouts else 0.0
        ),
        "completion_rate": (
            float(
                np.mean(
                    [
                        finished
                        for row in rows
                        for finished in row["finished"]
                    ]
                )
            )
            if rows
            else 0.0
        ),
        "conditional_avg_accuracy": (
            float(np.mean(parsed_correct)) if parsed_correct else 0.0
        ),
        "majority_coverage": (
            len(covered_rows) / len(rows) if rows else 0.0
        ),
        "conditional_majority_accuracy": (
            float(np.mean(covered_correct)) if covered_correct else 0.0
        ),
        f"avg_at_{num_rollouts}": (
            float(np.mean(rollout_correct)) if rows else 0.0
        ),
        f"maj_at_{num_rollouts}": (
            float(np.mean(majority_correct)) if rows else 0.0
        ),
        "rollout_index_accuracies": rollout_index_accuracies,
        "rollout_accuracy_mean": float(np.mean(rollout_index_accuracies)),
        "rollout_accuracy_std": (
            float(np.std(rollout_index_accuracies, ddof=1))
            if num_rollouts > 1
            else 0.0
        ),
        "ece": ece(
            [row["confidence"] for row in rows],
            majority_correct,
            bins,
        )
        if rows
        else 0.0,
        "ece_on_covered": (
            ece(
                [row["confidence"] for row in covered_rows],
                covered_correct,
                bins,
            )
            if covered_rows
            else 0.0
        ),
    }


def resolve_adapter_path(path: str) -> Path:
    source = Path(path)
    nested = source / "student"
    return nested if (nested / "adapter_config.json").exists() else source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-domain-limit", type=int)
    parser.add_argument("--num-rollouts", type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This evaluator requires one CUDA GPU.")
    config = load_config(args.config)
    evaluation = config["evaluation"]
    num_rollouts = args.num_rollouts or evaluation["num_rollouts"]
    set_seed(config["seed"])
    model_name = args.model or config["model"]["name"]
    tokenizer = load_tokenizer(
        model_name, config["model"]["trust_remote_code"]
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_model(
        model_name,
        torch_dtype(config["model"]["dtype"]),
        config["model"]["trust_remote_code"],
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, resolve_adapter_path(args.adapter), is_trainable=False
        )
    model = model.to("cuda:0").eval()
    examples = []
    domain_counts: defaultdict[str, int] = defaultdict(int)
    for row in read_jsonl(args.data):
        if row["split"] != "test" or row.get("answer") is None:
            continue
        domain = row["domain"]
        if args.domain and domain.lower() != args.domain.lower():
            continue
        if (
            args.per_domain_limit
            and domain_counts[domain] >= args.per_domain_limit
        ):
            continue
        examples.append(Example.from_dict(row))
        domain_counts[domain] += 1
    if args.limit:
        examples = examples[: args.limit]

    rows = []
    for index, example in enumerate(examples, start=1):
        encoded = encode_messages(
            tokenizer,
            answer_only_messages(example),
            config["generation"]["max_prompt_tokens"],
            torch.device("cuda:0"),
            enable_thinking=False,
        )
        labels = example.choice_labels or [
            chr(ord("A") + choice) for choice in range(len(example.choices))
        ]
        constraint = choice_constraint(
            tokenizer, labels, encoded["input_ids"].shape[1]
        )
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=True,
                num_return_sequences=num_rollouts,
                max_new_tokens=evaluation["max_new_tokens"],
                temperature=evaluation["temperature"],
                top_p=evaluation["top_p"],
                top_k=evaluation.get("top_k", 20),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                prefix_allowed_tokens_fn=constraint,
            )
        texts = [
            tokenizer.decode(
                candidate[encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for candidate in output
        ]
        generated_tokens = output[:, encoded["input_ids"].shape[1] :]
        eos_ids = tokenizer.eos_token_id
        eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
        eos_ids = {token for token in eos_ids if token is not None}
        finished = [
            any(int(token) in eos_ids for token in candidate.tolist())
            for candidate in generated_tokens
        ]
        completion_lengths = []
        for candidate in generated_tokens:
            tokens = candidate.tolist()
            stop = next(
                (
                    position + 1
                    for position, token in enumerate(tokens)
                    if int(token) in eos_ids
                ),
                len(tokens),
            )
            completion_lengths.append(stop)
        predictions = [extract_choice(text, labels) for text in texts]
        majority, confidence = majority_vote(predictions)
        rows.append(
            {
                "uid": example.uid,
                "dataset": example.dataset,
                "domain": example.domain,
                "answer": example.answer,
                "predictions": predictions,
                "majority_prediction": majority,
                "confidence": confidence,
                "finished": finished,
                "completion_lengths": completion_lengths,
                "texts": texts,
            }
        )
        print(f"evaluated {index}/{len(examples)}", end="\r", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl.gz", rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[row["domain"]].append(row)
    metrics = {
        name: summarize(group, evaluation["ece_bins"], num_rollouts)
        for name, group in sorted(groups.items())
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
