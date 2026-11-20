from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed

from .io import Example, load_config, read_jsonl, write_jsonl
from .modeling import load_causal_model, load_tokenizer, torch_dtype
from .opsd_common import decoded_answer, generate_reasoning, labels_for


def ece(confidence, correctness, bins=15):
    confidence = np.asarray(confidence, dtype=np.float64)
    correctness = np.asarray(correctness, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (confidence >= edges[index]) & (
            confidence <= edges[index + 1]
            if index == bins - 1
            else confidence < edges[index + 1]
        )
        if mask.any():
            result += mask.mean() * abs(confidence[mask].mean() - correctness[mask].mean())
    return float(result)


def summarize(rows, rollouts):
    rollout_accuracies = [
        float(np.mean([row["predictions"][index] == row["answer"] for row in rows]))
        for index in range(rollouts)
    ]
    average = float(np.mean(rollout_accuracies))
    majority_correct = [row["majority"] == row["answer"] for row in rows]
    return {
        "count": len(rows),
        f"avg_at_{rollouts}": float(average),
        f"maj_at_{rollouts}": float(np.mean(majority_correct)),
        "rollout_index_accuracies": rollout_accuracies,
        "rollout_accuracy_std": float(np.std(rollout_accuracies)),
        "ece": ece([row["confidence"] for row in rows], majority_correct),
        "parse_rate": float(
            np.mean([value is not None for row in rows for value in row["predictions"]])
        ),
        "mean_output_tokens": float(
            np.mean([value for row in rows for value in row["lengths"]])
        ),
    }


def resolve_adapter(path: str):
    root = Path(path)
    nested = root / "student"
    return nested if (nested / "adapter_config.json").exists() else root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    evaluation = config["evaluation"]
    set_seed(int(config["seed"]))
    name = args.model or config["model"]["name"]
    tokenizer = load_tokenizer(name, config["model"]["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_model(
        name,
        torch_dtype(config["model"]["dtype"]),
        config["model"]["trust_remote_code"],
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, resolve_adapter(args.adapter))
    model = model.to("cuda:0").eval()
    examples = [
        Example.from_dict(row)
        for row in read_jsonl(args.data)
        if row["split"] == "test" and row.get("answer") is not None
    ]
    if args.limit:
        examples = examples[: args.limit]
    rollouts = int(evaluation["num_rollouts"])
    rows = []
    for index, example in enumerate(examples, start=1):
        labels = labels_for(example)
        _prompt, responses = generate_reasoning(
            model,
            tokenizer,
            example,
            (),
            max_prompt_tokens=int(evaluation["max_prompt_tokens"]),
            max_new_tokens=int(evaluation["max_new_tokens"]),
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            num_return_sequences=rollouts,
        )
        predictions = [decoded_answer(tokenizer, response, example) for response in responses]
        valid = [prediction for prediction in predictions if prediction is not None]
        if valid:
            counts = Counter(valid)
            largest = max(counts.values())
            majority = next(label for label in labels if counts.get(label, 0) == largest)
            confidence = largest / rollouts
        else:
            majority, confidence = None, 0.0
        rows.append(
            {
                "uid": example.uid,
                "domain": example.domain,
                "answer": example.answer,
                "predictions": predictions,
                "majority": majority,
                "confidence": confidence,
                "lengths": [len(response) for response in responses],
            }
        )
        print(f"evaluated {index}/{len(examples)}", end="\r", flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "predictions.jsonl.gz", rows)
    groups = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[row["domain"]].append(row)
    metrics = {name: summarize(group, rollouts) for name, group in groups.items()}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
