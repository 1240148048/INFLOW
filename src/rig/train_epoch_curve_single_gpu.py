from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from .chat_encoding import encode_messages
from .context_selection import vote_answer
from .io import Example, load_config, read_jsonl
from .model_inputs import distillation_inputs_from_prompt_ids
from .modeling import load_causal_model, load_tokenizer, torch_dtype
from .prompts import answer_only_messages, training_messages
from .selection import option_distribution, option_token_ids
from .train_single_gpu import response_logits


def prompt_ids(tokenizer, messages, limit: int) -> list[int]:
    encoded = encode_messages(
        tokenizer, messages, limit, torch.device("cpu"), enable_thinking=False
    )
    return encoded["input_ids"][0].tolist()


@torch.inference_mode()
def evaluate_model(
    model,
    tokenizer,
    examples: list[Example],
    prompts: list[list[int]],
    config: dict[str, Any],
    num_rollouts: int,
) -> tuple[dict[str, Any], float]:
    del num_rollouts
    device = torch.device("cuda:0")
    model.eval()
    old_cache = model.config.use_cache
    model.config.use_cache = False
    losses: list[float] = []
    correct: list[float] = []
    started = time.perf_counter()
    for example, ids in zip(examples, prompts):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        encoded = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        labels = example.choice_labels or [
            chr(ord("A") + index) for index in range(len(example.choices))
        ]
        if example.answer not in labels:
            raise ValueError(
                f"Gold answer {example.answer!r} is not in labels for {example.uid}."
            )
        with torch.autocast(
            "cuda", dtype=torch_dtype(config["model"]["dtype"])
        ):
            logits = response_logits(model, encoded, width=1)[0, -1]
        probabilities = option_distribution(
            logits.float(), option_token_ids(tokenizer, labels)
        )
        gold_index = labels.index(example.answer)
        losses.append(
            float(-torch.log(probabilities[gold_index].clamp_min(1e-12)).item())
        )
        correct.append(float(int(torch.argmax(probabilities).item()) == gold_index))
    elapsed = time.perf_counter() - started
    model.config.use_cache = old_cache
    model.train()
    values = torch.tensor(losses, dtype=torch.float64)
    accuracy_values = torch.tensor(correct, dtype=torch.float64)
    standard_error = (
        float(values.std(unbiased=True).item() / math.sqrt(len(losses)))
        if len(losses) > 1
        else 0.0
    )
    return {
        "test_log_loss": float(values.mean().item()),
        "test_log_loss_standard_error": standard_error,
        "test_count": len(losses),
        "test_accuracy": float(accuracy_values.mean().item()),
        "test_accuracy_standard_error": (
            float(
                accuracy_values.std(unbiased=True).item()
                / math.sqrt(len(correct))
            )
            if len(correct) > 1
            else 0.0
        ),
    }, elapsed


def write_progress(output: Path, records: list[dict[str, Any]]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "epoch_metrics.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    with (output / "epoch_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            row = dict(record)
            if "rollout_index_accuracies" in row:
                row["rollout_index_accuracies"] = json.dumps(
                    row["rollout_index_accuracies"]
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one fixed subset and evaluate after every epoch."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--selected", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num-rollouts", type=int, default=3)
    parser.add_argument("--random-sample-count", type=int, default=0)
    parser.add_argument("--candidate-pool")
    parser.add_argument("--context-distillation", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Optimizer batch size; zero preserves full-batch training.",
    )
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--eval-every-updates", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This entry point requires one CUDA GPU.")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive.")
    if args.random_sample_count < 0:
        raise ValueError("--random-sample-count cannot be negative.")
    if args.batch_size < 0:
        raise ValueError("--batch-size cannot be negative.")
    if args.max_updates < 0 or args.eval_every_updates < 0:
        raise ValueError("Update counts cannot be negative.")
    if args.max_updates and not args.eval_every_updates:
        raise ValueError("--max-updates requires --eval-every-updates.")

    config = load_config(args.config)
    train_config = config["training"]
    set_seed(config["seed"])
    device = torch.device("cuda:0")
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
    ).to(device)
    lora = LoraConfig(
        r=train_config["lora_rank"],
        lora_alpha=train_config["lora_alpha"],
        lora_dropout=train_config["lora_dropout"],
        target_modules=train_config["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if train_config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False

    selected = [Example.from_dict(row) for row in read_jsonl(args.selected)]
    tests = [
        Example.from_dict(row)
        for row in read_jsonl(args.data)
        if row["split"] == "test" and row.get("answer") is not None
    ]
    if not selected or not tests:
        raise ValueError("Training subset and test set must both be non-empty.")

    candidate_by_uid: dict[str, Example] = {}
    if args.context_distillation:
        if not args.candidate_pool:
            raise ValueError("--context-distillation requires --candidate-pool.")
        candidate_by_uid = {
            example.uid: example
            for example in (
                Example.from_dict(row) for row in read_jsonl(args.candidate_pool)
            )
        }
        for example in candidate_by_uid.values():
            example.rollout = vote_answer(example)
        missing = {
            uid
            for example in selected
            for uid in example.demo_ids
            if uid not in candidate_by_uid
        }
        if missing:
            raise ValueError(f"Missing retrieved demonstrations: {sorted(missing)[:5]}")

    train_prompts = [
        prompt_ids(
            tokenizer,
            (
                answer_only_messages(example)
                if args.context_distillation
                else training_messages(example)
            ),
            train_config["max_sequence_tokens"],
        )
        for example in selected
    ]
    teacher_prompts = (
        [
            prompt_ids(
                tokenizer,
                training_messages(
                    example,
                    [candidate_by_uid[uid] for uid in example.demo_ids],
                ),
                train_config["max_sequence_tokens"],
            )
            for example in selected
        ]
        if args.context_distillation
        else []
    )
    test_prompts = [
        prompt_ids(
            tokenizer,
            answer_only_messages(example),
            config["generation"]["max_prompt_tokens"],
        )
        for example in tests
    ]
    responses = []
    eos = tokenizer.eos_token_id
    eos = eos[0] if isinstance(eos, list) else eos
    for example in selected:
        response = tokenizer(
            example.rollout or "", add_special_tokens=False
        )["input_ids"][: train_config["rollout_max_new_tokens"]]
        if not response:
            raise ValueError(f"Empty target for {example.uid}.")
        if eos is not None and response[-1] != eos:
            response.append(eos)
        responses.append(response)
    if args.random_sample_count > len(selected):
        raise ValueError(
            "--random-sample-count cannot exceed the candidate pool size."
        )

    epoch_example_count = args.random_sample_count or len(selected)
    batch_size = args.batch_size or epoch_example_count
    batch_size = min(batch_size, epoch_example_count)
    updates_per_epoch = math.ceil(epoch_example_count / batch_size)
    total_updates = args.max_updates or args.epochs * updates_per_epoch
    planned_epochs = (
        math.ceil(total_updates * batch_size / epoch_example_count)
        if args.max_updates
        else args.epochs
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=train_config["learning_rate"], weight_decay=0.0
    )
    scheduler = get_scheduler(
        train_config["scheduler"],
        optimizer,
        num_warmup_steps=int(total_updates * train_config["warmup_ratio"]),
        num_training_steps=total_updates,
    )
    rng = random.Random(config["seed"])
    records = []
    seen_examples: set[int] = set()
    cumulative_training = 0.0
    cumulative_optimizer_updates = 0
    output = Path(args.output_dir)

    teacher_choice_probabilities: list[torch.Tensor] = []
    if args.context_distillation:
        model.eval()
        with model.disable_adapter(), torch.inference_mode():
            for example, ids in tqdm(
                zip(selected, teacher_prompts),
                total=len(selected),
                desc="caching retrieved-context teacher distributions",
            ):
                input_ids = torch.tensor([ids], dtype=torch.long, device=device)
                encoded = {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }
                labels = example.choice_labels or [
                    chr(ord("A") + index) for index in range(len(example.choices))
                ]
                with torch.autocast(
                    "cuda", dtype=torch_dtype(config["model"]["dtype"])
                ):
                    logits = response_logits(model, encoded, width=1)[0, -1]
                teacher_choice_probabilities.append(
                    option_distribution(
                        logits.float(), option_token_ids(tokenizer, labels)
                    ).cpu()
                )
        model.train()
    original_metrics, original_evaluation_seconds = evaluate_model(
        model,
        tokenizer,
        tests,
        test_prompts,
        config,
        args.num_rollouts,
    )

    def append_evaluation(
        epoch: int,
        order: list[int],
        mean_train_loss: float,
        updates_since_evaluation: int,
        training_seconds: float,
    ) -> None:
        metrics, evaluation_seconds = evaluate_model(
            model,
            tokenizer,
            tests,
            test_prompts,
            config,
            args.num_rollouts,
        )
        selected_uids = sorted(selected[index].uid for index in order)
        selection_hash = hashlib.sha256(
            "\n".join(selected_uids).encode("utf-8")
        ).hexdigest()[:16]
        domain_counts = dict(
            sorted(Counter(selected[index].domain for index in order).items())
        )
        record = {
            "variant": args.variant,
            "epoch": epoch,
            "update": cumulative_optimizer_updates,
            "selected_count": len(order),
            "candidate_pool_size": len(selected),
            "cumulative_unique_examples": len(seen_examples),
            "selection_hash": selection_hash,
            "domain_counts": domain_counts,
            "train_loss": mean_train_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "training_seconds_since_evaluation": training_seconds,
            "cumulative_training_seconds": cumulative_training,
            "evaluation_seconds": evaluation_seconds,
            "test_log_loss": metrics["test_log_loss"],
            "test_log_loss_standard_error": metrics[
                "test_log_loss_standard_error"
            ],
            "test_count": metrics["test_count"],
            "test_accuracy": metrics["test_accuracy"],
            "test_accuracy_standard_error": metrics[
                "test_accuracy_standard_error"
            ],
            "original_test_log_loss": original_metrics["test_log_loss"],
            "original_test_log_loss_standard_error": original_metrics[
                "test_log_loss_standard_error"
            ],
            "original_evaluation_seconds": original_evaluation_seconds,
            "original_test_accuracy": original_metrics["test_accuracy"],
            "objective": (
                "retrieved_context_teacher_to_student_kl"
                if args.context_distillation
                else "hard_answer_sft"
            ),
            "batch_size": batch_size,
            "optimizer_updates_since_evaluation": updates_since_evaluation,
            "cumulative_optimizer_updates": cumulative_optimizer_updates,
        }
        records.append(record)
        write_progress(output, records)
        print(json.dumps(record, indent=2))

    interval_loss_sum = 0.0
    interval_loss_units = 0
    interval_training_seconds = 0.0
    interval_updates = 0
    carry: list[int] = []
    for epoch in range(1, planned_epochs + 1):
        model.train()
        if args.random_sample_count:
            order = rng.sample(
                range(len(selected)), args.random_sample_count
            )
        else:
            order = list(range(len(selected)))
            rng.shuffle(order)
        seen_examples.update(order)
        training_order = carry + order if args.max_updates else order
        if args.max_updates:
            processable = (len(training_order) // batch_size) * batch_size
            carry = training_order[processable:]
            training_order = training_order[:processable]
        total_tokens = sum(len(responses[index]) for index in order)
        epoch_loss_sum = 0.0
        epoch_training_seconds = 0.0
        progress = tqdm(
            total=min(
                len(training_order),
                (total_updates - cumulative_optimizer_updates) * batch_size,
            ),
            desc=f"{args.variant} update {cumulative_optimizer_updates}/{total_updates}",
        )
        epoch_updates = 0
        for start in range(0, len(training_order), batch_size):
            if cumulative_optimizer_updates >= total_updates:
                break
            batch = training_order[start : start + batch_size]
            batch_tokens = sum(len(responses[index]) for index in batch)
            optimizer.zero_grad(set_to_none=True)
            batch_started = time.perf_counter()
            batch_loss_sum = 0.0
            for index in batch:
                if args.context_distillation:
                    input_ids = torch.tensor(
                        [train_prompts[index]], dtype=torch.long, device=device
                    )
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids),
                    }
                    labels = selected[index].choice_labels or [
                        chr(ord("A") + item)
                        for item in range(len(selected[index].choices))
                    ]
                    with torch.autocast(
                        "cuda", dtype=torch_dtype(config["model"]["dtype"])
                    ):
                        logits = response_logits(model, inputs, width=1)[0, -1]
                    student = option_distribution(
                        logits.float(), option_token_ids(tokenizer, labels)
                    ).clamp_min(1e-8)
                    student = student / student.sum()
                    teacher = teacher_choice_probabilities[index].to(device).clamp_min(1e-8)
                    teacher = teacher / teacher.sum()
                    sample_loss = torch.sum(
                        teacher * (teacher.log() - student.log())
                    )
                    loss = sample_loss / len(batch)
                    epoch_loss_sum += float(sample_loss.item())
                    batch_loss_sum += float(sample_loss.item())
                else:
                    inputs, mask = distillation_inputs_from_prompt_ids(
                        train_prompts[index],
                        responses[index],
                        train_config["max_sequence_tokens"],
                        device,
                    )
                    width = len(responses[index])
                    with torch.autocast(
                        "cuda", dtype=torch_dtype(config["model"]["dtype"])
                    ):
                        logits = response_logits(model, inputs, width)
                        targets = torch.tensor(
                            responses[index], dtype=torch.long, device=device
                        ).view(1, -1)
                        token_loss = F.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            reduction="none",
                        ).view(1, -1)
                        sample_loss = (token_loss * mask).sum()
                        loss = sample_loss / max(batch_tokens, 1)
                    epoch_loss_sum += float(sample_loss.item())
                    batch_loss_sum += float(sample_loss.item())
                loss.backward()
                progress.update(1)
            torch.nn.utils.clip_grad_norm_(
                trainable, train_config["gradient_clip_norm"]
            )
            optimizer.step()
            scheduler.step()
            batch_seconds = time.perf_counter() - batch_started
            epoch_training_seconds += batch_seconds
            interval_training_seconds += batch_seconds
            epoch_updates += 1
            cumulative_optimizer_updates += 1
            interval_updates += 1
            interval_loss_sum += batch_loss_sum
            interval_loss_units += len(batch) if args.context_distillation else batch_tokens

            should_evaluate = bool(
                args.max_updates
                and (
                    cumulative_optimizer_updates % args.eval_every_updates == 0
                    or cumulative_optimizer_updates == total_updates
                )
            )
            if should_evaluate:
                cumulative_training += interval_training_seconds
                append_evaluation(
                    epoch,
                    order,
                    interval_loss_sum / max(interval_loss_units, 1),
                    interval_updates,
                    interval_training_seconds,
                )
                interval_loss_sum = 0.0
                interval_loss_units = 0
                interval_training_seconds = 0.0
                interval_updates = 0
        progress.close()
        epoch_loss = epoch_loss_sum / max(
            len(order) if args.context_distillation else total_tokens, 1
        )
        if not args.max_updates:
            cumulative_training += epoch_training_seconds
            append_evaluation(
                epoch,
                order,
                epoch_loss,
                epoch_updates,
                epoch_training_seconds,
            )
        if cumulative_optimizer_updates >= total_updates:
            break

    final = output / "final_adapter"
    model.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    print(f"Saved final adapter and {len(records)} epoch records to {output}")


if __name__ == "__main__":
    main()
