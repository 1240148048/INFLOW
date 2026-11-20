from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint
from transformers import get_scheduler, set_seed

from .chat_encoding import encode_messages
from .io import Example, load_config, read_jsonl
from .model_inputs import distillation_inputs_from_prompt_ids
from .modeling import load_causal_model, load_tokenizer, torch_dtype
from .opsd_common import reasoning_messages


def response_logits(model, inputs, width):
    try:
        output = model(**inputs, logits_to_keep=width, use_cache=False)
        return output.logits[:, -width:, :]
    except TypeError:
        output = model(**inputs, use_cache=False)
        return output.logits[:, -width:, :]


def adapter_pairs(model):
    parameters = dict(model.named_parameters())
    pairs = []
    for name, student in parameters.items():
        if ".student." not in name:
            continue
        teacher_name = name.replace(".student.", ".teacher.")
        if teacher_name in parameters:
            pairs.append((parameters[teacher_name], student))
    if not pairs:
        raise RuntimeError("Could not pair student and EMA-teacher LoRA parameters.")
    return pairs


def activate(model, name: str):
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        if ".lora_" in parameter_name:
            parameter.requires_grad_(
                name == "student" and ".student." in parameter_name
            )


@torch.no_grad()
def initialize_teacher(pairs):
    for teacher, student in pairs:
        teacher.copy_(student)


@torch.no_grad()
def update_teacher(pairs, rate: float):
    for teacher, student in pairs:
        teacher.mul_(1.0 - rate).add_(student, alpha=rate)


def prompt_ids(tokenizer, messages, max_length, device):
    return encode_messages(
        tokenizer,
        messages,
        max_length,
        device,
        enable_thinking=False,
    )["input_ids"][0].tolist()


def trim_generated_response(response: list[int], tokenizer) -> list[int]:
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    output = []
    for token in response:
        if token == tokenizer.pad_token_id:
            break
        output.append(int(token))
        if token in eos_ids:
            break
    if not output:
        raise RuntimeError("The on-policy student generated an empty trajectory.")
    return output


def reverse_kl(student_logits, teacher_logits, chunk_tokens: int | None = None):
    """Exact token-mean reverse KL with an optional lower-peak-memory path."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "Student and teacher logits must have the same shape, got "
            f"{tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )
    tokens = student_logits.shape[-2]
    if tokens < 1:
        raise ValueError("Reverse KL requires at least one response token.")
    if chunk_tokens is None:
        chunk_tokens = tokens
    if chunk_tokens < 1:
        raise ValueError("training.kl_chunk_tokens must be positive.")

    def chunk_sum(student_chunk, teacher_chunk):
        log_student = F.log_softmax(student_chunk.float(), dim=-1)
        log_teacher = F.log_softmax(teacher_chunk.float(), dim=-1)
        return (
            log_student.exp() * (log_student - log_teacher)
        ).sum(dim=-1).sum()

    use_checkpoint = (
        chunk_tokens < tokens
        and torch.is_grad_enabled()
        and student_logits.requires_grad
    )
    total = student_logits.new_zeros((), dtype=torch.float32)
    for start in range(0, tokens, chunk_tokens):
        end = min(start + chunk_tokens, tokens)
        student_chunk = student_logits[..., start:end, :]
        teacher_chunk = teacher_logits[..., start:end, :]
        if use_checkpoint:
            value = checkpoint(
                chunk_sum, student_chunk, teacher_chunk, use_reentrant=False
            )
        else:
            value = chunk_sum(student_chunk, teacher_chunk)
        total = total + value
    return total / student_logits[..., :, 0].numel()


def save_student_checkpoint(model, tokenizer, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    activate(model, "student")
    model.save_pretrained(
        destination, selected_adapters=["student"], safe_serialization=True
    )
    tokenizer.save_pretrained(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--checkpoint-every-trajectories", type=int)
    parser.add_argument("--final-checkpoint-only", action="store_true")
    parser.add_argument("--stop-after-updates", type=int,
                        help="Stop early while retaining the learning-rate schedule of the full trajectory budget")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("OPSD training requires CUDA.")
    config = load_config(args.config)
    training = config["training"]
    kl_chunk_tokens = int(training.get("kl_chunk_tokens", 0)) or None
    set_seed(int(config["seed"]))
    device = torch.device("cuda:0")
    name = args.model or config["model"]["name"]
    tokenizer = load_tokenizer(name, config["model"]["trust_remote_code"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = load_causal_model(
        name,
        torch_dtype(config["model"]["dtype"]),
        config["model"]["trust_remote_code"],
    ).to(device)
    lora = LoraConfig(
        r=int(training["lora_rank"]),
        lora_alpha=int(training["lora_alpha"]),
        lora_dropout=float(training["lora_dropout"]),
        target_modules=training["lora_targets"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora, adapter_name="student")
    model.add_adapter("teacher", lora)
    pairs = adapter_pairs(model)
    initialize_teacher(pairs)
    activate(model, "student")
    model.config.use_cache = False
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Student trainable parameters: {trainable_count:,}")

    selected = [Example.from_dict(row) for row in read_jsonl(args.selected)]
    candidates = {
        example.uid: example
        for example in (Example.from_dict(row) for row in read_jsonl(args.candidates))
    }
    if not selected:
        raise ValueError("The selected subset is empty.")
    for example in selected:
        missing = set(example.demo_ids) - set(candidates)
        if missing:
            raise KeyError(f"Missing demonstrations for {example.uid}: {sorted(missing)}")
    method = selected[0].metadata.get("opsd_selection", {}).get("method")
    retrieval_definition = selected[0].metadata.get("opsd_retrieval", {}).get(
        "definition"
    )
    if any(
        example.metadata.get("opsd_selection", {}).get("method") != method
        for example in selected
    ):
        raise ValueError("A training run must contain exactly one selection method.")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    if args.trajectories is not None:
        max_trajectories = args.trajectories
    elif "max_trajectories" in training:
        max_trajectories = int(training["max_trajectories"])
    elif args.updates is not None:
        max_trajectories = args.updates * batch_size
    else:
        max_trajectories = int(training["updates"]) * batch_size
    if max_trajectories < 1:
        raise ValueError("The trajectory budget must be positive.")
    updates = math.ceil(max_trajectories / batch_size)
    completed_updates = args.stop_after_updates if args.stop_after_updates is not None else updates
    if not 1 <= completed_updates <= updates:
        raise ValueError("--stop-after-updates must lie within the full update horizon")
    warmup = int(round(updates * float(training["warmup_ratio"])))
    scheduler = get_scheduler(
        training["scheduler"], optimizer, warmup, updates
    )
    randomizer = random.Random(int(config["seed"]))
    started = time.perf_counter()
    trajectories = 0
    tokens = 0
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.final_checkpoint_only:
        milestones = []
    elif args.checkpoint_every_trajectories:
        stride = int(args.checkpoint_every_trajectories)
        if stride < 1:
            raise ValueError("--checkpoint-every-trajectories must be positive.")
        milestones = list(range(stride, max_trajectories + 1, stride))
        if not milestones or milestones[-1] != max_trajectories:
            milestones.append(max_trajectories)
    else:
        milestones = sorted(
            set(
                int(value)
                for value in training.get("checkpoint_trajectories", [])
                if 0 < int(value) <= max_trajectories
            )
            | {max_trajectories}
        )
    saved_trajectories = []
    next_milestone = 0
    epoch_order: list[int] = []
    epoch_position = 0

    def next_example() -> Example:
        nonlocal epoch_order, epoch_position
        if epoch_position >= len(epoch_order):
            epoch_order = list(range(len(selected)))
            randomizer.shuffle(epoch_order)
            epoch_position = 0
        example = selected[epoch_order[epoch_position]]
        epoch_position += 1
        return example

    progress = tqdm(
        total=completed_updates,
        desc=f"training {method}",
        unit="update",
        dynamic_ncols=True,
    )
    for update in range(1, completed_updates + 1):
        current_batch_size = min(batch_size, max_trajectories - trajectories)
        batch = [next_example() for _ in range(current_batch_size)]
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for example in batch:
            if not example.rollout:
                raise ValueError(
                    f"Selected example {example.uid} has no prepared teacher response."
                )
            response = tokenizer(
                example.rollout, add_special_tokens=False
            )["input_ids"]
            eos = tokenizer.eos_token_id
            eos_id = eos[0] if isinstance(eos, list) else eos
            if eos_id is not None and (not response or response[-1] != eos_id):
                response.append(int(eos_id))
            response = trim_generated_response(response, tokenizer)
            demonstrations = []
            for uid in example.demo_ids:
                demonstration = copy.deepcopy(candidates[uid])
                demonstrations.append(demonstration)
            maximum = int(training["max_sequence_tokens"]) - len(response) + 1

            activate(model, "teacher")
            teacher_prompt = prompt_ids(
                tokenizer,
                reasoning_messages(example, demonstrations),
                maximum,
                device,
            )
            teacher_inputs, _ = distillation_inputs_from_prompt_ids(
                teacher_prompt,
                response,
                int(training["max_sequence_tokens"]),
                device,
            )
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits = response_logits(model, teacher_inputs, len(response))

            activate(model, "student")
            student_prompt = prompt_ids(
                tokenizer, reasoning_messages(example), maximum, device
            )
            student_inputs, _ = distillation_inputs_from_prompt_ids(
                student_prompt,
                response,
                int(training["max_sequence_tokens"]),
                device,
            )
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = response_logits(model, student_inputs, len(response))
                loss = reverse_kl(
                    student_logits, teacher_logits, kl_chunk_tokens
                ) / current_batch_size
            loss.backward()
            mean_loss += float(loss.detach())
            trajectories += 1
            tokens += len(response)
            del teacher_logits, student_logits, teacher_inputs, student_inputs

        torch.nn.utils.clip_grad_norm_(
            trainable, float(training["gradient_clip_norm"])
        )
        optimizer.step()
        scheduler.step()
        update_teacher(pairs, float(training["ema_rate"]))
        progress.update(1)
        while (
            next_milestone < len(milestones)
            and trajectories >= milestones[next_milestone]
        ):
            milestone = milestones[next_milestone]
            save_student_checkpoint(
                model, tokenizer, output / f"checkpoint-{milestone}"
            )
            saved_trajectories.append(milestone)
            next_milestone += 1
    progress.close()

    save_student_checkpoint(model, tokenizer, output)
    (output / "training_metrics.json").write_text(
        json.dumps(
            {
                "objective": "prepared_teacher_trajectory_reverse_kl_with_ema_teacher",
                "optimizer_updates": completed_updates,
                "scheduler_horizon_updates": updates,
                "batch_size": batch_size,
                "trajectory_budget": max_trajectories,
                "trajectories": trajectories,
                "completed_dataset_passes": trajectories / len(selected),
                "tokens": tokens,
                "seconds": time.perf_counter() - started,
                "ema_rate": training["ema_rate"],
                "kl_chunk_tokens": kl_chunk_tokens,
                "student_trainable_parameters": trainable_count,
                "context_count": len(selected[0].demo_ids),
                "retrieval_definition": retrieval_definition,
                "checkpoints_by_trajectory": saved_trajectories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved student adapter to {output}")


if __name__ == "__main__":
    main()
