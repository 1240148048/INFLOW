from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import get_scheduler, set_seed

from .chat_encoding import encode_messages
from .io import Example, load_config, read_jsonl
from .model_inputs import distillation_inputs_from_prompt_ids
from .modeling import load_causal_model, load_tokenizer, torch_dtype
from .prompts import training_messages


def response_logits(model, inputs: dict[str, torch.Tensor], width: int):
    try:
        output = model(**inputs, logits_to_keep=width, use_cache=False)
        logits = output.logits
    except TypeError:
        output = model(**inputs, use_cache=False)
        logits = output.logits[:, -width:, :]
    return logits[:, -width:, :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model")
    parser.add_argument("--selected", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This entry point requires one CUDA GPU.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
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
    if not selected:
        raise ValueError("Training selection is empty.")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=train_config["learning_rate"], weight_decay=0.0
    )
    scheduler = get_scheduler(
        train_config["scheduler"],
        optimizer,
        num_warmup_steps=int(args.epochs * train_config["warmup_ratio"]),
        num_training_steps=args.epochs,
    )
    rng = random.Random(config["seed"])
    step = 0
    tokens_seen = 0
    examples_seen = 0
    started_at = time.perf_counter()
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(selected)))
        rng.shuffle(order)
        responses = {}
        total_tokens = 0
        for index in order:
            response = tokenizer(
                selected[index].rollout or "", add_special_tokens=False
            )["input_ids"][: train_config["rollout_max_new_tokens"]]
            if not response:
                raise ValueError(f"Empty target for {selected[index].uid}.")
            eos = tokenizer.eos_token_id
            eos = eos[0] if isinstance(eos, list) else eos
            if eos is not None and response[-1] != eos:
                response.append(eos)
            responses[index] = response
            total_tokens += len(response)
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for index in tqdm(
            order, desc=f"accumulating full-batch SFT epoch {epoch + 1}/{args.epochs}"
        ):
            example = selected[index]
            response = responses[index]
            encoded_prompt = encode_messages(
                tokenizer,
                training_messages(example),
                train_config["max_sequence_tokens"] - len(response) + 1,
                device,
                enable_thinking=False,
            )
            inputs, mask = distillation_inputs_from_prompt_ids(
                encoded_prompt["input_ids"][0].tolist(),
                response,
                train_config["max_sequence_tokens"],
                device,
            )
            width = len(response)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = response_logits(model, inputs, width)
                targets = torch.tensor(
                    response, dtype=torch.long, device=device
                ).view(1, -1)
                token_loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).view(1, -1)
                loss = (token_loss * mask).sum() / max(total_tokens, 1)
            loss.backward()
            epoch_loss += float(loss.item())
            tokens_seen += len(response)
            examples_seen += 1
        torch.nn.utils.clip_grad_norm_(
            trainable, train_config["gradient_clip_norm"]
        )
        optimizer.step()
        scheduler.step()
        step += 1
        allocated = torch.cuda.max_memory_allocated() / 2**30
        print(
            f"epoch={epoch + 1}/{args.epochs} examples={len(selected)} "
            f"tokens={total_tokens} loss={epoch_loss:.6f} "
            f"lr={scheduler.get_last_lr()[0]:.3e} peak_gpu={allocated:.2f}GiB"
        )
    output = Path(args.output_dir or train_config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    elapsed = time.perf_counter() - started_at
    (output / "training_metrics.json").write_text(
        json.dumps(
            {
                "training_seconds": elapsed,
                "objective": "hard_answer_sft",
                "optimizer_steps": step,
                "epochs": args.epochs,
                "selected_examples": len(selected),
                "training_examples_seen": examples_seen,
                "total_target_tokens": tokens_seen,
                "mean_target_tokens": tokens_seen / examples_seen,
                "optimizer_updates_per_epoch": 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved LoRA adapter to {output}")


if __name__ == "__main__":
    main()
