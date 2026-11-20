from __future__ import annotations

import json
import random
import time
from pathlib import Path

from ..io import append_jsonl, read_jsonl, write_jsonl
from .data import dump
from .runtime import Runtime, messages


def trim_tokens(tokens, eos, pad):
    end = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
    result = []
    for token in tokens:
        if token == pad and token not in end:
            break
        result.append(int(token))
        if token in end:
            break
    if not result:
        raise ValueError("The student generated an empty trajectory")
    return result


def adapter_directory(path):
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return path
    if (path / "student" / "adapter_config.json").exists():
        return path / "student"
    raise FileNotFoundError(f"No saved student adapter in {path}")


def update_checkpoint(output, method, update):
    return Path(output) / "checkpoints" / method / "updates" / f"step-{update:06d}"


def initialize_adapters(runtime, config):
    from peft import LoraConfig, get_peft_model
    from ..opsd_train import activate, adapter_pairs, initialize_teacher
    from transformers import set_seed
    training = config["training"]
    set_seed(config["seed"])
    lora = LoraConfig(r=training["lora_rank"], lora_alpha=training["lora_alpha"], lora_dropout=0.0, target_modules=training["lora_targets"], bias="none", task_type="CAUSAL_LM")
    runtime.model = get_peft_model(runtime.model, lora, adapter_name="student")
    runtime.model.add_adapter("teacher", lora)
    pairs = adapter_pairs(runtime.model)
    initialize_teacher(pairs)
    activate(runtime.model, "student")
    runtime.model.config.use_cache = False
    if training.get("gradient_checkpointing", True):
        runtime.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        runtime.model.enable_input_require_grads()
    return pairs


def on_policy_loss(runtime, row, demonstrations, schema, config, seed):
    """Refresh student rollouts, then differentiate KL on fixed sampled prefixes."""
    import torch
    from ..model_inputs import distillation_inputs_from_prompt_ids
    from ..opsd_train import activate, response_logits, reverse_kl
    training = config["training"]
    activate(runtime.model, "student")
    runtime.model.eval()
    # No retrieval in student generation. Prepared teacher trajectories are not used.
    prompt_version = config.get("preparation_prompt_version")
    _, responses, student_prompt = runtime.sample(messages(row, schema, prompt_version=prompt_version), 1, seed, config["rollout"])
    response = responses[0]
    maximum = training["max_sequence_tokens"] - len(response) + 1
    if maximum < len(student_prompt):
        raise ValueError("Student prompt + rollout exceeds max_sequence_tokens")
    teacher_prompt = runtime.encode(
        messages(row, schema, demonstrations, self_feedback=row.get("self_feedback"), prompt_version=prompt_version), maximum
    )["input_ids"][0].tolist()
    teacher_inputs, _ = distillation_inputs_from_prompt_ids(teacher_prompt, response, training["max_sequence_tokens"], runtime.device)
    student_inputs, _ = distillation_inputs_from_prompt_ids(student_prompt, response, training["max_sequence_tokens"], runtime.device)
    activate(runtime.model, "teacher")
    runtime.model.eval()
    with torch.no_grad():
        teacher_logits = response_logits(runtime.model, teacher_inputs, len(response))
    activate(runtime.model, "student")
    runtime.model.train()
    student_logits = response_logits(runtime.model, student_inputs, len(response))
    loss = reverse_kl(student_logits, teacher_logits, training["kl_chunk_tokens"])
    return loss, len(response)


def train(config, data_dir, output, method, runtime_factory=Runtime):
    import torch
    from transformers import get_scheduler
    from ..opsd_train import save_student_checkpoint, update_teacher
    from .pipeline import experiment, records
    output = Path(output)
    _, schema = experiment(config, data_dir, output)
    selected = list(read_jsonl(output / "subsets" / f"{method}.jsonl"))
    if not selected or any(row["selector"] != method or row["split"] != "train" for row in selected):
        raise ValueError("Invalid selected training subset")
    candidates = {} if method == "consistency" else records(output / "initial.jsonl")
    for row in selected:
        if any(uid == row["uid"] or uid not in candidates for uid in row["demo_ids"]):
            raise ValueError("Missing/self demonstration")
    target = output / "checkpoints" / method
    identity = {"config": config, "method": method,
                "selection": [{"uid": r["uid"], "demo_ids": r["demo_ids"], "self_feedback": r.get("self_feedback")} for r in selected]}
    manifest = target / "run.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError("Checkpoint configuration/subset changed; use a new output")
    dump(manifest, identity)
    if (target / "training_metrics.json").exists():
        adapter_directory(target)
        print(f"Reusing completed training: {method}")
        return
    load_started = time.perf_counter()
    runtime = runtime_factory(config)
    setup_started = time.perf_counter()
    pairs = initialize_adapters(runtime, config)
    training = config["training"]
    parameters = [parameter for parameter in runtime.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=0.0)
    updates, batch_size = training["updates"], training["batch_size"]
    if updates < 1 or batch_size < 1:
        raise ValueError("Training budgets must be positive")
    scheduler = get_scheduler("cosine", optimizer, int(updates * training["warmup_ratio"]), updates)
    rng = random.Random(config["seed"])
    order, cursor, completed, tokens, previous_seconds = [], 0, 0, 0, 0.0
    previous_startup = 0.0
    state_path = target / "training_state.pt"
    if state_path.exists():
        state = torch.load(state_path, map_location=runtime.device, weights_only=False)
        named = dict(runtime.model.named_parameters())
        with torch.no_grad():
            for name, value in state["adapters"].items():
                named[name].copy_(value)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        order, cursor, completed, tokens, previous_seconds = state["order"], state["cursor"], state["completed"], state["tokens"], state["seconds"]
        previous_startup = state.get("startup_seconds", 0.0)
        ready = update_checkpoint(output, method, completed) / "ready_time.json"
        if ready.exists():
            previous_seconds = json.loads(ready.read_text())["distillation_seconds"]
        rng.setstate(state["random_state"])
        print(f"Resuming {method} after update {completed}")
    curve = target / "training_curve.jsonl"
    if curve.exists():
        write_jsonl(curve, [row for row in read_jsonl(curve) if row["update"] <= completed])
    started = time.perf_counter()
    startup_seconds = previous_startup + started - load_started

    def save(update):
        snapshot = update_checkpoint(output, method, update)
        save_student_checkpoint(runtime.model, runtime.tokenizer, snapshot)
        dump(snapshot / "checkpoint.json", {"method": method, "optimizer_update": update})
        if update == updates:
            save_student_checkpoint(runtime.model, runtime.tokenizer, target)
        adapters = {name: p.detach().cpu() for name, p in runtime.model.named_parameters() if ".student." in name or ".teacher." in name}
        state = {"adapters": adapters, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "order": order, "cursor": cursor, "completed": update, "tokens": tokens, "random_state": rng.getstate(), "seconds": previous_seconds + time.perf_counter() - started, "startup_seconds": startup_seconds}
        temporary = state_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(state_path)
        dump(snapshot / "ready_time.json", {"distillation_seconds": previous_seconds + time.perf_counter() - started,
                                            "startup_seconds": startup_seconds})

    for update in range(completed, updates):
        optimizer.zero_grad(set_to_none=True)
        batch_loss = 0.0
        for item in range(batch_size):
            if cursor >= len(order):
                order = list(range(len(selected)))
                rng.shuffle(order)
                cursor = 0
            row = selected[order[cursor]]
            cursor += 1
            demos = [candidates[uid] for uid in row["demo_ids"]]
            loss, length = on_policy_loss(runtime, row, demos, schema, config, config["seed"] + 1_000_000 + update * batch_size + item)
            (loss / batch_size).backward()
            batch_loss += float(loss.detach()) / batch_size
            tokens += length
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        update_teacher(pairs, 1 - training["ema_retention"])
        append = {"update": update + 1, "loss": batch_loss, "tokens": tokens, "seconds": previous_seconds + time.perf_counter() - started}
        append_jsonl(curve, append)
        print(f"{method}: update {update + 1}/{updates}, KL={batch_loss:.6f}", flush=True)
        if (update + 1) % training["checkpoint_every"] == 0 or update + 1 == updates:
            save(update + 1)
    dump(target / "training_metrics.json", {
        "method": method, "objective": "on_policy_fixed_rollout_token_reverse_kl",
        "updates": updates, "trajectories": updates * batch_size, "tokens": tokens,
        "seconds": previous_seconds + time.perf_counter() - started,
        "model_loading_seconds": setup_started - load_started,
        "adapter_setup_seconds": started - setup_started,
        "selected_count": len(selected), "ema_retention": training["ema_retention"],
    })
