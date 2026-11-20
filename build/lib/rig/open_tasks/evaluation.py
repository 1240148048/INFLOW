from __future__ import annotations

import ast
import hashlib
import json
import statistics
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from ..io import append_jsonl, read_jsonl, write_jsonl
from .data import dump
from .runtime import Runtime, messages
from .semantics import extract_code, tree_metrics
from .training import adapter_directory, update_checkpoint


def evaluation_dir(output, update):
    return output if update is None else output / "curve" / f"step-{update:06d}"


def predict(config, data_dir, output, method, runtime_factory=Runtime, update=None):
    data_dir, output = Path(data_dir), Path(output)
    rows = list(read_jsonl(data_dir / "eval.jsonl"))
    schema = json.loads((data_dir / "schema.json").read_text(encoding="utf-8"))
    if not rows or any(row["split"] != "test" for row in rows):
        raise ValueError("Evaluation input must contain only test rows")
    if method == "original" and update is not None:
        raise ValueError("Original is evaluated once, without an update")
    target = output / "checkpoints" / method if update is None else update_checkpoint(output, method, update)
    if update is not None:
        marker = json.loads((target / "checkpoint.json").read_text(encoding="utf-8"))
        if marker != {"method": method, "optimizer_update": update}:
            raise ValueError("Adapter snapshot does not match the requested optimizer update")
    adapter = None if method == "original" else str(adapter_directory(target))
    path = evaluation_dir(output, update) / "predictions" / f"{method}.jsonl"
    manifest = path.with_suffix(".meta.json")
    identity = {"config": config, "method": method, "adapter": adapter, "eval_rows": rows,
                "prompt_protocol": "task-contract-answer-only-v5"}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != identity:
        raise ValueError("Prediction input/config changed; use a new output directory")
    dump(manifest, identity)
    seen = set()
    cached = {}
    if path.exists():
        for row in read_jsonl(path):
            key = row["uid"], row["sample_index"]
            if key in seen:
                raise ValueError("Duplicate cached prediction")
            seen.add(key)
            cached[key] = row
    count = config["evaluation"]["num_samples"]
    expected = {(row["uid"], i) for row in rows for i in range(count)}
    if seen - expected:
        raise ValueError("Unexpected cached evaluation IDs/sample indices")
    if seen == expected:
        write_jsonl(path.with_name(f"{method}.code.jsonl"), [p for p in cached.values() if p.get("task") == "code"])
        print(f"Reusing completed predictions: {method}")
        return
    runtime = runtime_factory(config, adapter=adapter)
    lookup = {row["uid"]: row for row in rows}
    for index, row in enumerate(rows):
        for sample in range(count):
            if (row["uid"], sample) in seen:
                continue
            started = time.perf_counter()
            effective = dict(row)
            if row["task"] == "coqa":
                from .qa import question
                history = [(lookup[past]["query"], cached[past, sample]["text"]) for past in row["history_uids"]]
                effective["question"] = question(row["context"], row["query"], history)
            texts, tokens, _ = runtime.sample(messages(effective, schema), 1, config["seed"] + 3_000_000 + index * count + sample, config["evaluation"])
            result = {"uid": row["uid"], "sample_index": sample, "task": row["task"], "text": texts[0], "tokens": len(tokens[0]), "seconds": time.perf_counter() - started}
            if row["task"] == "code":
                result.update({"task_id": row["uid"], "solution": extract_code(texts[0])})
            append_jsonl(path, result)
            cached[row["uid"], sample] = result
        print(f"evaluate {method}: {index + 1}/{len(rows)}", flush=True)
    write_jsonl(path.with_name(f"{method}.code.jsonl"), [p for p in cached.values() if p.get("task") == "code"])


def validate_predictions(rows, predictions, count):
    expected = {(row["uid"], i) for row in rows for i in range(count)}
    actual = [(row["uid"], row["sample_index"]) for row in predictions]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Predictions are duplicated/incomplete or use a different evaluation set")


def score(config, data_dir, output, method, execution_results=None, update=None):
    output = Path(output)
    evaluation_file = Path(data_dir) / "eval.jsonl"
    rows = list(read_jsonl(evaluation_file))
    target = evaluation_dir(output, update)
    predictions = list(read_jsonl(target / "predictions" / f"{method}.jsonl"))
    count = config["evaluation"]["num_samples"]
    validate_predictions(rows, predictions, count)
    references = {row["uid"]: row for row in rows}
    tasks = sorted({row["task"] for row in rows})
    by_task = {}
    for task in tasks:
        subset = [row for row in rows if row["task"] == task]
        selected_predictions = [p for p in predictions if references[p["uid"]]["task"] == task]
        by_task[task] = score_task(task, subset, selected_predictions, count, execution_results)
    per_question = {uid: value for item in by_task.values() for uid, value in item["per_question"].items()}
    rollout_accuracies = [
        sum(item["questions"] * item["rollout_index_accuracies"][i] for item in by_task.values()) / len(rows)
        for i in range(count)
    ]
    metrics = {
        "task": tasks[0] if len(tasks) == 1 else "mixed", "method": method,
        "questions": len(rows), "samples_per_question": count, "by_task": by_task,
        "overall_accuracy": 100 * statistics.mean(per_question.values()),
        "rollout_index_accuracies": rollout_accuracies,
        "rollout_accuracy_sd": statistics.pstdev(rollout_accuracies),
        "optimizer_update": 0 if method == "original" else update,
        "model": config.get("model", {}).get("name"),
        "evaluation_id": hashlib.sha256(evaluation_file.read_bytes()).hexdigest(),
        "overall_task_contributions_pp": {
            task: 100 * sum(item["per_question"].values()) / len(rows) for task, item in by_task.items()
        },
        "per_question": per_question,
        "bootstrap_groups": {row["uid"]: row.get("conversation_id", row["uid"]) for row in rows},
        "generation_seconds": sum(p["seconds"] for p in predictions),
        "generated_tokens": sum(p["tokens"] for p in predictions),
        "units": "percent except per_question fractions; overall is question-weighted TOPv2 labeled-tree EM, QA EM and MBPP pass@1",
    }
    if "topv2" in by_task:
        top = by_task["topv2"]
        metrics["surface_overall_accuracy"] = metrics["overall_accuracy"] + (
            (top["surface_exact_match"] - top["exact_match"]) * top["questions"] / len(rows)
        )
    if len(tasks) == 1:
        metrics.update(by_task[tasks[0]])
    dump(target / "metrics" / f"{method}.json", metrics)
    if update is not None and update == config["training"]["updates"]:
        dump(output / "metrics" / f"{method}.json", metrics)
    print(f"Scored {method}: {len(rows)} questions across {tasks}")
    return metrics


def score_task(task, rows, predictions, count, execution_results):
    references = {row["uid"]: row for row in rows}
    columns = [[] for _ in range(count)]
    if task == "topv2":
        totals = Counter()
        per_question = {row["uid"]: [] for row in rows}
        for prediction in predictions:
            values = tree_metrics(prediction["text"], references[prediction["uid"]]["reference"])
            totals.update(values)
            columns[prediction["sample_index"]].append(values["exact"])
            per_question[prediction["uid"]].append(values["exact"])
        denominator = totals["slot_pred"] + totals["slot_gold"]
        metrics = {
            "exact_match": 100 * totals["exact"] / len(predictions),
            "surface_exact_match": 100 * totals["surface_exact"] / len(predictions),
            "strict_exact_match": 100 * totals["strict_exact"] / len(predictions),
            "intent_accuracy": 100 * totals["intent"] / len(predictions),
            "hierarchical_slot_f1": 100 * 2 * totals["slot_tp"] / denominator if denominator else 100.0,
            "valid_parse_rate": 100 * totals["valid"] / len(predictions),
            "per_question": {uid: sum(v) / len(v) for uid, v in per_question.items()},
        }
    elif task in {"coqa", "squad", "hotpotqa"}:
        from .qa import qa_metrics
        per_question = {row["uid"]: [] for row in rows}
        totals = Counter()
        for prediction in predictions:
            if task == "hotpotqa":
                from .hotpot import answer_metrics
                values = answer_metrics(prediction["text"], references[prediction["uid"]]["references"])
            else:
                values = qa_metrics(prediction["text"], references[prediction["uid"]]["references"])
            totals.update(values)
            per_question[prediction["uid"]].append(values["exact"])
            columns[prediction["sample_index"]].append(values["exact"])
        metrics = {"exact_match": 100 * totals["exact"] / len(predictions), "token_f1": 100 * totals["f1"] / len(predictions), "valid_answer_rate": 100 * totals["valid"] / len(predictions), "per_question": {uid: statistics.mean(values) for uid, values in per_question.items()}}
    else:
        if not execution_results:
            raise ValueError("Run the Docker MBPP evaluator first, then pass --execution-results")
        evaluation = json.loads(Path(execution_results).read_text(encoding="utf-8"))["eval"]
        if set(evaluation) != set(references) or any(len(items) != count for items in evaluation.values()):
            raise ValueError("MBPP task IDs/sample counts do not match the complete predictions")
        # Verify this evaluator result belongs to these exact generated programs.
        for uid, items in evaluation.items():
            supplied = Counter(p["solution"] for p in predictions if p["uid"] == uid)
            evaluated = Counter(item["solution"] for item in items)
            if supplied != evaluated:
                raise ValueError(f"Stale MBPP execution results for {uid}")
        valid = 0
        for row in predictions:
            try:
                ast.parse(row["solution"])
                valid += 1
            except SyntaxError:
                pass
        base = {}
        statuses = defaultdict(deque)
        for uid, items in evaluation.items():
            base[uid] = sum(item["base_status"] == "pass" for item in items) / count
            for item in items:
                statuses[uid, item["solution"]].append(item["base_status"] == "pass")
        for prediction in predictions:
            value = statuses[prediction["uid"], prediction["solution"]].popleft()
            columns[prediction["sample_index"]].append(value)
        metrics = {"pass_at_1": 100 * statistics.mean(base.values()), "syntax_valid_rate": 100 * valid / len(predictions), "per_question": base}
    metrics["questions"] = len(rows)
    metrics["rollout_index_accuracies"] = [100 * statistics.mean(column) for column in columns]
    metrics["rollout_accuracy_sd"] = statistics.pstdev(metrics["rollout_index_accuracies"])
    return metrics
