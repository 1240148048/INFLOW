"""Extend the original MC ablation while retaining the original run artifacts."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

print("[init] Loading NumPy, PyTorch and Transformers...", flush=True)
import numpy as np

from .io import Example, append_jsonl, load_config, read_jsonl, write_jsonl
from .open_tasks.data import dump
from .opsd_ablation_finalize import attach_selection
from .opsd_common import labels_for
from .opsd_select import information_scores, ranked_contexts, representation_tensor, write_retrieval
from .opsd_teacher_prepare import jensen_shannon
print("[init] Runtime imports complete", flush=True)


VARIANTS = ("last_layer_certainty", "last_layer_no_certainty", "sample_1_1_js",
            "sample_5_5_js", "sample_1_5_hard_flip")
EXISTING = ("ours", "no_certainty", "no_similarity", "random20", "original")
TEACHER = "opsd_teacher_generation"
INITIAL = "neuron_opsd"


def examples(path):
    return [Example.from_dict(row) for row in read_jsonl(path)]


def load_unique(path):
    rows = examples(path) if path.exists() else []
    result = {row.uid: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate cached IDs in {path}")
    return result


def probability(samples, labels):
    matrix = np.asarray([[sample["choice_probabilities"][label] for label in labels]
                         for sample in samples], dtype=float)
    if not len(matrix) or not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Invalid cached choice probabilities")
    if not np.allclose(matrix.sum(axis=1), 1):
        raise ValueError("Choice distributions must sum to one")
    return matrix.mean(axis=0)


def score_and_select(rows, initial, variant, seed, fraction=.2):
    count = max(1, round(len(rows) * fraction))
    scored = []
    for row in rows:
        row = copy.deepcopy(row)
        labels = labels_for(row)
        left = probability(initial[row.uid], labels)
        samples = row.metadata[TEACHER]["samples"]
        if variant == "sample_1_1_js":
            samples = samples[:1]
            row.rollout = samples[0]["text"]
        right = probability(samples, labels)
        flip = int(left.argmax() != right.argmax())
        js = jensen_shannon(left.copy(), right.copy())
        score = flip if variant == "sample_1_5_hard_flip" else js
        row.metadata["rebuttal_beliefs"] = {"initial": left.tolist(), "contextual": right.tolist(),
                                           "labels": labels, "js": js, "hard_flip": flip}
        scored.append((row, float(score)))
    # Random ordering is independent of JS, including within both binary groups.
    scored.sort(key=lambda item: item[0].uid)
    if variant == "sample_1_5_hard_flip":
        random.Random(seed).shuffle(scored)
        scored.sort(key=lambda item: -item[1])
    else:
        scored.sort(key=lambda item: (-item[1], item[0].uid))
    selected = [attach_selection(row, variant, i, variant, score)
                for i, (row, score) in enumerate(scored[:count], 1)]
    return selected, [row.to_dict() for row, _ in scored]


def import_existing(source, ablation, output, steps):
    """Import full precision raw metrics. Never estimate values from the PNG."""
    records = []
    exported = {}
    for root in dict.fromkeys((source, ablation)):
        path = root / "module_ablation_results.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    key = row["method"], int(row["optimizer_update"])
                    value = float(row["accuracy"]), float(row["accuracy_std"]), str(path)
                    if key in exported and exported[key][:2] != value[:2]:
                        raise ValueError(f"Conflicting original CSV values for {key}")
                    exported[key] = value
    for method in EXISTING:
        root = source if method in {"ours", "original"} else ablation
        points = [0] if method == "original" else steps
        for step in points:
            folder = root / "eval" / method
            if step:
                folder /= f"update-{step}"
            path = folder / "metrics.json"
            record = {"method": method, "optimizer_update": step, "avg_percent": "", "sd_percent": "",
                      "status": "missing_raw_metrics", "source": str(path)}
            if path.exists():
                overall = json.loads(path.read_text())["overall"]
                key = next(key for key in overall if key.startswith("avg_at_"))
                mean, sd = float(overall[key]), float(overall["rollout_accuracy_std"])
                if not 0 <= mean <= 1 or not 0 <= sd <= 1:
                    raise ValueError(f"Expected fraction units in {path}")
                record.update(avg_percent=f"{100 * mean:.2f}", sd_percent=f"{100 * sd:.2f}",
                              status="reused_raw_metrics")
            elif (method, step) in exported:
                mean, sd, origin = exported[method, step]
                if not 0 <= mean <= 1 or not 0 <= sd <= 1:
                    raise ValueError(f"Expected fraction units in {origin}")
                record.update(avg_percent=f"{100 * mean:.2f}", sd_percent=f"{100 * sd:.2f}",
                              status="reused_original_csv", source=origin)
            records.append(record)
    digitized_csv = os.environ.get("DIGITIZED_CSV")
    if digitized_csv:
        with Path(digitized_csv).open(newline="", encoding="utf-8-sig") as handle:
            estimates = {}
            for row in csv.DictReader(handle):
                key = row["method"], int(row["optimizer_update"])
                if key in estimates:
                    raise ValueError(f"Duplicate digitized point: {key}")
                mean = float(row["avg_percent"])
                sd = float(row["sd_percent"]) if row["sd_percent"].strip() else None
                if not 0 <= mean <= 100 or (sd is not None and not 0 <= sd <= 100):
                    raise ValueError(f"Invalid percentage values for {key}")
                if sd is None and row["method"] != "original":
                    raise ValueError(f"Missing digitized SD for {key}")
                estimates[key] = (mean, sd)
        for record in records:
            key = record["method"], record["optimizer_update"]
            if record["status"] == "missing_raw_metrics" and key in estimates:
                mean, sd = estimates[key]
                record.update(avg_percent=f"{mean:.2f}",
                              sd_percent=f"{sd:.2f}" if sd is not None else "",
                              status="digitized_estimate", source=str(digitized_csv))
    output.mkdir(parents=True, exist_ok=True)
    dump(output / "existing_results_sources.json", records)
    write_csv(output / "existing_results.csv", display_records(records))
    return records


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_records(rows):
    return [{key: row[key] for key in ("method", "optimizer_update", "avg_percent", "sd_percent")} for row in rows]


def prepare(args, config, candidates, base_teacher):
    from .opsd_prepare import rademacher, analyze_response
    from .opsd_teacher_prepare import prepare_one
    from .opsd_common import generate_reasoning
    from .modeling import load_tokenizer, load_causal_model, torch_dtype, hidden_size, model_device
    from transformers import set_seed
    import torch

    output = args.output
    candidates_by_uid = {row.uid: row for row in candidates}
    base = {row.uid: row for row in base_teacher}
    teacher_sources = [base]
    for method in ("no_certainty", "no_similarity"):
        path = args.ablation_run / "teacher" / f"{method}_all.jsonl.gz"
        if path.exists():
            cached_source = load_unique(path)
            if set(cached_source) != set(base):
                raise ValueError(f"Original ablation teacher pool differs: {path}")
            teacher_sources.append(cached_source)
    initial = {row.uid: row.metadata[INITIAL]["samples"][:1] for row in candidates}
    model = tokenizer = projection = None

    def runtime():
        nonlocal model, tokenizer, projection
        if model is None:
            tokenizer = load_tokenizer(args.model, config["model"]["trust_remote_code"])
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = load_causal_model(args.model, torch_dtype(config["model"]["dtype"]),
                                      config["model"]["trust_remote_code"]).to("cuda:0").eval()
            projection = rademacher(hidden_size(model), config["representation"]["projection_dim"],
                                    config["seed"], model_device(model))
        return model, tokenizer, projection

    for variant in args.variants:
        rows = base_teacher
        left = initial
        if variant.startswith("last_layer"):
            matrices = representation_tensor(candidates)
            similarity = np.clip(matrices[:, -1] @ matrices[:, -1].T, 0, 1)
            scores = information_scores(similarity, candidates) if variant == "last_layer_certainty" else similarity**2
            contexts, values = ranked_contexts(scores, config["selection"]["context_count"], variant)
            retrieval = output / "retrieval" / f"{variant}.jsonl.gz"
            write_retrieval(candidates, contexts, values, variant, retrieval)
            targets = examples(retrieval)
            path = output / "teacher" / f"{variant}.jsonl.gz"
            cached = load_unique(path)
            # Identical contexts imply identical teacher prompts. Reuse every matching row.
            for index, row in enumerate(targets):
                if row.uid in cached:
                    continue
                matching = next((source[row.uid] for source in teacher_sources
                                 if source[row.uid].demo_ids == row.demo_ids), None)
                if matching is not None:
                    value = copy.deepcopy(matching)
                    value.metadata["opsd_retrieval"] = row.metadata["opsd_retrieval"]
                else:
                    m, tok, proj = runtime()
                    set_seed(config["seed"] + args.teacher_seed_offset + index)
                    value = prepare_one(m, tok, row, [candidates_by_uid[uid] for uid in row.demo_ids],
                                        proj, config, "ours")
                append_jsonl(path, value.to_dict())
                cached[row.uid] = value
                print(f"teacher {variant}: {index + 1}/{len(targets)}", flush=True)
            rows = [cached[row.uid] for row in targets]
        elif variant == "sample_5_5_js":
            path = output / "initial_five.jsonl.gz"
            cached = load_unique(path)
            generation = config["generation"]
            for index, row in enumerate(candidates):
                if row.uid in cached:
                    continue
                m, tok, proj = runtime()
                set_seed(config["seed"] + args.teacher_seed_offset + index)
                prompt, responses = generate_reasoning(m, tok, row, (),
                    max_prompt_tokens=generation["max_prompt_tokens"], max_new_tokens=generation["max_new_tokens"],
                    temperature=generation["temperature"], top_p=generation["top_p"], num_return_sequences=5)
                samples = [analyze_response(m, tok, row, prompt, response, proj, False, config) for response in responses]
                value = copy.deepcopy(row)
                value.metadata["rebuttal_initial_five"] = samples
                append_jsonl(path, value.to_dict())
                cached[row.uid] = value
                print(f"initial five: {index + 1}/{len(candidates)}", flush=True)
            left = {row.uid: cached[row.uid].metadata["rebuttal_initial_five"] for row in candidates}
        selected, beliefs = score_and_select(rows, left, variant, args.selection_seed, config["selection"]["fraction"])
        write_jsonl(output / "subsets" / f"{variant}.jsonl.gz", [row.to_dict() for row in selected])
        write_jsonl(output / "beliefs" / f"{variant}.jsonl.gz", beliefs)


def run_command(*arguments):
    print("[stage] " + " ".join(map(str, arguments)), flush=True)
    subprocess.run([sys.executable, "-u", "-m", *map(str, arguments)], check=True)


def rebuild_source(args, config):
    """Prepare missing generation caches without training any old curve method."""
    candidates_path = args.source_run / "candidates/shared.jsonl.gz"
    candidate_marker = Path(str(candidates_path) + ".done")
    if not candidate_marker.exists():
        run_command("rig.opsd_prepare", "--config", args.config, "--input", args.data,
                    "--output", candidates_path, "--model", args.model, "--resume")
    teacher_path = args.source_run / "teacher/ours_all.jsonl.gz"
    teacher_marker = Path(str(teacher_path) + ".done")
    if not teacher_marker.exists():
        candidates = examples(candidates_path)
        matrices = representation_tensor(candidates)
        vectors = matrices.reshape(len(candidates), -1) / np.sqrt(matrices.shape[1])
        similarity = np.clip(vectors @ vectors.T, 0, 1)
        contexts, values = ranked_contexts(information_scores(similarity, candidates),
                                            config["selection"]["context_count"], "trajectory + certainty")
        retrieval = args.source_run / "retrieval/ours_all.jsonl.gz"
        write_retrieval(candidates, contexts, values, "ours", retrieval)
        run_command("rig.opsd_teacher_prepare", "--config", args.config, "--retrieved", retrieval,
                    "--candidates", candidates_path, "--output", teacher_path,
                    "--method", "ours", "--model", args.model, "--resume")
    dump(args.source_run / "reconstructed_source.json", {
        "model": args.model, "data": str(args.data), "config": config,
        "retrieval": "trajectory + certainty", "purpose": "generation caches for new MC ablations"})


def report(args, reused):
    rows = list(reused)
    for variant in args.variants:
        path = args.output / "eval" / variant / f"update-{args.updates}" / "metrics.json"
        record = {"method": variant, "optimizer_update": args.updates, "avg_percent": "", "sd_percent": "",
                  "status": "pending", "source": str(path)}
        if path.exists():
            overall = json.loads(path.read_text())["overall"]
            key = next(key for key in overall if key.startswith("avg_at_"))
            record.update(avg_percent=f"{100*overall[key]:.2f}", sd_percent=f"{100*overall['rollout_accuracy_std']:.2f}", status="measured")
        rows.append(record)
    dump(args.output / "all_results_sources.json", rows)
    write_csv(args.output / "all_results.csv", display_records(rows))
    endpoint = [row for row in rows if row["optimizer_update"] in {0, args.updates}]
    tex = [r"\begin{tabular}{lrr}", r"\toprule", r"Variant & Avg@3 (\%) & SD (\%) \\", r"\midrule"]
    tex += [row["method"].replace("_", r"\_") + " & " + (row["avg_percent"] or "--") + " & " + (row["sd_percent"] or "--") + r" \\" for row in endpoint]
    tex += [r"\bottomrule", r"\end{tabular}"]
    (args.output / "ablation_table.tex").write_text("\n".join(tex) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("import", "prepare", "run", "report"))
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--ablation-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rebuild-source", action="store_true", help="Reconstruct missing candidate/teacher caches")
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument("--training-updates", type=int, default=500,
                        help="Original curve's full LR-schedule horizon; stop and evaluate at --updates")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--teacher-seed-offset", type=int, default=0)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()
    if args.updates not in range(50, 501, 50):
        parser.error("--updates must be a plotted checkpoint (50,100,...,500)")
    if args.updates > args.training_updates:
        parser.error("Evaluation step cannot exceed the training schedule horizon")
    reused = import_existing(args.source_run, args.ablation_run, args.output, list(range(50, 501, 50)))
    if args.action in {"import", "report"}:
        report(args, reused)
        return
    if args.config is None or args.model is None or args.data is None:
        parser.error("prepare/run require --config, --model and --data")
    config = load_config(args.config)
    source_metrics = args.source_run / "checkpoints/ours/training_metrics.json"
    if source_metrics.exists():
        old = json.loads(source_metrics.read_text())
        horizon = old.get("scheduler_horizon_updates", old.get("optimizer_updates"))
        if horizon is not None and horizon != args.training_updates:
            raise ValueError("--training-updates must match the original curve's scheduler horizon")
    if config["generation"]["num_samples"] != 1 or config["teacher_generation"]["num_samples"] != 5:
        raise ValueError("The source protocol must be 1/5")
    if args.rebuild_source:
        rebuild_source(args, config)
    candidates_path = args.source_run / "candidates/shared.jsonl.gz"
    candidates = examples(candidates_path)
    base_teacher = examples(args.source_run / "teacher/ours_all.jsonl.gz")
    candidates_by_uid = {row.uid: row for row in candidates}
    if len(candidates_by_uid) != len(candidates) or set(candidates_by_uid) != {row.uid for row in base_teacher}:
        raise ValueError("Candidate and teacher pool IDs must match uniquely")
    if any(row.split != "train" for row in candidates + base_teacher):
        raise ValueError("Only training targets may be selected")
    if any(len(row.metadata[TEACHER]["samples"]) != 5 for row in base_teacher):
        raise ValueError("The source teacher cache must have five candidates per question")
    train_ids = {row["uid"] for row in read_jsonl(args.data) if row["split"] == "train"}
    if train_ids != set(candidates_by_uid):
        raise ValueError("DATA must be the original source pool/test file")
    if [row.uid for row in candidates] != [row.uid for row in base_teacher]:
        raise ValueError("Candidate and teacher order must match for the recorded seed schedule")
    identity = {"config": config, "model": args.model, "data": str(args.data.resolve()),
                "source": str(args.source_run.resolve()), "ablation": str(args.ablation_run.resolve()),
                "updates": args.updates, "training_updates": args.training_updates, "selection_seed": args.selection_seed,
                "teacher_seed_offset": args.teacher_seed_offset, "variants": list(args.variants)}
    path = args.output / "experiment.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Run configuration changed; use a new output directory")
    dump(path, identity)
    prepare(args, config, candidates, base_teacher)
    if args.action == "run":
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        for variant in args.variants:
            subset = args.output / "subsets" / f"{variant}.jsonl.gz"
            evaluation = args.output / "eval" / variant / f"update-{args.updates}"
            def signature(row):
                return row.uid, row.question, row.choices, row.choice_labels, row.demo_ids, row.rollout
            selected_signature = [signature(row) for row in examples(subset)]
            matching_metrics = None
            for previous in EXISTING:
                if previous == "original":
                    continue
                root = args.source_run if previous == "ours" else args.ablation_run
                old_subset = root / "subsets" / f"{previous}.jsonl.gz"
                old_metrics = root / "eval" / previous / f"update-{args.updates}" / "metrics.json"
                if old_subset.exists() and old_metrics.exists() and (
                    [signature(row) for row in examples(old_subset)] == selected_signature
                ):
                    matching_metrics = old_metrics
                    break
            if matching_metrics is not None:
                old_metrics = matching_metrics
                dump(evaluation / "metrics.json", json.loads(old_metrics.read_text()))
                dump(evaluation / "reuse.json", {"source": str(old_metrics), "reason": "identical ordered training inputs and schedule"})
                print(f"Reusing original metrics for identical variant {variant}", flush=True)
                continue
            target = args.output / "checkpoints" / variant
            if not (target / "training_metrics.json").exists():
                run_command("rig.opsd_train", "--config", args.config, "--model", args.model,
                            "--selected", subset,
                            "--candidates", candidates_path, "--output-dir", target,
                            "--trajectories", args.training_updates * config["training"]["batch_size"],
                            "--stop-after-updates", args.updates, "--final-checkpoint-only")
            if not (evaluation / "metrics.json").exists():
                run_command("rig.opsd_evaluate", "--config", args.config, "--model", args.model,
                            "--data", args.data, "--adapter", target, "--output", evaluation)
    report(args, reused)


if __name__ == "__main__":
    main()
