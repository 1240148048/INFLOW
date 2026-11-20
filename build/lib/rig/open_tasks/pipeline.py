from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from ..io import append_jsonl, read_jsonl, write_jsonl
from .beliefs import belief_scores, joint_cluster, predictive_certainty, weighted_distribution
from .data import dump, read_pool
from .runtime import Runtime, messages, output_contract
from .paper_baselines import METHODS, PAPER_METHODS, family, paper_contexts, redundancy_subset

PROTOCOL = "open_inflow_joint_1_5_js_only_clustering_v8"


def experiment(config, data_dir, output):
    data_dir, output = Path(data_dir), Path(output)
    rows = read_pool(data_dir)
    schema = json.loads((data_dir / "schema.json").read_text(encoding="utf-8"))
    identity = {"protocol": PROTOCOL, "config": config, "training_rows": rows, "schema": schema}
    path = output / "experiment.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError("Existing experiment uses different data/config. Use a new output directory.")
    else:
        dump(path, identity)
    return rows, schema


def records(path):
    if not path.exists():
        return {}
    rows = list(read_jsonl(path))
    if len({row["uid"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate cached records in {path}")
    return {row["uid"]: row for row in rows}


def preparation_metrics(output, ordered, contextual, config, retrieval_seconds):
    values = list(contextual.values())
    dump(output / "preparation_metrics.json", {
        "initial_seconds": sum(row["initial_seconds"] for row in ordered),
        "retrieval_seconds": retrieval_seconds,
        "teacher_and_grouping_seconds": sum(row["seconds"] for row in values),
        "pool_size": len(ordered), "initial_samples": 1, "retrieved_samples": 5, "semantic_clustering": "joint six-answer LLM partition",
        "initial_valid_fraction": sum(key is not None for row in ordered for key in row["keys"]) / sum(len(row["keys"]) for row in ordered),
        "contextual_valid_fraction": sum(key is not None for row in values for key in row["keys"]) / sum(len(row["keys"]) for row in values),
        "js_near_zero_fraction": sum(row["scores"]["js"] < 1e-8 for row in values) / len(values),
        "js_near_max_fraction": sum(row["scores"]["js"] > math.log(2) - 1e-8 for row in values) / len(values),
        "note": "Initial timing includes one sample, representation and predictive entropy. Contextual timing includes five samples, predictive entropy and joint six-answer LLM clustering. Model loading is excluded.",
    })


def retrieve(initial, config):
    import numpy as np
    count = config["selection"]["context_count"]
    if not 0 < count < len(initial):
        raise ValueError("Retrieval depth must be positive and smaller than the pool")
    matrices = np.asarray([row["trajectory"] for row in initial], dtype=np.float64)
    matrices /= np.maximum(np.linalg.norm(matrices, axis=2, keepdims=True), 1e-12)
    if config["representation"].get("mode", "trajectory") == "final_layer":
        vectors = matrices[:, -1]
    else:
        vectors = matrices.reshape(len(initial), -1) / math.sqrt(matrices.shape[1])
    similarity = np.clip(vectors @ vectors.T, 0, 1)
    confidence = np.array([row["certainty"] for row in initial])
    if not config["representation"].get("use_certainty", True):
        confidence = np.ones_like(confidence)
    score = -0.5 * np.log1p(-np.minimum(similarity**2 * confidence[None, :], 1 - 1e-6))
    np.fill_diagonal(score, -np.inf)
    result = {}
    for i, row in enumerate(initial):
        eligible = [j for j, source in enumerate(initial) if j != i and output_contract(source) == output_contract(row)]
        if len(eligible) < count:
            raise ValueError("Each output-contract pool must have more training rows than retrieval depth")
        indices = sorted(eligible, key=lambda j: (-score[i, j], initial[j]["uid"]))[:count]
        # Strongest source appears nearest to the target question.
        result[row["uid"]] = [initial[j]["uid"] for j in reversed(indices)]
    return result


def teacher_path(output, name):
    return output / ("teacher.jsonl" if name == "js" else f"teacher.{name}.jsonl")


def modal_index(keys, entropies):
    counts = Counter(key for key in keys if key is not None)
    valid = [i for i, key in enumerate(keys) if key is not None]
    return min(valid, key=lambda i: (-counts[keys[i]], entropies[i], i)) if valid else 0


def prepare(config, data_dir, output, runtime_factory=Runtime, methods=PAPER_METHODS):
    output = Path(output)
    pool, schema = experiment(config, data_dir, output)
    source_by_uid = {row["uid"]: row for row in pool}
    if config["generation"]["num_samples"] != 1 or config["teacher_generation"]["num_samples"] != 5:
        raise ValueError("The shared protocol requires exactly one initial and five retrieved samples")
    for key in ("temperature", "top_p"):
        if config["generation"].get(key) != config["teacher_generation"].get(key):
            raise ValueError("Initial and retrieved answers must use the same decoding configuration")
    requested = tuple(dict.fromkeys(family(method) for method in methods))
    needs_shared_initial = any(name != "consistency" for name in requested)
    initial = records(output / "initial.jsonl") if needs_shared_initial else {}
    if not methods or any(m not in METHODS or m == "original" for m in methods):
        raise ValueError("Preparation requires one or more trainable methods")
    expected = {row["uid"] for row in pool}
    caches = {name: records(teacher_path(output, name)) for name in requested}
    if set(initial) - expected or any(set(cache) - expected for cache in caches.values()):
        raise ValueError("Unexpected cached IDs")
    if all(set(cache) == expected for cache in caches.values()):
        if needs_shared_initial and len(initial) != len(pool):
            raise ValueError("Teacher cache is complete but initial cache is incomplete")
        print("Reusing completed preparation")
        return
    load_started = time.perf_counter()
    runtime = runtime_factory(config)
    load_path = output / "preparation_loading.json"
    if not load_path.exists():
        dump(load_path, {"seconds": time.perf_counter() - load_started})
    generation = config["generation"]
    prompt_version = config.get("preparation_prompt_version")
    judge_budget = config.get("semantic_judge", {}).get("max_prompt_tokens", 16384)
    for index, source in enumerate(pool if needs_shared_initial else []):
        uid = source["uid"]
        if uid in initial:
            continue
        started = time.perf_counter()
        row = dict(source)
        if source["task"] == "coqa":
            from .qa import question
            history = [(initial[past]["query"], initial[past]["demonstration"]) for past in source["history_uids"]]
            row["question"] = question(source["context"], source["query"], history)
        texts, tokens, prompt_ids = runtime.sample(messages(row, schema, prompt_version=prompt_version), generation["num_samples"], config["seed"] + 100_000 + index, generation)
        sampled = time.perf_counter()
        entropies = runtime.predictive_entropies(prompt_ids, tokens)
        keys = ["class:0" if texts[0].strip() else None]
        representatives = list(texts)
        distribution = weighted_distribution(keys, entropies)
        best = 0
        grouped = time.perf_counter()
        trajectory = runtime.trajectory(prompt_ids, tokens[best])
        finished = time.perf_counter()
        row.update({"responses": texts, "keys": keys, "representatives": representatives,
                    "predictive_entropies": entropies, "distribution": distribution,
                    "certainty": predictive_certainty(entropies[0]),
                    "initial_response": texts[0], "initial_predictive_entropy": entropies[0],
                    "certainty_definition": "exp(-mean full-vocabulary predictive entropy)",
                    "trajectory": trajectory,
                    "source_prompt_ids": prompt_ids, "source_response_ids": tokens[best],
                    "demonstration": texts[best], "initial_seconds": finished - started,
                    "timing": {"generation_seconds": sampled - started, "belief_seconds": grouped - sampled,
                               "feature_seconds": finished - grouped}})
        initial[uid] = row
        append_jsonl(output / "initial.jsonl", row)
        print(f"initial 1/1 {index + 1}/{len(pool)}: {uid}", flush=True)
    ordered = [initial[row["uid"]] for row in pool] if needs_shared_initial else pool
    for name, contextual in caches.items():
        if set(contextual) == expected:
            continue
        retrieval_started = time.perf_counter()
        previous_feature_seconds = 0.0
        neurons = None
        if name == "neuron":
            feature_path = output / "neuron_sources.jsonl"
            cached = records(feature_path)
            neurons = {}
            for row in ordered:
                uid = row["uid"]
                if uid not in cached:
                    feature_started = time.perf_counter()
                    features = runtime.neuron_features(row["source_prompt_ids"], row["source_response_ids"])
                    cached[uid] = {"uid": uid, "neurons": features, "seconds": time.perf_counter() - feature_started}
                    append_jsonl(feature_path, cached[uid])
                else:
                    previous_feature_seconds += cached[uid].get("seconds", 0.0)
                neurons[uid] = cached[uid]["neurons"]
        contexts = ({uid: (ids, 0.) for uid, ids in retrieve(ordered, config).items()} if name == "js"
                    else paper_contexts(ordered, config["selection"]["context_count"], name, neurons))
        dump(output / f"retrieval.{name}.json", contexts)
        retrieval_path = output / f"retrieval.{name}.timing.json"
        if retrieval_path.exists():
            retrieval_seconds = json.loads(retrieval_path.read_text())["seconds"]
        else:
            retrieval_seconds = time.perf_counter() - retrieval_started + previous_feature_seconds
            dump(retrieval_path, {"seconds": retrieval_seconds})
        for index, row in enumerate(ordered):
            uid = row["uid"]
            if uid in contextual:
                continue
            started = time.perf_counter()
            demo_ids, retrieval_gain = contexts[uid]
            demos = [initial[demo] for demo in demo_ids]
            feedback = None  # Consistency preserves the original no-retrieval protocol.
            effective = row
            if name == "consistency":
                effective = dict(source_by_uid[uid])
                if effective["task"] == "coqa":
                    from .qa import question
                    history = [(source_by_uid[past]["query"], contextual[past]["demonstration"])
                               for past in effective["history_uids"]]
                    effective["question"] = question(effective["context"], effective["query"], history)
            chat = messages(effective, schema, demos, self_feedback=feedback, prompt_version=prompt_version)
            texts, tokens, prompt_ids = runtime.sample(chat, 5, config["seed"] + 200_000 + index, config["teacher_generation"])
            sampled = time.perf_counter()
            entropies = runtime.predictive_entropies(prompt_ids, tokens)
            uses_initial = name != "consistency"
            partition = None
            initial_distribution = None
            joint_answers = None
            if name == "js":
                joint_answers = [row["initial_response"], *texts]
                try:
                    partition = joint_cluster(runtime, effective, joint_answers, judge_budget)
                except ValueError as error:
                    dump(output / "clustering_failures" / name / f"{index:05d}.json",
                         {"uid": uid, "question": effective["question"], "joint_responses": joint_answers, "error": str(error)})
                    raise
                keys = partition["keys"][1:]
                initial_keys = partition["keys"][:1]
                initial_distribution = weighted_distribution(initial_keys, [row["initial_predictive_entropy"]])
                distribution = weighted_distribution(keys, entropies)
            else:
                from .qa import answer_keys
                # Baselines compare normalized answer strings, without an LLM judge.
                keys = answer_keys(texts)
                distribution = None
            record = {"uid": uid, "demo_ids": demo_ids, "responses": texts,
                      "keys": keys if name == "js" else None,
                      "answer_vote_keys": keys if name != "js" else None,
                      "representatives": partition["representatives"] if partition else None,
                      "predictive_entropies": entropies, "distribution": distribution,
                      "self_feedback": feedback,
                      "initial_response": row["initial_response"] if uses_initial else None,
                      "retrieved_responses": texts if uses_initial else None,
                      "independent_responses": texts if not uses_initial else None,
                      "initial_predictive_entropy": row["initial_predictive_entropy"] if uses_initial else None,
                      "question": effective["question"], "uses_shared_initial": uses_initial,
                      "demonstration": texts[modal_index(keys, entropies)],
                      "retrieved_predictive_entropies": entropies if uses_initial else None,
                      "independent_predictive_entropies": entropies if not uses_initial else None,
                      "initial_distribution": initial_distribution,
                      "joint_responses": joint_answers,
                      "joint_class_ids": partition["keys"] if partition else None,
                      "joint_clusters": partition["groups"] if partition else None,
                      "clustering": partition}
            if name == "js":
                record["scores"] = belief_scores({**row, "keys": initial_keys, "distribution": initial_distribution}, record)
            elif name == "consistency":
                votes = Counter(key for key in keys if key is not None)
                record["scores"] = {"consistency": max(votes.values(), default=0) / len(keys)}
            elif name == "mutual":
                record["scores"] = {"mutual": retrieval_gain}
            else:
                record["scores"] = {}
            grouped = time.perf_counter()
            if name == "full":
                representative = modal_index(keys, entropies)
                record["final_representation"] = runtime.trajectory(prompt_ids, tokens[representative])[-1]
            if name == "neuron":
                sizes = [len(runtime.neuron_features(prompt_ids, token_ids)) for token_ids in tokens]
                record["scores"]["neuron"] = -sum(sizes) / len(sizes)
            finished = time.perf_counter()
            record["seconds"] = finished - started
            record["timing"] = {"generation_seconds": sampled - started, "belief_seconds": grouped - sampled,
                                "feature_seconds": finished - grouped}
            contextual[uid] = record
            append_jsonl(teacher_path(output, name), record)
            detail = " + joint 6-answer clustering" if name == "js" else " (no LLM clustering)"
            print(f"teacher {name} 5/5{detail} {index + 1}/{len(pool)}: {uid}", flush=True)
        if name == "js":
            preparation_metrics(output, ordered, contextual, config, retrieval_seconds)
        dump(output / f"preparation.{name}.json", {
            "retrieval_seconds": retrieval_seconds, "teacher_seconds": sum(v["seconds"] for v in contextual.values()),
            "questions": len(contextual), "family": name, "uses_shared_initial": name != "consistency",
            "llm_semantic_clustering": name == "js",
        })


def select(config, output, methods):
    output = Path(output)
    identity = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    source_by_uid = {row["uid"]: row for row in identity["training_rows"]}
    if identity.get("protocol") != PROTOCOL or identity["config"] != config:
        raise ValueError("Selection requires preparation with the current protocol and config")
    initial = records(output / "initial.jsonl") if any(method != "consistency" for method in methods) else {}
    expected = {row["uid"] for row in identity["training_rows"]}
    if any(method != "consistency" for method in methods) and set(initial) != expected:
        raise ValueError("Preparation is incomplete")
    count = max(1, math.floor(len(expected) * config["selection"]["fraction"] + 0.5))
    if not 0 < config["selection"]["fraction"] <= 1:
        raise ValueError("Selection fraction must be in (0,1]")
    manifest = {}
    caches = {}
    for method in methods:
        selection_started = time.perf_counter()
        if method not in METHODS or method == "original":
            raise ValueError(f"Unknown selector: {method}")
        name = family(method)
        if name not in caches:
            caches[name] = records(teacher_path(output, name))
        teacher = caches[name]
        if set(teacher) != expected:
            raise ValueError(f"Teacher preparation is incomplete for {method}")
        order = sorted(expected)
        random.Random(config["seed"]).shuffle(order)
        if method == "redundancy":
            order = redundancy_subset(order, teacher, count)
        elif method not in {"full", "random"}:
            order.sort(key=lambda uid: -teacher[uid]["scores"][method])
        selected = order if method == "full" else order[:count]
        rows = []
        for uid in selected:
            source = source_by_uid[uid] if method == "consistency" else initial[uid]
            row = {k: v for k, v in source.items() if k not in {"source_prompt_ids", "source_response_ids", "trajectory", "responses", "keys", "representatives", "predictive_entropies", "distribution"}}
            for field in ("initial_response", "retrieved_responses", "initial_predictive_entropy",
                          "retrieved_predictive_entropies", "initial_distribution", "joint_responses",
                          "joint_class_ids", "joint_clusters"):
                row[field] = teacher[uid][field]
            row["question"] = teacher[uid]["question"]
            row["independent_responses"] = teacher[uid].get("independent_responses")
            row["independent_predictive_entropies"] = teacher[uid].get("independent_predictive_entropies")
            row["uses_shared_initial"] = method != "consistency"
            row["retrieved_distribution"] = teacher[uid]["distribution"]
            row["demo_ids"] = teacher[uid]["demo_ids"]
            row["selector"] = method
            row["scores"] = teacher[uid]["scores"]
            row["self_feedback"] = teacher[uid]["self_feedback"]
            rows.append(row)
        write_jsonl(output / "subsets" / f"{method}.jsonl", rows)
        manifest[method] = {"count": len(rows), "pool": len(expected), "teacher_family": name, "counts_by_task": dict(Counter(row["task"] for row in rows)), "tie_break": "stable sort after seed-fixed shuffle", "seconds": time.perf_counter() - selection_started}
        manifest[method]["selection_scope"] = "global"
        print(f"selected {method}: {len(rows)}/{len(expected)}, tasks={manifest[method]['counts_by_task']}", flush=True)
    dump(output / "selection_manifest.json", manifest)
