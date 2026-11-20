from __future__ import annotations

import ast
import builtins
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl, write_jsonl


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stratified(rows: list[dict], count: int, seed: int) -> list[dict]:
    if count == 0:
        return rows
    if count < 1 or count > len(rows):
        raise ValueError(f"Requested {count} rows from a pool of {len(rows)}")
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("domain", "code")].append(row)
    rng = random.Random(seed)
    quotas = {key: int(count * len(group) / len(rows)) for key, group in groups.items()}
    remainder = count - sum(quotas.values())
    ordered = sorted(groups, key=lambda key: (-(count * len(groups[key]) / len(rows) - quotas[key]), key))
    for key in ordered[:remainder]:
        quotas[key] += 1
    result = []
    for key, group in sorted(groups.items()):
        group = group.copy()
        rng.shuffle(group)
        result.extend(group[:quotas[key]])
    rng.shuffle(result)
    return result


def validate_pool(rows: list[dict]):
    if not rows or len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("Training pool is empty or contains duplicate IDs")
    for row in rows:
        if row.get("split") != "train" or row.get("task") not in {"topv2", "code", "coqa", "squad", "hotpotqa"}:
            raise ValueError("Only training rows may enter preparation/distillation")
        if set(row) - {"uid", "task", "split", "question", "domain", "context", "query", "history_uids", "conversation_id"}:
            raise ValueError("Training input must not contain gold labels, solutions or tests")


def public_interface(tests):
    """Expose callable name and input expressions, never assertion outputs."""
    calls = []
    for test in tests:
        tree = ast.parse(test)
        candidates = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        call = next((node for node in candidates if node.func.id not in vars(builtins)), candidates[0] if candidates else None)
        if call is None:
            raise ValueError("Cannot derive a callable interface from this MBPP test")
        calls.append(call)
    names = {call.func.id for call in calls}
    if len(names) != 1:
        raise ValueError(f"Ambiguous MBPP callable interface: {names}")
    return calls[0].func.id, [ast.unparse(call) for call in calls]


def prepare_data(args):
    from datasets import load_dataset
    destination = Path(args.output)
    request = {key: getattr(args, key) for key in ("task", "dataset", "revision", "train_split", "test_split", "train_count", "test_count", "seed")}
    if args.task == 'hotpotqa':
        request['sampling_protocol'] = 'train-medium-disjoint-v1'
    if (destination / "data_manifest.json").exists():
        manifest = json.loads((destination / "data_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("request") != request:
            raise ValueError("Data preparation arguments changed; use a new output directory")
        read_pool(destination)
        if not (destination / "eval.jsonl").exists() or not (destination / "schema.json").exists():
            raise ValueError("Cached data files are incomplete")
        print(f"Reusing prepared data in {destination}")
        return
    if args.task == "topv2":
        if args.train_split == args.test_split:
            raise ValueError("Training and test splits must differ")
        ds = load_dataset(args.dataset or "WillHeld/top_v2", revision=args.revision)
        if args.train_split not in ds or args.test_split not in ds:
            raise ValueError(f"Available splits: {list(ds)}; specify --train-split and --test-split explicitly")
        train = [
            {"uid": f"topv2:train:{i}", "task": "topv2", "split": "train", "question": row["utterance"], "domain": row["domain"]}
            for i, row in enumerate(ds[args.train_split])
        ]
        # Only the global task ontology is extracted, never per-question gold outputs.
        labels = sorted(set(label for row in ds[args.train_split] for label in re.findall(r"\[((?:IN|SL):[A-Z0-9_]+)", row["semantic_parse"])))
        evaluation = [
            {"uid": f"topv2:test:{i}", "task": "topv2", "split": "test", "question": row["utterance"], "domain": row["domain"], "reference": row["semantic_parse"]}
            for i, row in enumerate(ds[args.test_split])
        ]
        schema = {"labels": labels, "source": "global label inventory of training split; no gold examples"}
        source = {"dataset": args.dataset or "WillHeld/top_v2", "train_split": args.train_split, "test_split": args.test_split, "revision": args.revision}
    elif args.task == "code":
        ds = load_dataset("google-research-datasets/mbpp", "full", revision=args.revision)
        train = [
            {"uid": f"mbpp:train:{row['task_id']}", "task": "code", "split": "train", "question": row["text"], "domain": "code"}
            for row in ds["train"]
        ]
        evaluation = []
        private_tests = {}
        for row in ds["test"]:
            entry, examples = public_interface(row["test_list"])
            uid = f"mbpp:test:{row['task_id']}"
            question = row["text"] + "\nRequired callable: " + entry + ".\nInput-only interface examples (no expected outputs):\n" + "\n".join(examples)
            evaluation.append({"uid": uid, "task": "code", "split": "test", "question": question, "entry_point": entry, "domain": "code"})
            private_tests[uid] = {"uid": uid, "tests": list(row["test_list"]), "setup": row.get("test_setup_code", "")}
        schema = {}
        source = {"dataset": "google-research-datasets/mbpp", "config": "full", "train_split": "train", "test_split": "test", "revision": args.revision, "eval_interface": "name and input expressions only, extracted from test AST; expected outputs withheld"}
    elif args.task == 'hotpotqa':
        from .hotpot import medium_rows
        name = args.dataset or 'hotpotqa/hotpot_qa'
        print('Loading HotpotQA distractor/train; sampling disjoint medium train/test questions...', flush=True)
        local_json = os.environ.get('HOTPOT_TRAIN_JSON')
        if local_json:
            from .hotpot import load_training_json
            print(f'Reading local HotpotQA training JSON: {local_json}', flush=True)
            ds = load_training_json(local_json)
        else:
            ds = load_dataset(name, 'distractor', split='train', revision=args.revision)
        train, evaluation, available = medium_rows(ds, args.train_count, args.test_count, args.seed)
        schema = {}
        source = {'dataset': name, 'config': 'distractor', 'revision': args.revision,
                  'source_split': 'train', 'level': 'medium', 'available_unique_questions': available,
                  'evaluation_split': 'custom disjoint holdout from train-medium; not official dev/test',
                  'context': 'all supplied paragraphs, original order; no supporting-fact filtering',
                  'metric': 'HotpotQA answer EM/F1; no supporting-fact evaluation'}
        if local_json:
            source['local_training_json'] = str(Path(local_json).resolve())
    elif args.task == "squad":
        from .qa import question
        name = args.dataset or "rajpurkar/squad"
        ds = load_dataset(name, revision=args.revision)
        def convert(row, split):
            item = {"uid": f"squad:{split}:{row['id']}", "task": "squad", "split": split, "question": question(row["context"], row["question"]), "domain": "reading"}
            if split == "test":
                item["references"] = list(row["answers"]["text"])
            return item
        train = [convert(row, "train") for row in ds["train"]]
        evaluation = [convert(row, "test") for row in ds["validation"]]
        schema = {}
        source = {"dataset": name, "revision": args.revision, "train_split": "train", "held_out_split": "validation", "metric": "normalized EM and max-reference token F1"}
    else:
        from .qa import coqa_rows
        name = args.dataset or "stanfordnlp/coqa"
        ds = load_dataset(name, revision=args.revision)
        train = coqa_rows(ds["train"], "train", args.train_count, args.seed)
        evaluation = coqa_rows(ds["validation"], "test", args.test_count, args.seed + 1)
        schema = {}
        source = {"dataset": name, "revision": args.revision, "train_split": "train", "held_out_split": "validation", "sampling_unit": "QA turn; chronological prefixes of shuffled conversations", "history": "model-generated, not gold", "metric": "single-reference EM/F1 using HF input_text; not official multi-annotator CoQA leaderboard score"}
    if args.task not in {"coqa", "hotpotqa"}:
        evaluation = stratified(evaluation, args.test_count, args.seed + 1)
    # Drop exact cross-split duplicate inputs before any training; preserve test set.
    test_text = {" ".join(row["question"].split()).casefold() for row in evaluation}
    before = len(train)
    train = [row for row in train if " ".join(row["question"].split()).casefold() not in test_text]
    removed = before - len(train)
    if args.task not in {"coqa", "hotpotqa"}:
        train = stratified(train, args.train_count, args.seed)
    elif args.task == "coqa" and args.train_count and len(train) != args.train_count:
        raise ValueError("CoQA overlap removal changed the requested count; use another fixed seed")
    validate_pool(train)
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "eval.jsonl", evaluation)
    if args.task == "code":
        write_jsonl(destination / "eval_tests.jsonl", [private_tests[row["uid"]] for row in evaluation])
    dump(destination / "schema.json", schema)
    dump(destination / "data_manifest.json", {"task": args.task, "request": request, "source": source, "seed": args.seed, "train_count": len(train), "eval_count": len(evaluation), "exact_overlap_removed": removed, "full_test": args.test_count == 0, "gold_in_train": False})
    print(f"Prepared {len(train)} unlabeled training / {len(evaluation)} evaluation rows in {destination}")


def read_pool(directory: Path):
    rows = list(read_jsonl(directory / "train.jsonl"))
    validate_pool(rows)
    return rows


def mix_data(components, output):
    destination = Path(output)
    train, evaluation, tests, schema, sources = [], [], [], {}, {}
    expected = (
        {"topv2": 150, "code": 150, "coqa": 100, "squad": 100}
        if "topv2" in components else {("hotpotqa" if "hotpotqa" in components else "code"): 200, "coqa": 150, "squad": 150}
    )
    if set(components) != set(expected):
        raise ValueError(f"Expected mixed components {sorted(expected)}")
    for task, directory in components.items():
        directory = Path(directory)
        pool = read_pool(directory)
        held_out = list(read_jsonl(directory / "eval.jsonl"))
        if len(pool) != expected[task] or len(held_out) != expected[task] or any(row["task"] != task for row in pool + held_out):
            raise ValueError(f"Expected {expected[task]} train/test rows for {task}")
        train.extend(pool)
        evaluation.extend(held_out)
        sources[task] = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
        if task == "topv2":
            schema.update(json.loads((directory / "schema.json").read_text(encoding="utf-8")))
        if task == "code":
            tests.extend(read_jsonl(directory / "eval_tests.jsonl"))
    manifest = {"task": "mixed", "train_count": 500, "eval_count": 500, "counts_per_split": expected, "source_components": sources, "gold_in_train": False}
    if (destination / "data_manifest.json").exists():
        if json.loads((destination / "data_manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Mixed sources changed; use a new directory")
        return
    validate_pool(train)
    write_jsonl(destination / "train.jsonl", train)
    write_jsonl(destination / "eval.jsonl", evaluation)
    write_jsonl(destination / "eval_tests.jsonl", tests)
    dump(destination / "schema.json", schema)
    dump(destination / "data_manifest.json", manifest)
