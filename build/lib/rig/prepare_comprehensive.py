from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import hf_hub_download

from .io import Example, write_jsonl
from .prepare_sciknow import canonical_domain, normalize_choices


SCIKNOW_COUNTS = {
    "Biology": 50,
    "Chemistry": 50,
    "Material": 50,
    "Physics": 50,
}

MMLU_SPECS = {
    "Math": ("category", "math", 100),
    "Psychology": ("category", "psychology", 40),
    "Global Facts": ("src", "ori_mmlu-global_facts", 40),
    "History": ("category", "history", 40),
    "Economics": ("category", "economics", 40),
    "Law": ("category", "law", 40),
}


def load_sciknow(path: str | None) -> list[dict[str, Any]]:
    if path:
        source = Path(path).expanduser().resolve()
    else:
        source = Path(
            hf_hub_download(
                repo_id="hicai-zju/SciKnowEval",
                repo_type="dataset",
                filename="data/v2/sciknoweval_test_v2.jsonl",
                local_files_only=True,
            )
        )
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                row["_source_index"] = source_index
                rows.append(row)
    return rows


def load_mmlu(path: str | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "MMLU-Pro preparation requires the `datasets` package. "
            "Install it once with `python -m pip install datasets`."
        ) from error

    if path:
        source = Path(path).expanduser().resolve()
        if source.is_dir():
            files = sorted(source.rglob("*test*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"No test parquet file found below {source}"
                )
            dataset = load_dataset(
                "parquet",
                data_files=[str(file) for file in files],
                split="train",
            )
        elif source.suffix == ".parquet":
            dataset = load_dataset(
                "parquet", data_files=str(source), split="train"
            )
        elif source.suffix in {".json", ".jsonl", ".gz"}:
            dataset = load_dataset(
                "json", data_files=str(source), split="train"
            )
        else:
            raise ValueError(f"Unsupported MMLU-Pro source: {source}")
    else:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    return [dict(row) for row in dataset]


def split_ordered(
    rows: list[Any], count: int, name: str
) -> tuple[list[Any], list[Any]]:
    required = 2 * count
    if len(rows) < required:
        raise ValueError(
            f"{name} has {len(rows)} matching rows, but {required} are "
            "required for non-overlapping train/test splits."
        )
    return rows[:count], rows[count:required]


def sciknow_examples(rows: Iterable[dict[str, Any]]) -> list[Example]:
    grouped: dict[str, list[dict[str, Any]]] = {
        domain: [] for domain in SCIKNOW_COUNTS
    }
    for row in rows:
        domain = canonical_domain(str(row.get("domain", "unknown")))
        if domain in grouped:
            grouped[domain].append(row)

    output = []
    for domain, count in SCIKNOW_COUNTS.items():
        ordered = sorted(grouped[domain], key=lambda row: row["_source_index"])
        train, test = split_ordered(ordered, count, f"SciKnowEval/{domain}")
        for split, selected in (("train", train), ("test", test)):
            for row in selected:
                choices, labels = normalize_choices(row.get("choices"))
                source_index = int(row["_source_index"])
                output.append(
                    Example(
                        uid=f"sciknoweval:{domain}:{source_index:08d}",
                        dataset="sciknoweval",
                        domain=domain,
                        split=split,
                        question=str(row["question"]),
                        choices=choices,
                        choice_labels=labels,
                        answer=str(row.get("answerKey", "")) or None,
                        metadata={"source_index": source_index},
                    )
                )
        print(f"SciKnowEval/{domain}: train={count}, test={count}")
    return output


def mmlu_examples(rows: list[dict[str, Any]]) -> list[Example]:
    output = []
    for domain, (field, value, count) in MMLU_SPECS.items():
        matching = [
            row
            for row in rows
            if str(row.get(field, "")).lower() == value.lower()
        ]
        ordered = sorted(matching, key=lambda row: int(row["question_id"]))
        train, test = split_ordered(ordered, count, f"MMLU-Pro/{domain}")
        for split, selected in (("train", train), ("test", test)):
            for row in selected:
                question_id = int(row["question_id"])
                options = [str(option) for option in row["options"]]
                labels = [
                    chr(ord("A") + index) for index in range(len(options))
                ]
                output.append(
                    Example(
                        uid=f"mmlu_pro:{domain}:{question_id:08d}",
                        dataset="mmlu_pro",
                        domain=domain,
                        split=split,
                        question=str(row["question"]),
                        choices=options,
                        choice_labels=labels,
                        answer=str(row["answer"]).upper(),
                        metadata={
                            "question_id": question_id,
                            "category": str(row.get("category", "")),
                            "src": str(row.get("src", "")),
                        },
                    )
                )
        print(
            f"MMLU-Pro/{domain}: filter {field}={value}, "
            f"available={len(ordered)}, train={count}, test={count}"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic 500-train/500-test benchmark."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sciknow-source")
    parser.add_argument("--mmlu-source")
    args = parser.parse_args()

    examples = sciknow_examples(load_sciknow(args.sciknow_source))
    examples.extend(mmlu_examples(load_mmlu(args.mmlu_source)))
    train_count = sum(example.split == "train" for example in examples)
    test_count = sum(example.split == "test" for example in examples)
    if (train_count, test_count) != (500, 500):
        raise AssertionError(
            f"Expected train=500/test=500, got {train_count}/{test_count}."
        )
    write_jsonl(args.output, (example.to_dict() for example in examples))
    print(f"Wrote train={train_count}, test={test_count} to {args.output}")


if __name__ == "__main__":
    main()
