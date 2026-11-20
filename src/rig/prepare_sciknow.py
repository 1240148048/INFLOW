from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from .io import Example, write_jsonl


def canonical_domain(value: str) -> str:
    lowered = value.lower()
    if "bio" in lowered:
        return "Biology"
    if "chem" in lowered:
        return "Chemistry"
    if "phys" in lowered:
        return "Physics"
    if "material" in lowered:
        return "Material"
    return value


def normalize_choices(raw: Any) -> tuple[list[str], list[str]]:
    if isinstance(raw, dict):
        text = raw.get("text") or raw.get("choices") or []
        labels = raw.get("label") or raw.get("labels") or []
        return list(map(str, text)), list(map(str, labels))
    if isinstance(raw, list):
        return list(map(str, raw)), [
            chr(ord("A") + index) for index in range(len(raw))
        ]
    return [], []


def stable_uid(domain: str, index: int) -> str:
    return f"sciknoweval:{domain}:{index:08d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source",
        help="Local sciknoweval_test_v2.jsonl; skips all network requests.",
    )
    parser.add_argument("--train-per-domain", type=int, default=100)
    parser.add_argument("--test-per-domain", type=int, default=20)
    parser.add_argument("--domain", action="append")
    args = parser.parse_args()
    if args.source:
        source = str(Path(args.source).expanduser().resolve())
        if not Path(source).is_file():
            raise FileNotFoundError(source)
    else:
        try:
            source = hf_hub_download(
                repo_id="hicai-zju/SciKnowEval",
                repo_type="dataset",
                filename="data/v2/sciknoweval_test_v2.jsonl",
                local_files_only=True,
            )
        except Exception as error:
            raise RuntimeError(
                "SciKnowEval is not available in the local Hugging Face cache. "
                "Pass --source /path/to/sciknoweval_test_v2.jsonl."
            ) from error
    allowed = set(args.domain or ["Biology", "Material", "Physics", "Chemistry"])
    grouped: dict[str, list[Example]] = {}
    with Path(source).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            domain = canonical_domain(str(row.get("domain", "unknown")))
            if domain not in allowed:
                continue
            choices, labels = normalize_choices(row.get("choices"))
            grouped.setdefault(domain, []).append(
                Example(
                    uid=stable_uid(domain, index),
                    dataset="sciknoweval",
                    domain=domain,
                    split="",
                    question=str(row["question"]),
                    choices=choices,
                    choice_labels=labels,
                    answer=str(row.get("answerKey", "")) or None,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {"question", "choices", "answerKey"}
                    },
                )
            )
    examples = []
    required = args.train_per_domain + args.test_per_domain
    for domain in sorted(grouped):
        rows = grouped[domain]
        if len(rows) < required:
            raise ValueError(
                f"{domain} has {len(rows)} rows; {required} are required."
            )
        train = rows[: args.train_per_domain]
        test = rows[
            args.train_per_domain :
            args.train_per_domain + args.test_per_domain
        ]
        for example in train:
            example.split = "train"
        for example in test:
            example.split = "test"
        examples.extend(train + test)
        print(
            f"{domain}: first {len(train)} rows -> train, "
            f"next {len(test)} rows -> test"
        )
    write_jsonl(args.output, (example.to_dict() for example in examples))
    print(f"Wrote {len(examples)} rows to {args.output}")


if __name__ == "__main__":
    main()
