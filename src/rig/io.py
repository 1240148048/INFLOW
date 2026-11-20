from __future__ import annotations

import gzip
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Example:
    uid: str
    dataset: str
    domain: str
    split: str
    question: str
    choices: list[str] = field(default_factory=list)
    choice_labels: list[str] = field(default_factory=list)
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rollout: str | None = None
    demo_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Example":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in row.items() if key in known}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        if source.suffix == ".json":
            return json.load(handle)
        import yaml

        return yaml.safe_load(handle)


def _text_reader(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gz.tmp"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _text_writer(path: Path, mode: str):
    if path.suffix == ".gz" or path.name.endswith(".gz.tmp"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with _text_reader(Path(path)) as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with _text_writer(temporary, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _text_writer(output, "a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic_split(
    examples: list[Example], train_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    indices = list(range(len(examples)))
    random.Random(seed).shuffle(indices)
    cut = int(len(indices) * train_fraction)
    train_indices = set(indices[:cut])
    train, test = [], []
    for index, example in enumerate(examples):
        target = train if index in train_indices else test
        example.split = "train" if index in train_indices else "test"
        target.append(example)
    return train, test
