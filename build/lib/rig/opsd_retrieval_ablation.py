from __future__ import annotations

import argparse
from pathlib import Path

from .io import Example, read_jsonl
from .opsd_select import (
    certainty_only_contexts,
    similarity_only_contexts,
    write_retrieval,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-count", type=int, default=3)
    args = parser.parse_args()

    examples = [Example.from_dict(row) for row in read_jsonl(args.input)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    no_certainty, no_certainty_scores = similarity_only_contexts(
        examples, args.context_count
    )
    no_similarity, no_similarity_scores = certainty_only_contexts(
        examples, args.context_count
    )
    write_retrieval(
        examples,
        no_certainty,
        no_certainty_scores,
        "no_certainty",
        output / "no_certainty_all.jsonl.gz",
    )
    write_retrieval(
        examples,
        no_similarity,
        no_similarity_scores,
        "no_similarity",
        output / "no_similarity_all.jsonl.gz",
    )


if __name__ == "__main__":
    main()
