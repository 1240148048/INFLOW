from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--original-metrics", required=True)
    for name in ("first", "standard", "selected", "positive"):
        parser.add_argument(f"--{name}-metrics", required=True)
        parser.add_argument(f"--{name}-training", required=True)
        parser.add_argument(f"--{name}-training-seconds", type=float, required=True)
    parser.add_argument("--weighted-selection", required=True)
    parser.add_argument("--positive-selection", required=True)
    parser.add_argument("--preparation-seconds", type=float, required=True)
    parser.add_argument("--selection-seconds", type=float, required=True)
    args = parser.parse_args()
    methods = {
        "original": {
            "self_distillation_seconds": 0.0,
            "evaluation": read_json(args.original_metrics)["overall"],
        }
    }
    labels = {
        "first": "first_sample_all_500",
        "standard": "minimum_entropy_all_500",
        "selected": "reliability_times_information_100",
        "positive": "positive_reliability_times_information_top_20_percent",
    }
    for name, label in labels.items():
        training_seconds = getattr(args, f"{name}_training_seconds")
        extra_selection = args.selection_seconds if name in {"selected", "positive"} else 0.0
        method = {
            "self_distillation_seconds": (
                args.preparation_seconds + extra_selection + training_seconds
            ),
            "training_wall_seconds": training_seconds,
            "training": read_json(getattr(args, f"{name}_training")),
            "evaluation": read_json(getattr(args, f"{name}_metrics"))["overall"],
        }
        if name == "selected":
            method["selection"] = read_json(args.weighted_selection)
        elif name == "positive":
            method["selection"] = read_json(args.positive_selection)
        methods[label] = method
    report = {
        "reliability_definition": "R = 2 / effective_rank - 1",
        "standard_target": "lowest legal-choice predictive entropy among five samples",
        "shared_preparation_seconds": args.preparation_seconds,
        "shared_selection_seconds": args.selection_seconds,
        "methods": methods,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
