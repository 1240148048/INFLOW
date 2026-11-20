from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    from .paper_baselines import METHODS, PAPER_METHODS

    parser = argparse.ArgumentParser(description="Open-output INFLOW: mixed 500 train / 500 test benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("data", help="Download and separate unlabeled train and held-out evaluation")
    data.add_argument("--task", choices=["topv2", "code", "coqa", "squad", "hotpotqa"], required=True)
    data.add_argument("--output", required=True)
    data.add_argument("--dataset")
    data.add_argument("--revision", default="main")
    data.add_argument("--train-split", default="train")
    data.add_argument("--test-split", default="test")
    data.add_argument("--train-count", type=int, help="Default TOPv2/MBPP 150; CoQA/SQuAD 100. 0 = all.")
    data.add_argument("--test-count", type=int, help="Default TOPv2/MBPP 150; CoQA/SQuAD 100. 0 = all.")
    data.add_argument("--seed", type=int, default=42)
    mixed = sub.add_parser("mix")
    mixed.add_argument("--topv2", help="Include the TOPv2 component in the four-task recipe")
    primary = mixed.add_mutually_exclusive_group(required=True)
    primary.add_argument('--code')
    primary.add_argument('--hotpotqa')
    for task in ("coqa", "squad"):
        mixed.add_argument("--" + task, required=True)
    mixed.add_argument("--output", required=True)
    execution = sub.add_parser("execute-code", help="Explicit Linux process backend for MBPP evaluation")
    execution.add_argument("--samples", required=True)
    execution.add_argument("--tests", required=True)
    execution.add_argument("--output", required=True)
    for name in ("prepare", "select", "train", "predict", "score"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--output", required=True)
        if name != "select":
            command.add_argument("--data", required=True)
        if name in {"prepare", "select"}:
            command.add_argument("--methods", default=",".join(PAPER_METHODS))
        if name in {"train", "predict", "score"}:
            command.add_argument("--method", required=True, choices=METHODS)
        if name in {"predict", "score"}:
            command.add_argument("--update", type=int)
        if name == "score":
            command.add_argument("--execution-results")
    summary = sub.add_parser("report")
    summary.add_argument("--output", required=True)
    plot = sub.add_parser("plot-curve")
    plot.add_argument("--run", nargs="+", required=True, help="One run directory per backbone")
    plot.add_argument("--output", required=True, help="Output stem, without an extension")
    plot.add_argument("--methods", nargs="+", choices=PAPER_METHODS, default=PAPER_METHODS,
                      help="Trainable curves to include. Original is always included.")
    args = parser.parse_args(argv)
    if args.command == "execute-code":
        from .process_execution import evaluate_mbpp
        return evaluate_mbpp(args.samples, args.tests, args.output)
    if args.command == "mix":
        from .data import mix_data
        components = {task: getattr(args, task) for task in ("topv2", "code", "coqa", "squad", "hotpotqa") if getattr(args, task) is not None}
        return mix_data(components, args.output)
    if args.command == "data":
        if args.train_count is None:
            args.train_count = 200 if args.task == 'hotpotqa' else 100 if args.task in {"coqa", "squad"} else 150
        if args.test_count is None:
            args.test_count = 200 if args.task == 'hotpotqa' else 100 if args.task in {"coqa", "squad"} else 150
        from .data import prepare_data
        return prepare_data(args)
    if args.command == "report":
        from .report import report
        return report(args.output)
    if args.command == "plot-curve":
        from .figure2 import plot_curves
        return plot_curves(args.run, args.output, args.methods)
    from ..io import load_config
    config = load_config(args.config)
    if args.command == "prepare":
        from .pipeline import prepare
        return prepare(config, args.data, args.output, methods=args.methods.split(","))
    if args.command == "select":
        from .pipeline import select
        return select(config, args.output, args.methods.split(","))
    if args.command == "train":
        if args.method == "original":
            parser.error("Original is evaluated without training")
        from .training import train
        return train(config, args.data, args.output, args.method)
    from .evaluation import predict, score
    if args.command == "predict":
        return predict(config, args.data, args.output, args.method, update=args.update)
    return score(config, args.data, args.output, args.method, args.execution_results, update=args.update)


if __name__ == "__main__":
    main()
