"""Portable entry points for the three experiment families in this repository.

This module only resolves paths/configuration and invokes the existing pipelines.
It imports no model libraries, so --help and --dry-run need no GPU or weights.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MC_METHODS = "full,redundancy,consistency,neuron,ours,mutual"
OPEN_METHODS = "full,redundancy,consistency,neuron,mutual,js"
EPOCH_METHODS = "random,difficulty,uncertainty,redundancy,activation,full,ours-inverse,ours"
CONFIGS = {"mc": "neuron_opsd_single_gpu.json", "open": "open_mixed_autodl.json",
           "mc-epochs": "single_gpu_48gb.json"}


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="experiment", required=True)
    for name, description in (
        ("mc", "Multiple-choice on-policy distillation and update curves"),
        ("open", "HotpotQA 200 + CoQA 150 + SQuAD 150 mixed self-distillation"),
        ("mc-epochs", "Multiple-choice cached-answer training and epoch log-loss curves"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--config", type=Path, default=os.environ.get("CONFIG"))
        command.add_argument("--model", default=os.environ.get("MODEL"),
                             help="Downloaded model directory, or a fully cached model ID")
        command.add_argument("--output", type=Path, default=os.environ.get("OUTPUT"))
        command.add_argument("--seed", type=int, default=os.environ.get("SEED"))
        command.add_argument("--dry-run", action="store_true", default=os.environ.get("DRY_RUN") == "1",
                             help="Print the plan without writing files or starting subprocesses")
        default_methods = {"mc": MC_METHODS, "open": OPEN_METHODS, "mc-epochs": EPOCH_METHODS}[name]
        command.add_argument("--methods", default=os.environ.get("METHODS", default_methods))
        if name != "open":
            command.add_argument("--data", type=Path, default=os.environ.get("DATA_FILE"))
        if name in {"mc", "open"}:
            command.add_argument("--updates", type=positive,
                                 default=os.environ.get("UPDATES", 500 if name == "mc" else 250))
        if name == "mc":
            command.add_argument("--batch-size", type=positive, default=os.environ.get("BATCH_SIZE"))
            command.add_argument("--eval-every", type=positive, default=os.environ.get("EVAL_EVERY_UPDATES", 50))
        elif name == "open":
            command.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"))
            command.add_argument("--curve", action="store_true", default=os.environ.get("CURVE") == "1")
        else:
            command.add_argument("--source-run", type=Path, default=os.environ.get("SOURCE_RUN"),
                                 help="Selection-curve artifacts, not an OPSD candidate directory")
            command.add_argument("--epochs", type=positive, default=os.environ.get("EPOCHS", 50))
            command.add_argument("--full-epoch-scale", type=positive, default=os.environ.get("FULL_EPOCH_SCALE", 5))
    return result


def build_plan(args):
    config_path = (args.config or ROOT / "configs" / CONFIGS[args.experiment]).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    model = args.model or config["model"]["name"]
    if Path(model).is_dir():
        model = str(Path(model).resolve())
    config["model"]["name"] = model
    if args.seed is not None:
        config["seed"] = args.seed
    seed = config["seed"]
    tag = Path(model.replace("\\", "/")).name
    default_output = ROOT / "artifacts" / f"{args.experiment}_{tag}_seed{seed}"
    output = (args.output or default_output).resolve()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    allowed = {"mc": MC_METHODS, "open": OPEN_METHODS, "mc-epochs": EPOCH_METHODS}[args.experiment].split(",")
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(allowed):
        raise ValueError(f"Choose distinct methods from: {', '.join(allowed)}")
    method_string = ",".join(methods)
    resolved = output / "resolved_config.json"
    env = {"ROOT": str(ROOT), "PYTHON": sys.executable, "MODEL": model,
           "OUTPUT": str(output), "SEED": str(seed), "METHODS": method_string,
           "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"}
    commands = []
    inputs = {"experiment": args.experiment, "config": config}
    if args.experiment == "mc":
        data = (args.data or ROOT / "artifacts/comprehensive_500_500.jsonl.gz").resolve()
        batch = args.batch_size or int(config["training"]["batch_size"])
        if args.updates % args.eval_every:
            raise ValueError("--updates must be divisible by --eval-every")
        config["training"]["batch_size"] = batch
        env.update(CONFIG=str(resolved), DATA_FILE=str(data), BATCH_SIZE=str(batch),
                   TRAJECTORIES=str(args.updates * batch), EPOCHS="0",
                   EVAL_EVERY_UPDATES=str(args.eval_every), PLOT_METHODS=method_string)
        inputs.update(data=str(data), updates=args.updates, batch_size=batch, eval_every=args.eval_every)
        if not data.is_file():
            commands.append({"command": ["bash", "scripts/download_and_prepare_comprehensive.sh"],
                             "env": {"COMPREHENSIVE_OUTPUT": str(data)}})
        commands.append({"command": ["bash", "scripts/run_neuron_opsd_comparison.sh"]})
    elif args.experiment == "open":
        if os.environ.get("WALL_BUDGET_SECONDS"):
            raise ValueError("This entry point runs fixed-update experiments; unset WALL_BUDGET_SECONDS")
        data = (args.data_dir or ROOT / "artifacts/data" / f"open_hotpot_coqa_squad_seed{seed}").resolve()
        config["training"]["updates"] = args.updates
        # The existing open-task script owns its resolved configuration and model profile.
        env.update(CONFIG=str(config_path), DATA_DIR=str(data), UPDATES=str(args.updates),
                   CURVE="1" if args.curve else "0", JS_TASK_PROMPT="1",
                   OPEN_MIX_RECIPE="hotpot_coqa_squad", SKIP_ORIGINAL="0")
        inputs.update(data_dir=str(data), updates=args.updates, curve=args.curve,
                      prompt="task_qa_v5", belief_protocol="joint_1_5_js_only_clustering_v8")
        commands.extend([{"command": ["bash", "scripts/run_open_tasks.sh", "mixed"]},
                         {"command": [sys.executable, "-m", "rig.open_tasks.costs", "--output", str(output)]}])
    else:
        data = (args.data or ROOT / "artifacts/comprehensive_500_500.jsonl.gz").resolve()
        source = (args.source_run or ROOT / "artifacts" / f"mc_selection_{tag}_seed{seed}").resolve()
        if source == output:
            raise ValueError("--source-run and --output must be different directories")
        inputs.update(data=str(data), source_run=str(source), epochs=args.epochs,
                      full_epoch_scale=args.full_epoch_scale)
        config_args = ["--config", str(resolved), "--model", model, "--data", str(data)]
        selection_manifest = source / "subsets/manifest.json"
        present = set()
        if selection_manifest.is_file():
            present = {row["method"] for row in json.loads(selection_manifest.read_text(encoding="utf-8"))["subsets"]}
        required = set(methods) - {"random"}
        if "random" in methods:
            required.add("full")
        if not required.issubset(present):
            selection_methods = [m for m in methods if m not in {"random", "full"}]
            commands.append({"command": [sys.executable, "-m", "rig.curve_experiment", *config_args,
                "--output-dir", str(source), "--methods", ",".join(selection_methods or ["uncertainty"]),
                "--fractions", "0.2", "--selection-only", "--generation-rollouts", "5",
                "--eval-rollouts", "3", "--max-new-tokens", "10"]})
        commands.append({"command": [sys.executable, "-m", "rig.epoch_curve_experiment", *config_args,
            "--source-run", str(source), "--output-dir", str(output), "--methods", method_string,
            "--epochs", str(args.epochs), "--full-epoch-scale", str(args.full_epoch_scale),
            "--eval-rollouts", "3", "--max-new-tokens", "10"]})
    return {"root": str(ROOT), "output": str(output), "config_path": str(config_path),
            "config": config, "identity": copy.deepcopy(inputs), "methods": methods,
            "environment": env, "steps": commands}


def record_run(plan):
    output = Path(plan["output"])
    manifest = output / "run_manifest.json"
    identity = plan["identity"]
    if manifest.is_file():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Run settings changed; choose a new --output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output already contains artifacts from another entry point; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"identity": identity, "methods": plan["methods"]}, indent=2) + "\n", encoding="utf-8")
    if identity["experiment"] != "open":
        (output / "resolved_config.json").write_text(json.dumps(plan["config"], indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return
        if args.experiment == "open" and not (Path(plan["environment"]["MODEL"]) / "config.json").is_file():
            raise ValueError("Open-task experiments require --model to point to a downloaded model directory")
        record_run(plan)
        environment = os.environ.copy()
        environment.update(plan["environment"])
        environment["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            environment.setdefault(key, "0")
        print(f"[Run] {args.experiment}; methods={','.join(plan['methods'])}", flush=True)
        print(f"[Output] {plan['output']}", flush=True)
        for step in plan["steps"]:
            step_env = {**environment, **step.get("env", {})}
            print("+ " + shlex.join(step["command"]), flush=True)
            subprocess.run(step["command"], cwd=ROOT, env=step_env, check=True)
    except (ValueError, OSError) as error:
        cli.error(str(error))


if __name__ == "__main__":
    main()
