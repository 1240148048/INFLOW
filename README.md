# INFLOW

Code for retrieval-guided data selection in annotation-free language-model self-distillation.

The repository contains three experiment families:

| Experiment | Entry point | Guide |
| --- | --- | --- |
| Multiple-choice on-policy distillation and update curves | `scripts/run_mc.sh` | [Multiple-choice experiments](docs/multiple_choice.md) |
| Open-ended HotpotQA / CoQA / SQuAD experiments | `scripts/run_open.sh` | [Open-ended experiments](docs/open_ended.md) |
| Multiple-choice selection and epoch log-loss curves | `scripts/run_mc_epochs.sh` | [Epoch experiments](docs/mc_epochs.md) |

The public entry points invoke the existing experiment implementations. The Python import name remains `rig` for compatibility with earlier runs. Method identifiers and the distinct training objectives of the three families are described in the guides.

## Install

Use Python 3.10+ on Linux with a CUDA GPU for training. Install a PyTorch build matched to your CUDA environment first, then install the project from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the CUDA-matched PyTorch build before the next command.
python -m pip install -e ".[test]"
```

For Ministral, install `python -m pip install -e ".[test,ministral]"` instead. The Transformers and PEFT versions are shared across all three experiment families in [pyproject.toml](pyproject.toml). For the PyTorch 2.5.1 AutoDL environment, [requirements-autodl.txt](requirements-autodl.txt) also pins the compatible NumPy/PyArrow data stack without requesting a different PyTorch build.

Download your model before training; the loaders use local files or a fully populated Hugging Face cache. Open-ended experiments require a local model directory:

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
```

## Run

Run these commands from the repository root. Each run prints its stages and training progress immediately. `--dry-run` shows the configuration, paths, and subprocess commands without downloading, training, or writing files.

```bash
# 1. Multiple-choice main comparison: 500 updates, evaluate every 50.
bash scripts/run_mc.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --output artifacts/mc_qwen3

# 2. One model trained on HotpotQA 200 + CoQA 150 + SQuAD 150.
#    The held-out pool has the same composition. Evaluate at update 250.
bash scripts/run_open.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --output artifacts/open_qwen3

# 3. Multiple-choice epoch curves: prepare 20% selections if needed,
#    then train and evaluate across 50 subset epochs.
bash scripts/run_mc_epochs.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --output artifacts/epochs_qwen3
```

Equivalent Python entry points are available with `python -m rig.launch mc`, `open`, or `mc-epochs`. Use `--help` for options. `MODEL`, `CONFIG`, `OUTPUT`, `SEED`, and `METHODS` remain supported as environment variables; command-line options take precedence.

Run directories contain a `run_manifest.json` and the resolved configuration. Reusing an output directory with different training settings raises an error. Resume a matching run with the same command; preparation and evaluation reuse their completed artifacts. The MC OPSD and epoch trainers restart an interrupted training stage, while the open-ended trainer restores its saved optimizer and adapter state.

## Data and evaluation

- **Multiple choice:** a 500-question training pool and a disjoint 500-question test pool from SciKnowEval and MMLU-Pro. The preparation script builds them when missing.
- **Open ended:** HotpotQA medium 200, CoQA 150, and SQuAD 150 per pool. HotpotQA uses a custom disjoint holdout from train-medium; CoQA and SQuAD use their validation sources for evaluation.
- Gold labels are used for evaluation and post-hoc analysis, not retrieval, target ranking, or distillation targets.
- Reported rollout SD is computed across three decoding rollouts of one trained model, not independent training seeds. Epoch log-loss bands instead use standard errors across test questions.

Dataset/model access and redistribution follow their respective licenses. Downloads, weights, checkpoints, predictions, and local experiment results are excluded from Git.

## Code map

```text
configs/                   Model and experiment settings
scripts/run_mc.sh          Multiple-choice OPSD entry point
scripts/run_open.sh        Open-ended mixed-task entry point
scripts/run_mc_epochs.sh   Multiple-choice epoch entry point
src/rig/launch.py          Shared paths, configuration, and experiment launch plans
src/rig/opsd_*.py          Multiple-choice preparation, selection, training, evaluation
src/rig/open_tasks/        Open-ended data, semantic beliefs, baselines, training, metrics
src/rig/*epoch*.py         Epoch experiments and cached-answer training
tests/                    CPU tests for model interfaces, selection, beliefs, and metrics
docs/                     Usage and implementation notes
docs/archive/             Historical experiment notes
```

Earlier scripts are retained for compatibility, including selection-ratio/context/component ablations and the older TOPv2/MBPP recipes. The three guides above describe the main entry points; historical notes in `docs/archive/` retain their original run-specific paths and instructions.

## Checks

```bash
python -m pytest
python -m rig.launch mc --dry-run
python -m rig.launch open --dry-run
python -m rig.launch mc-epochs --dry-run
python scripts/check_release.py
```

The CPU suite checks implementation behavior. Full model experiments require CUDA and downloaded model/data files.

## License

[MIT](LICENSE). Third-party models and datasets retain their own license terms.
