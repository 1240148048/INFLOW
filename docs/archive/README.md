> Historical README snapshot. For current entry points, see the [repository README](../../README.md).

# INFLOW: Internal Information Flow for Self-Distillation

The separate open-output benchmark mixes TOPv2 150 + MBPP 150 + CoQA 100 + SQuAD 100 questions per train/test split. See [OPEN_TASKS.md](OPEN_TASKS.md), then run `bash scripts/run_open_mixed_500.sh` on a CUDA machine.

INFLOW is a retrieval-guided framework for annotation-free, on-policy language-model self-distillation. It treats retrieval and training-data selection as two stages of the same information-transfer process:

1. **Potential information flow.** For each target question, INFLOW compares the layerwise hidden-state trajectories of final decision tokens. Trajectory similarity estimates whether a source follows a compatible internal computation, while source certainty discounts unreliable evidence.
2. **Realized information flow.** Retrieved examples condition an EMA teacher. INFLOW compares the teacher's initial semantic answer belief with its retrieval-conditioned belief and measures the induced change using Jensen--Shannon divergence.
3. **On-policy transfer.** Targets with the largest realized belief shifts are retained. A LoRA student is then trained on its own sampled prefixes against the retrieval-conditioned EMA teacher.

The central question is not only whether a retrieved example resembles the target, but whether it introduces information that is expressed in the teacher distribution and can subsequently be transferred to the student.

## Repository contents

```text
configs/        Model-specific generation and optimization configurations
scripts/        End-to-end experiments, ablations, analysis, and plotting
src/rig/        Candidate preparation, retrieval, selection, training, evaluation
tests/          Unit tests for data preparation, selection, training, and metrics
```

The Python package retains the internal module name `rig`; the method implemented by the main OPSD pipeline is INFLOW.

## Experimental protocol

The main benchmark contains 500 unlabeled training questions and 500 non-overlapping test questions drawn from SciKnowEval and MMLU-Pro. It covers mathematics, natural science, and humanities and social science. Gold labels are used only for evaluation and post-hoc analyses, never for retrieval, target selection, or distillation.

Unless overridden by a model-specific configuration, the main protocol uses three retrieved demonstrations, five retrieval-conditioned teacher generations, a 20% selection ratio, rank-16 LoRA, minibatches of four trajectories, and 2,000 on-policy trajectories. The latter corresponds to 500 optimizer updates; the paper's main table reports the checkpoint at 250 updates. Evaluation uses three sampled responses per test question and reports average rollout accuracy, its standard deviation across rollout indices, and majority-vote accuracy.

The controlled comparison includes:

- `full`: distillation on the complete generated pool;
- `redundancy`: quality-diversity selection using teacher representations;
- `consistency`: selection by agreement among teacher generations;
- `neuron`: selection and retrieval based on internal activation overlap;
- `mutual`: representation-level mutual-information selection;
- `ours`: INFLOW retrieval followed by semantic belief-shift selection.

All trainable methods use the same question pool, generation budget, LoRA optimization, EMA update, and evaluation protocol. Method-specific artifacts are stored separately so completed stages can be resumed.

## Installation

Python 3.10 or later and a CUDA-capable environment are required for the main experiments.

```bash
cd INFLOW
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a PyTorch build compatible with the local CUDA runtime before running the experiments. Ministral checkpoints may additionally require `mistral-common`; consult the checkpoint documentation for the compatible version.

Run the unit tests with:

```bash
python -m pip install -e ".[test]"
pytest
```

## Data preparation

The preparation script downloads the public SciKnowEval and MMLU-Pro sources and constructs the fixed 500/500 split:

```bash
cd INFLOW
bash scripts/download_and_prepare_comprehensive.sh
```

By default, processed data are written below `artifacts/`. Custom locations can be supplied without changing the code:

```bash
COMPREHENSIVE_DATA_DIR=/path/to/data-cache \
COMPREHENSIVE_OUTPUT=/path/to/benchmark.jsonl.gz \
bash scripts/download_and_prepare_comprehensive.sh
```

Users are responsible for complying with the licenses and access conditions of the datasets and model checkpoints they use.

## Main INFLOW experiment

Set `ROOT` to the repository root, `MODEL` to a Hugging Face identifier or local checkpoint, and `OUTPUT` to a writable run directory:

```bash
cd INFLOW

ROOT="$PWD" \
MODEL="Qwen/Qwen3-4B-Instruct-2507" \
OUTPUT="$PWD/artifacts/qwen3_main" \
bash scripts/run_neuron_opsd_comparison.sh
```

The entry point performs the complete experiment:

```text
initial generations and internal-state extraction
    -> method-specific retrieval
    -> five-sample contextual teacher generation
    -> target selection
    -> on-policy EMA self-distillation
    -> checkpoint evaluation and update curves
```

Important overrides include:

```bash
ROOT="$PWD" \
CONFIG="$PWD/configs/neuron_opsd_single_gpu.json" \
DATA_FILE="$PWD/artifacts/comprehensive_500_500.jsonl.gz" \
MODEL=/path/to/model \
OUTPUT="$PWD/artifacts/model_run" \
TRAJECTORIES=2000 \
BATCH_SIZE=4 \
EVAL_EVERY_UPDATES=50 \
METHODS=full,redundancy,consistency,neuron,ours,mutual \
bash scripts/run_neuron_opsd_comparison.sh
```

The default run directory contains:

```text
candidates/     Shared initial responses and internal signals
retrieval/      Retrieved contexts for each method
teacher/        Retrieval-conditioned teacher generations and beliefs
subsets/        Selected training targets and selection metadata
checkpoints/    LoRA checkpoints and training metrics
eval/           Per-checkpoint evaluation results
case_study/     Recomputable INFLOW examples
```

It also produces `curve_results.csv`, `curve_results.json`, and accuracy-update plots when the requested evaluations finish.

## Model configurations

The default configuration targets Qwen3-4B-Instruct. The same pipeline accepts other causal instruction models when their architecture is supported by the installed Transformers version. A dedicated configuration is included for Qwen3.5:

```bash
ROOT="$PWD" \
MODEL=/path/to/qwen3.5-checkpoint \
CONFIG="$PWD/configs/qwen35_9b_single_gpu.json" \
OUTPUT="$PWD/artifacts/qwen35_main" \
bash scripts/run_neuron_opsd_comparison.sh
```

Qwen3.5 uses its text-only causal-LM interface, adapts both attention and Gated DeltaNet projections, and chunks the full-vocabulary KL computation to reduce peak memory. For other model families, copy the closest configuration and update only architecture-dependent fields such as the model identifier and LoRA target modules.

## Ablations

The component ablation removes certainty calibration, removes trajectory similarity, or replaces belief-shift selection with a seeded random 20% subset. It reuses the candidate pool and original evaluation from a completed main run:

```bash
ROOT="$PWD" \
MODEL=/path/to/model \
SOURCE_RUN="$PWD/artifacts/qwen3_main" \
OUTPUT="$PWD/artifacts/component_ablation" \
bash scripts/run_qwen_module_ablation.sh
```

The selection-ratio experiment varies the retained fraction while fixing the optimization budget:

```bash
ROOT="$PWD" \
MODEL=/path/to/model \
SOURCE_RUN="$PWD/artifacts/qwen3_main" \
OUTPUT="$PWD/artifacts/selection_ratio" \
FRACTIONS=10,20,30,40,50,60,70,80,90 \
UPDATES=250 \
bash scripts/run_opsd_fraction_ablation.sh
```

The retrieval-depth experiment fixes the selected targets and varies the number of retrieved demonstrations:

```bash
ROOT="$PWD" \
MODEL=/path/to/model \
SOURCE_RUN="$PWD/artifacts/qwen3_main" \
OUTPUT="$PWD/artifacts/retrieval_depth" \
CONTEXT_COUNTS=1,2,3,4,5 \
UPDATES=250 \
bash scripts/run_qwen_context_ablation.sh
```

## Belief-shift direction analysis

The post-hoc analysis partitions examples into disjoint 10% intervals after sorting their JS belief shifts from largest to smallest. It reports the rates of incorrect-to-correct (`Fix`), correct-to-incorrect (`Break`), wrong-to-wrong, and right-to-right transitions, together with initial accuracy, retrieval-conditioned teacher accuracy, and mean JS. Gold answers are used only for this diagnostic.

```bash
python scripts/analyze_belief_shift_direction.py \
  --candidates-dir "$PWD/artifacts/qwen3_main/candidates"
```

Results are written to `belief_shift_direction/summary.csv` below the run directory. The initial decision is the argmax of the stored no-retrieval semantic belief, and the teacher decision is the argmax of the mean retrieval-conditioned belief, matching the distributions used to compute INFLOW's JS score.

## Retrieval visualization

The retrieval visualization projects candidate representations into two PCA dimensions and draws the source-target edges produced by INFLOW, Mutual, NEURON, and cosine retrieval for the same target set:

```bash
ROOT="$PWD" \
SOURCE_RUN="$PWD/artifacts/qwen3_main" \
OUTPUT="$PWD/artifacts/retrieval_visualization" \
CONTEXT_COUNT=3 \
MAX_TARGETS=60 \
bash scripts/plot_qwen_retrieval_graph.sh
```

The command also exports the projected coordinates, retrieval edges, same-domain rates, and original-space similarity statistics.

## Reproducibility notes

The default random seed is 42. Candidate preparation, retrieval, subset construction, optimization, and evaluation preserve deterministic ordering where possible, while method-specific offsets prevent different teacher-generation methods from sharing identical sampling streams. The scripts validate cached artifact metadata before reuse and resume completed candidate and teacher-generation stages.

Evaluation standard deviation is computed across the three rollout-index accuracies of a fixed trained checkpoint. It measures decoding variation and should not be interpreted as variance across independent training runs.
