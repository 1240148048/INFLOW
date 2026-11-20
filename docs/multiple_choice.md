# Multiple-choice main experiments

## Run the comparison

```bash
bash scripts/run_mc.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --data artifacts/comprehensive_500_500.jsonl.gz \
  --output artifacts/mc_qwen3 \
  --updates 500 --eval-every 50
```

The default batch size is four trajectories. A 500-update run therefore consumes 2,000 freshly sampled student trajectories; its update-250 checkpoint consumes 1,000. The scheduler spans the requested full training budget. To retain the 500-update schedule and report update 250, run the full curve and read `eval/<method>/update-250/metrics.json`. A separate `--updates 250` run uses a shorter schedule and is a different configuration.

Use `--methods full,ours` to restrict training/evaluation. The shared preprocessing pipeline currently prepares retrieval and teacher artifacts for all six strategies even when training a subset. Original is evaluated without training.

## Pipeline and method identifiers

| Stage | Module |
| --- | --- |
| Shared one-answer preparation, entropy and internal features | `rig.opsd_prepare` |
| Retrieval | `rig.opsd_select` |
| Five teacher samples per target | `rig.opsd_teacher_prepare` |
| Fixed target selection | `rig.opsd_finalize` |
| Fresh student rollout, token reverse KL, EMA teacher | `rig.opsd_train` |
| Three-rollout evaluation | `rig.opsd_evaluate` |

| Identifier | Retrieval | Target selection |
| --- | --- | --- |
| `full` | Final-layer cosine | All targets |
| `redundancy` | Same retrieval and teacher pool as Full | Greedy representation redundancy removal |
| `consistency` | No retrieved context | Five-answer agreement |
| `neuron` | Contributing-neuron Jaccard overlap | Mean activated-neuron count |
| `mutual` | Layerwise Gaussian-information score | Stored retrieval-information score |
| `ours` | Certainty-calibrated final-layer information score in this OPSD implementation | Before/after answer-distribution JS |

The current `ours_contexts` function in `opsd_select.py` uses the last layer; `mutual_contexts` uses flattened layerwise features. The trajectory-based source reconstruction and final-layer controls are implemented separately in `rig.rebuttal_ablation`. The release launcher preserves these existing computations.

OPSD training samples a fresh student response for every example at every update. Student prefixes are held fixed while differentiating token-mean reverse KL against the retrieved-context EMA teacher. Training never reads the gold answer as its target.

## Dataset

Missing processed data trigger `scripts/download_and_prepare_comprehensive.sh`, which downloads SciKnowEval and MMLU-Pro and constructs the 500/500 split. To prepare it independently:

```bash
COMPREHENSIVE_DATA_DIR="$PWD/artifacts/raw_mc" \
COMPREHENSIVE_OUTPUT="$PWD/artifacts/comprehensive_500_500.jsonl.gz" \
bash scripts/download_and_prepare_comprehensive.sh
```

If the data are already prepared, pass their existing path with `--data`. `HF_ENDPOINT` and Hugging Face authentication can be configured externally; no endpoint or access token is embedded in the code.

## Backbones

Standard Qwen3, Ministral, and Llama attention/MLP LoRA modules use `configs/neuron_opsd_single_gpu.json`. Qwen3.5 has additional DeltaNet projections:

```bash
bash scripts/run_mc.sh \
  --model models/Qwen3.5-9B \
  --config configs/qwen35_9b_single_gpu.json \
  --output artifacts/mc_qwen35
```

Model-specific loading and MLP access live in `modeling.py`. Ministral uses `mistral-common`; the checkpoints are loaded through their text-generation interfaces.

## Outputs and restart behavior

```text
run_manifest.json          Launcher configuration identity
resolved_config.json       Exact model/training settings
candidates/shared.jsonl.gz Shared initial answers and features
retrieval/                 Per-method retrieved source IDs
teacher/                   Per-method teacher samples and answer beliefs
subsets/                   Selected question IDs and ranking metadata
checkpoints/               Student adapters and update snapshots
eval/                      Original and checkpoint-level metrics
curve_results.csv          Accuracy/SD/majority accuracy by update
accuracy_update_curve.pdf  Update curve
```

Repeat the same command to reuse completed candidate, teacher, training, and evaluation artifacts. The existing MC trainer has no optimizer-state resume; an interrupted method is retrained from its initial state by the comparison script. Do not change model, seed, data, or training budget in the same run directory.

## Other MC experiments

The original entry points remain available:

- `run_qwen_module_ablation.sh`: similarity/certainty/selection controls.
- `run_opsd_fraction_ablation.sh`: selection ratio.
- `run_qwen_context_ablation.sh`: retrieval depth.
- `run_rebuttal_mc_ablation.sh` and `start_rebuttal_mc_curve.sh`: controlled sampling, hard-flip, and last-layer comparisons.
- `analyze_belief_shift_direction.py`: gold-label post-hoc analysis only.

Their earlier invocation notes are preserved in [archive](archive/README.md).
