# Multiple-choice epoch experiments

## Run

```bash
bash scripts/run_mc_epochs.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --data artifacts/comprehensive_500_500.jsonl.gz \
  --output artifacts/epochs_qwen3 \
  --epochs 50
```

The launcher first prepares candidate features and 20% selections if a compatible selection manifest does not exist. It then invokes `rig.epoch_curve_experiment`, which trains each requested variant and evaluates choice log loss after every epoch.

To reuse an earlier selection-curve run:

```bash
bash scripts/run_mc_epochs.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --source-run artifacts/my_selection_run \
  --output artifacts/epochs_qwen3
```

`--source-run` must contain `subsets/manifest.json` from `rig.curve_experiment` / `rig.curve_selection`, or their standard candidate files. OPSD `candidates/shared.jsonl.gz` and `opsd_finalize` subset metadata use a different format and are not interchangeable with this experiment.

## Objective and comparison

This is the existing cached-answer selection experiment. It trains LoRA with cross-entropy on the stored answer targets; it does not resample on-policy student trajectories or update an EMA teacher. The separate context-distillation option in `train_epoch_curve_single_gpu.py` is not activated by this entry point.

| Identifier | Selection |
| --- | --- |
| `random` | Resample 100 questions from the 500-question pool at each epoch |
| `difficulty` | Disagreement with the canonical answer among other samples |
| `uncertainty` | Predictive entropy |
| `redundancy` | Representation redundancy removal |
| `activation` | Activated-neuron statistic |
| `full` | All 500 training questions |
| `ours` | Historical QA-information ordering in `qa_information.py` |
| `ours-inverse` | Inverse QA-information ordering |

These historical `ours`/`ours-inverse` identifiers belong to the epoch experiment. They do not denote the before/after JS selector of the main OPSD pipeline. The release retains their original computation and output names.

## Exposure normalization

Selected methods use 100 questions per epoch. Full uses 500 questions, so `--full-epoch-scale 5` runs 10 actual Full epochs for a 50-subset-epoch comparison. A Full point at actual epoch 2 is plotted at normalized exposure 10. The total question exposure is equal at the endpoint; the number of optimizer updates is not forced equal.

The existing trainer defaults to full-batch optimization: one update per selected-data epoch, or one per actual Full epoch. Smaller minibatches are available through its direct `--batch-size` option and define a different run.

## Outputs and metrics

```text
run_manifest.json           Launcher configuration identity
resolved_config.json        Epoch model, prompt, and optimizer configuration
<variant>/epoch_metrics.csv Per-epoch log loss, accuracy, timing, and exposure
<variant>/final_adapter/    Final LoRA adapter
epoch_curve_results.csv     Combined records with normalized exposure
log_loss_epoch_curve.pdf    Choice log-loss curves
```

Choice log loss is evaluated on the held-out gold option probabilities. Shaded bands are standard errors across test questions, not SD across decoding rollouts. The original-model choice loss is shown as a reference line.

Completed variants are reused. An interrupted variant is trained again from its initial state; this older trainer does not save an optimizer resume state. Use a new output directory when changing the epoch budget, source selections, seed, or configuration.
