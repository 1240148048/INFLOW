# Open-ended mixed-task experiments

## Run

```bash
bash scripts/run_open.sh \
  --model models/Qwen3-4B-Instruct-2507 \
  --output artifacts/open_qwen3 \
  --updates 250
```

The entry point trains one model per method on a mixed pool of 500 questions. It evaluates Original and all six trainable methods only at the final checkpoint. Use `--curve` for checkpoint evaluations every 50 updates. `--methods full,js` restricts preparation/training to those method families.

Backbones share the same program. Change `--model` to the local Qwen3.5, Ministral, or Llama checkpoint directory. `open_tasks/profiles.py` reads `config.json` and resolves LoRA module names; Qwen3.5 includes its DeltaNet projections.

## Data composition

| Task | Training questions | Held-out questions | Evaluation source |
| --- | ---: | ---: | --- |
| HotpotQA | 200 | 200 | Disjoint samples from distractor/train, medium difficulty |
| CoQA | 150 | 150 | Separate train/validation conversations, sampled chronological QA turns |
| SQuAD | 150 | 150 | Train/validation sources |
| Total | 500 | 500 | Per-task and question-weighted Overall results |

CoQA uses model-generated history, with previous answers propagated within each conversation. HotpotQA includes all supplied paragraphs, without supporting-fact filtering. The data manifest records source revisions, sampling rules, and overlap checks.

If Hugging Face access is unavailable for HotpotQA, supply an already downloaded official training JSON without editing code:

```bash
HOTPOT_TRAIN_JSON=/path/to/hotpot_train_v1.1.json \
bash scripts/run_open.sh --model models/Qwen3-4B-Instruct-2507
```

`--data-dir` controls the prepared mixed dataset. Its default is `artifacts/data/open_hotpot_coqa_squad_seed42`; per-task components are placed next to it and reused across backbones. There is no code execution or MBPP probe generation in this recipe.

## Beliefs, retrieval, and selection

1. Methods other than Consistency share one initial answer per training question. It supplies retrieval features and certainty `exp(-mean predictive entropy)`, with the entropy averaged over response prediction positions.
2. Each retrieval family samples five contextual answers. Consistency instead samples without retrieval. Full and Redundancy share their final-layer retrieval and teacher pool.
3. **Only `js` (INFLOW) calls the LLM semantic-equivalence judge.** The judge partitions the one initial and five contextual answers jointly. It judges equivalence in question/context, not correctness.
4. Each answer contributes `exp(-mean predictive entropy)` to its semantic class. Class masses are normalized separately before and after retrieval, and JS is computed on the shared class space. The initial empirical belief has one answer.
5. INFLOW selects the global top 20% across the mixed pool, without fixed task quotas. Full retains all targets. Other methods keep their own selection principles.

The shared answer-only prompt is `task_qa_v5`, and the belief protocol is `joint_1_5_js_only_clustering_v8`. The semantic partition parser supports six labels or a complete list of index groups and merges identical answer strings. The current prompts and constrained generation implementation live in `open_tasks/beliefs.py` and `runtime.py`.

The method identifier `js` means INFLOW in these scripts and output files. It is not an additional baseline.

## Training and evaluation

Training uses fresh student rollouts, fixed sampled prefixes, a retrieved-context EMA teacher, and token-mean reverse KL. The default is 250 updates, batch size four, LoRA rank 16, and three final evaluation rollouts. All methods share the answer-only prompt and optimization settings.

`accuracy` in `results.csv` is normalized answer exact match in percent. `rollout_accuracy_sd` is the population SD across the three rollout-index accuracies. Overall is weighted by the 200/150/150 question counts; its SD is computed from the three Overall rollout accuracies, not by averaging per-task SDs. CoQA here uses single-reference EM/F1, not the official multi-annotator leaderboard score. F1 is also exported but is a different metric from SD.

## Outputs

```text
run_manifest.json          Launcher configuration identity
resolved_config.json       Model profile and full protocol
initial.jsonl              Shared initial answers and predictive entropies
teacher.<family>.jsonl     Contextual samples (teacher.jsonl for INFLOW)
selection_manifest.json    Method selections and selection timing
subsets/                   Selected training rows
checkpoints/               Student/teacher adapters and optimizer resume state
predictions/               Three-rollout held-out predictions
metrics/                   Per-task and Overall scores
results.csv                Final method comparison
cost_breakdown.csv         Recorded stage timings, in seconds
```

Repeat the exact command to resume. Completed samples and predictions are reused; training restores the saved student, EMA teacher, optimizer, scheduler, data order, and RNG state from its last saved update. A changed configuration requires another output directory.

The launcher exports the stage costs automatically. To refresh them without loading a model:

```bash
python -m rig.open_tasks.costs --output artifacts/open_qwen3
```

The export charges shared preparation to each applicable method and separates evaluation. It reconstructs costs from recorded successful-stage timings; failed attempts and unrecorded orchestration overhead are not included. A cost table alone is not a fixed-wall-clock accuracy experiment.

The older TOPv2/MBPP recipes and their process/Docker execution backends remain available through `run_open_tasks.sh` and the [historical notes](archive/OPEN_TASKS.md).
