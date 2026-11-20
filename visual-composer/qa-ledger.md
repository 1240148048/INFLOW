# Supplementary ablation curve QA

- Six local orchestration tests passed: five training commands, fifty checkpoint evaluation commands, completed-work reuse, immutable existing subsets, missing-input behavior, old-result provenance, and non-mutating planning.
- Plot renderer inspected against the supplied CSV: all old points retained, new 250-update-only series shown as isolated markers, missing checkpoints labeled and never interpolated.
- Axis is optimizer update, not epoch; uncertainty remains rollout-index SD.
- Full server experiments have not been executed locally. The existing subsets, base weights, and runtime data are on the user's server, not this machine.
- The QA render is not the completed supplementary result figure and is excluded from the deployment patch.
