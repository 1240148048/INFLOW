# Supplementary MC ablation curves

- Artifact: three-panel scientific line plot, PNG/PDF/editable SVG.
- Source: existing comparison CSV plus new per-checkpoint `eval/<variant>/update-*/metrics.json`.
- Question: how component, retrieval-layer, and belief/selection ablations evolve over optimizer updates.
- Panels: (a) existing component comparisons; (b) last-layer variants vs Ours; (c) sampling and hard-flip variants vs Ours.
- X: optimizer updates 50, 100, ..., 500, not epochs or test sample counts.
- Y: original Avg@3 accuracy in percent; shading: original rollout-index population SD, not across-seed uncertainty.
- Palette: categorical Okabe-Ito, distinct markers/line styles, stable orange-red Ours.
- Traceability: all_results.csv and all_results_sources.json; retain digitization provenance in old comparison records.
- Missing values: gaps, never interpolation; pending new points are never filled with old endpoint scores.
- Status: execution/plotting implementation; no new GPU experiment results have been produced locally.
