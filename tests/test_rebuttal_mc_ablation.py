import json
import math

import pytest

from rig.io import Example
from rig.rebuttal_ablation import import_existing, probability, score_and_select


def row(uid, probabilities):
    samples = [{"text": str(i), "choice_probabilities": {"A": p, "B": 1-p}}
               for i, p in enumerate(probabilities)]
    return Example(uid=str(uid), dataset="mmlu_pro", domain="math", split="train", question="Q",
                   choices=["yes", "no"], choice_labels=["A", "B"], rollout="original teacher",
                   metadata={"opsd_teacher_generation": {"samples": samples}})


def test_flip_uses_argmax_of_mean_distribution_and_independent_ties():
    rows = [row(i, [.9, .1, .1, .1, .1]) for i in range(10)]
    initial = {r.uid: [{"choice_probabilities": {"A": .8, "B": .2}}] for r in rows}
    selected, beliefs = score_and_select(rows, initial, "sample_1_5_hard_flip", 42, .3)
    assert all(r.metadata["rebuttal_beliefs"]["hard_flip"] == 1 for r in selected)
    again, _ = score_and_select(rows, initial, "sample_1_5_hard_flip", 42, .3)
    other, _ = score_and_select(rows, initial, "sample_1_5_hard_flip", 43, .3)
    assert [r.uid for r in selected] == [r.uid for r in again]
    assert [r.uid for r in selected] != [r.uid for r in other]
    assert len(beliefs) == 10 and rows[0].rollout == "original teacher"


def test_one_one_uses_first_cached_teacher_and_balanced_initial_is_averaged():
    r = row(1, [.9, .1, .1, .1, .1])
    initial = {r.uid: [{"choice_probabilities": {"A": .8, "B": .2}}]}
    selected, _ = score_and_select([r], initial, "sample_1_1_js", 42, 1)
    assert selected[0].metadata["rebuttal_beliefs"]["contextual"] == pytest.approx([.9, .1])
    assert selected[0].rollout == "0"
    balanced = {r.uid: [{"choice_probabilities": {"A": p, "B": 1-p}} for p in [.1, .2, .3, .4, .5]]}
    selected, _ = score_and_select([r], balanced, "sample_5_5_js", 42, 1)
    assert selected[0].metadata["rebuttal_beliefs"]["initial"] == pytest.approx([.3, .7])


def test_import_reads_raw_values_and_never_fills_missing_pixels(tmp_path):
    source, ablation, out = tmp_path/"source", tmp_path/"ablation", tmp_path/"out"
    path = source/"eval/ours/update-250/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"overall": {"avg_at_3": .63133333333, "rollout_accuracy_std": .00529}}))
    results = import_existing(source, ablation, out, [250])
    ours = next(r for r in results if r["method"] == "ours")
    assert (ours["avg_percent"], ours["sd_percent"]) == ("63.13", "0.53")
    assert all(r["avg_percent"] == "" for r in results if r["method"] != "ours")
    assert ours["status"] == "reused_raw_metrics"


def test_invalid_probabilities_are_not_silently_normalized():
    with pytest.raises(ValueError):
        probability([{"choice_probabilities": {"A": math.nan, "B": .2}}], ["A", "B"])


def test_import_can_reuse_original_plot_csv_when_json_is_missing(tmp_path):
    source, ablation = tmp_path/"source", tmp_path/"ablation"
    ablation.mkdir()
    (ablation/"module_ablation_results.csv").write_text(
        "method,optimizer_update,accuracy,accuracy_std\nno_certainty,250,0.622,0.00471\n")
    rows = import_existing(source, ablation, tmp_path/"out", [250])
    value = next(row for row in rows if row["method"] == "no_certainty")
    assert (value["avg_percent"], value["sd_percent"], value["status"]) == ("62.20", "0.47", "reused_original_csv")


def test_digitized_csv_remains_marked_as_estimate_with_original_sd_empty(tmp_path, monkeypatch):
    path = tmp_path / "figure.csv"
    path.write_text("method,optimizer_update,avg_percent,sd_percent\nours,250,63.14,0.52\noriginal,0,59.94,\n")
    monkeypatch.setenv("DIGITIZED_CSV", str(path))
    rows = import_existing(tmp_path/"source", tmp_path/"ablation", tmp_path/"out", [250])
    ours = next(row for row in rows if row["method"] == "ours")
    original = next(row for row in rows if row["method"] == "original")
    assert ours["status"] == "digitized_estimate" and ours["avg_percent"] == "63.14"
    assert original["sd_percent"] == ""
