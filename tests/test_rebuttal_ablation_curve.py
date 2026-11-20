import argparse
import csv
import json
from pathlib import Path

import pytest

from rig.io import write_jsonl
from rig import rebuttal_ablation_curve as curve


def setup_run(tmp_path):
    previous, source, data = tmp_path / "old", tmp_path / "source", tmp_path / "data.jsonl.gz"
    config = {"training": {"batch_size": 4}, "evaluation": {"num_rollouts": 3}}
    curve.save_json(previous / "experiment.json", {
        "config": config, "model": "original-model", "data": str(data),
        "source": str(source), "training_updates": 500,
    })
    row = dict(uid="a", split="train", rollout="cached teacher", demo_ids=[])
    write_jsonl(data, [row])
    write_jsonl(source / "candidates/shared.jsonl.gz", [row])
    for variant in curve.VARIANTS:
        write_jsonl(previous / "subsets" / f"{variant}.jsonl.gz", [row])
    (previous / "all_results.csv").write_text(
        "method,optimizer_update,avg_percent,sd_percent\n"
        "ours,50,60.61,1.13\noriginal,0,59.94,\n"
        "sample_1_1_js,250,62.33,0.25\n", encoding="utf-8")
    return argparse.Namespace(previous_run=previous, output=None, variants=curve.VARIANTS,
                              config=None, model=None, data=None, source_run=None,
                              subsets_dir=None, existing_results=None, action="run")


def test_full_curve_reuses_subsets_and_evaluates_every_saved_adapter(tmp_path, monkeypatch):
    args = setup_run(tmp_path)
    calls = []
    original_subsets = {p.name: p.read_bytes() for p in (args.previous_run / "subsets").iterdir()}

    def run(*arguments):
        module, *rest = map(str, arguments)
        options = dict(zip(rest[::2], rest[1::2]))
        calls.append((module, options))
        if module == "rig.opsd_train":
            assert options["--trajectories"] == "2000"
            assert options["--checkpoint-every-trajectories"] == "200"
            assert "--final-checkpoint-only" not in rest
            target = Path(options["--output-dir"])
            curve.save_json(target / "training_metrics.json", {
                "optimizer_updates": 500, "scheduler_horizon_updates": 500,
            })
            for step in curve.STEPS:
                adapter = target / f"checkpoint-{4 * step}" / "student"
                curve.save_json(adapter / "adapter_config.json", {})
                (adapter / "adapter_model.safetensors").touch()
        elif module == "rig.opsd_evaluate":
            curve.save_json(Path(options["--output"]) / "metrics.json", {
                "overall": {"avg_at_3": .6, "rollout_accuracy_std": .01},
            })
        else:
            pytest.fail(f"Generation/selection unexpectedly invoked: {module}")

    monkeypatch.setattr(curve, "run_command", run)
    monkeypatch.setattr(curve, "plot_results", lambda *a: None)
    curve.execute(args)
    assert len([c for c in calls if c[0] == "rig.opsd_train"]) == 5
    assert len([c for c in calls if c[0] == "rig.opsd_evaluate"]) == 50
    output = args.previous_run / "curve_500"
    with (output / "all_results.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 52
    assert all(len([r for r in rows if r["method"] == variant]) == 10 for variant in curve.VARIANTS)
    assert original_subsets == {p.name: p.read_bytes() for p in (args.previous_run / "subsets").iterdir()}
    calls.clear()
    curve.execute(args)
    assert calls == []


def test_old_250_point_is_not_used_to_fill_new_curve(tmp_path):
    args = setup_run(tmp_path)
    rows = curve.collect_results(args.previous_run, args.previous_run / "curve_500")
    new = [r for r in rows if r["method"] == "sample_1_1_js"]
    assert len(new) == 10
    assert all(r["status"] == "pending" and r["avg_percent"] == "" for r in new)


def test_missing_subset_does_not_trigger_generation(tmp_path, monkeypatch):
    args = setup_run(tmp_path)
    (args.previous_run / "subsets" / f"{curve.VARIANTS[0]}.jsonl.gz").unlink()
    monkeypatch.setattr(curve, "run_command", lambda *a: pytest.fail("Must fail before training"))
    with pytest.raises(FileNotFoundError, match="nothing will be regenerated"):
        curve.execute(args)


def test_old_result_provenance_is_preserved(tmp_path):
    args = setup_run(tmp_path)
    curve.save_json(args.previous_run / "all_results_sources.json", [{
        "method": "ours", "optimizer_update": 50, "status": "digitized_estimate",
        "source": "original-figure.csv",
    }])
    rows = curve.collect_results(args.previous_run, args.previous_run / "curve_500")
    assert rows[0]["status"] == "digitized_estimate"
    assert rows[0]["source"] == "original-figure.csv"


def test_plan_reads_existing_files_without_writing_or_running(tmp_path, monkeypatch):
    args = setup_run(tmp_path)
    args.action = "plan"
    monkeypatch.setattr(curve, "run_command", lambda *a: pytest.fail("Plan must not train"))
    curve.execute(args)
    assert not (args.previous_run / "curve_500").exists()


def test_old_run_cannot_be_overwritten(tmp_path):
    args = setup_run(tmp_path)
    args.output = args.previous_run
    with pytest.raises(ValueError, match="separate output"):
        curve.execute(args)
