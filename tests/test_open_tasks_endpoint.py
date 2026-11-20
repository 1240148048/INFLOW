from rig.io import write_jsonl
from rig.open_tasks.data import dump
from rig.open_tasks.endpoint import METHODS, budget_plan, write_tables
from rig.open_tasks.paper_baselines import PAPER_METHODS, family
from rig.open_tasks.pipeline import teacher_path
from rig.open_tasks.training import update_checkpoint


def test_budget_counts_preparation_and_reports_only_saved_eligible_checkpoints(tmp_path):
    output = tmp_path
    timing = {"generation_seconds": 10, "belief_seconds": 5, "feature_seconds": 3}
    records = [{"uid": "q", "timing": timing}]
    dump(output / "wall_clock_policy.json", {"budget_seconds": 100, "update_cap": 250})
    dump(output / "preparation_loading.json", {"seconds": 4})
    dump(output / "selection_manifest.json", {m: {"seconds": 2} for m in PAPER_METHODS})
    write_jsonl(output / "initial.jsonl", records)
    for name in {family(m) for m in PAPER_METHODS}:
        write_jsonl(teacher_path(output, name), records)
        dump(output / f"retrieval.{name}.timing.json", {"seconds": 8})
    # Preparation plus startup costs 52 seconds, leaving 48 seconds to train.
    for method in PAPER_METHODS:
        for update, elapsed in [(50, 30 if method != "neuron" else 49), (250, 40 if method == "full" else 80)]:
            directory = update_checkpoint(output, method, update)
            dump(directory / "checkpoint.json", {"method": method, "optimizer_update": update})
            dump(directory / "ready_time.json", {"distillation_seconds": elapsed, "startup_seconds": 2})
    plan = budget_plan(output)
    assert plan["js"]["update"] == 50
    assert plan["js"]["adaptation_seconds"] == 82
    assert plan["full"]["update"] == 250
    assert plan["neuron"]["update"] == 0 and plan["neuron"]["adaptation_seconds"] is None

    metrics = {"model": "/models/Qwen3-4B-Instruct-2507", "evaluation_id": "same-test-set",
               "by_task": {"code": {"pass_at_1": 60., "rollout_accuracy_sd": 2.},
                           "coqa": {"exact_match": 50., "rollout_accuracy_sd": 1.},
                           "squad": {"exact_match": 70., "rollout_accuracy_sd": 1.}},
               "overall_accuracy": 60., "rollout_accuracy_sd": 1., "generation_seconds": 99.}
    for method in METHODS:
        dump(output / "metrics" / f"{method}.json", metrics)
    for method in PAPER_METHODS:
        if plan[method]["update"] == 50:
            dump(output / "curve/step-000050/metrics" / f"{method}.json", {**metrics, "optimizer_update": 50})
    write_tables(output)
    assert "MBPP,60.0 +/- 2.00" in (output / "endpoint_table.csv").read_text()
    assert "Updates,0,250,50,50,--,50,50" in (output / "wall_clock_table.csv").read_text()
    assert "Distillation and checkpointing" in (output / "cost_table.tex").read_text()
