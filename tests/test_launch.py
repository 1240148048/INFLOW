"""Public entry points must preserve each experiment's protocol and artifacts."""
import json
from pathlib import Path

import pytest

from rig.launch import build_plan, main, parser, record_run


@pytest.fixture(autouse=True)
def clean_launch_environment(monkeypatch):
    for name in ("MODEL", "CONFIG", "OUTPUT", "SEED", "METHODS", "UPDATES", "EPOCHS",
                 "BATCH_SIZE", "EVAL_EVERY_UPDATES", "DATA_FILE", "DATA_DIR", "SOURCE_RUN",
                 "FULL_EPOCH_SCALE", "CURVE", "DRY_RUN", "WALL_BUDGET_SECONDS"):
        monkeypatch.delenv(name, raising=False)


def plan(*arguments):
    return build_plan(parser().parse_args(arguments))


def test_mc_maps_updates_to_trajectories_without_changing_training_objective(tmp_path):
    data = tmp_path / "mc.jsonl.gz"
    data.touch()
    result = plan("mc", "--data", str(data), "--output", str(tmp_path / "run"),
                  "--updates", "500", "--eval-every", "50", "--methods", "full,ours")
    assert result["environment"]["TRAJECTORIES"] == "2000"
    assert result["environment"]["EPOCHS"] == "0"
    assert result["environment"]["METHODS"] == "full,ours"
    assert result["config"]["teacher_generation"]["num_samples"] == 5
    assert result["steps"] == [{"command": ["bash", "scripts/run_neuron_opsd_comparison.sh"]}]


def test_open_preserves_joint_one_five_protocol_and_exports_costs(tmp_path):
    result = plan("open", "--model", str(tmp_path / "model"), "--output", str(tmp_path / "run"))
    env = result["environment"]
    assert env["OPEN_MIX_RECIPE"] == "hotpot_coqa_squad"
    assert env["UPDATES"] == "250" and env["CURVE"] == "0"
    assert env["JS_TASK_PROMPT"] == "1" and env["SKIP_ORIGINAL"] == "0"
    assert env["METHODS"] == "full,redundancy,consistency,neuron,mutual,js"
    assert result["identity"]["belief_protocol"] == "joint_1_5_js_only_clustering_v8"
    assert result["steps"][-1]["command"][2] == "rig.open_tasks.costs"


def test_epochs_prepare_selection_pool_then_use_existing_epoch_trainer(tmp_path):
    result = plan("mc-epochs", "--source-run", str(tmp_path / "source"),
                  "--output", str(tmp_path / "run"), "--methods", "full,ours", "--epochs", "50")
    selection, training = [row["command"] for row in result["steps"]]
    assert selection[2] == "rig.curve_experiment" and "--selection-only" in selection
    assert selection[selection.index("--methods") + 1] == "ours"
    assert training[2] == "rig.epoch_curve_experiment"
    assert training[training.index("--full-epoch-scale") + 1] == "5"
    source_manifest = tmp_path / "source/subsets/manifest.json"
    source_manifest.parent.mkdir(parents=True)
    source_manifest.write_text(json.dumps({"subsets": [{"method": method} for method in
        ("random", "difficulty", "uncertainty", "redundancy", "activation", "full", "ours-inverse", "ours")]}), encoding="utf-8")
    assert len(plan("mc-epochs", "--source-run", str(tmp_path / "source"))["steps"]) == 1


def test_run_identity_allows_more_methods_but_rejects_changed_budget(tmp_path):
    output = tmp_path / "run"
    record_run(plan("mc", "--output", str(output), "--methods", "ours"))
    record_run(plan("mc", "--output", str(output), "--methods", "full,ours"))
    with pytest.raises(ValueError, match="Run settings changed"):
        record_run(plan("mc", "--output", str(output), "--updates", "250"))


def test_dry_run_has_no_side_effects_or_model_load(tmp_path, monkeypatch, capsys):
    output = tmp_path / "run"
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("dry-run started a process"))
    main(["open", "--model", "not-downloaded", "--output", str(output), "--dry-run"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"]["experiment"] == "open"
    assert not output.exists()


def test_existing_artifacts_are_preserved(tmp_path):
    output = tmp_path / "old_run"
    output.mkdir()
    artifact = output / "predictions.json"
    artifact.write_text('{"keep":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="already contains artifacts"):
        record_run(plan("mc", "--output", str(output)))
    assert artifact.read_text(encoding="utf-8") == '{"keep":true}'


def test_invalid_checkpoint_interval_and_cross_family_methods_are_rejected():
    with pytest.raises(ValueError, match="divisible"):
        plan("mc", "--updates", "251")
    with pytest.raises(ValueError, match="distinct methods"):
        plan("open", "--methods", "ours")


def test_cli_overrides_environment_without_overwriting_source_config(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL", "environment-model")
    monkeypatch.setenv("UPDATES", "100")
    result = plan("mc", "--model", "explicit-model", "--updates", "250", "--seed", "7")
    assert result["environment"]["MODEL"] == "explicit-model"
    assert result["environment"]["TRAJECTORIES"] == "1000"
    assert result["config"]["seed"] == 7
    original = json.loads(Path(result["config_path"]).read_text(encoding="utf-8"))
    assert original["seed"] == 42
