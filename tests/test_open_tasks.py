from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from rig.io import read_jsonl, write_jsonl
from rig.open_tasks.data import dump, stratified, validate_pool
from rig.open_tasks.evaluation import score, validate_predictions
from rig.open_tasks.pipeline import prepare, records, retrieve, select
from rig.open_tasks.report import paired_interval, report
from rig.open_tasks.semantics import certainty, js_divergence, parse_tree, selection_scores, tree_metrics, validate_contract
from rig.open_tasks.training import trim_tokens


def test_tree_equivalence_preserves_slot_values_and_nesting():
    a = "[IN:SET alarm [SL:TIME 7 am ] ]"
    assert parse_tree(a) == parse_tree("[ IN:SET  alarm [ SL:TIME 7 am ] ]")
    assert tree_metrics(a, a)["exact"] == 1
    split = tree_metrics("[IN:SET alarm [SL:TIME 7am]]", a)
    assert (split["exact"], split["strict_exact"], split["slot_tp"]) == (1, 0, 1)
    compact = tree_metrics("[IN:SET [SL:TIME 7am]]", a)
    assert (compact["exact"], compact["surface_exact"], compact["slot_tp"]) == (1, 0, 1)
    assert tree_metrics("[IN:SET alarm [SL:TIME 8 am ] ]", a)["exact"] == 0
    assert tree_metrics("[IN:SET alarm [SL:OTHER 7 am ] ]", a)["slot_tp"] == 0
    assert tree_metrics("[IN:GO [SL:PLACE [IN:LOC Rome]]]",
                        "[IN:GO [SL:PLACE [IN:LOC Paris]]]")["exact"] == 0
    assert tree_metrics("text " + a, a)["valid"] == 0
    with pytest.raises(ValueError):
        parse_tree("[IN:SET alarm")


def test_beliefs_do_not_turn_common_failures_into_high_certainty():
    assert certainty(["a"] * 5) == pytest.approx(1)
    assert certainty([None] * 5) == pytest.approx(0)
    assert certainty([str(i) for i in range(5)]) == pytest.approx(0)
    assert 0 < certainty(["a", "a", "a", None, None]) < 0.6
    assert js_divergence({"a": 1}, {"b": 1}) == pytest.approx(math.log(2))
    assert js_divergence({"a": 1}, {"a": 1}) == 0
    assert selection_scores(["a"] * 5, ["b"] * 5)["hard_flip"] == 1


def test_eos_is_not_discarded_when_it_is_also_padding():
    assert trim_tokens([4, 2, 2, 2], 2, 2) == [4, 2]
    assert trim_tokens([4, 2, 0], 2, 0) == [4, 2]


def test_training_pool_rejects_labels_and_test_rows():
    clean = {"uid": "a", "task": "code", "split": "train", "question": "Q", "domain": "code"}
    validate_pool([clean])
    with pytest.raises(ValueError):
        validate_pool([{**clean, "reference": "leaked answer"}])
    with pytest.raises(ValueError):
        validate_pool([{**clean, "split": "test"}])
    rows = [{"uid": str(i), "domain": "a" if i < 7 else "b"} for i in range(10)]
    assert len(stratified(rows, 5, 42)) == 5
    assert stratified(rows, 5, 42) == stratified(rows, 5, 42)


def test_contract_rejects_expected_outputs_and_mismatched_interface():
    value = {"entry_point": "f", "signature": "def f(x):", "probes": [{"args": [1], "kwargs": {}}]}
    assert validate_contract(json.dumps(value), 1) == value
    with pytest.raises(ValueError):
        validate_contract(json.dumps({**value, "entry_point": "g"}), 1)
    value["probes"][0]["expected"] = 2
    with pytest.raises(ValueError):
        validate_contract(json.dumps(value), 1)


def test_retrieval_uses_cross_layer_information_and_excludes_self():
    rows = [
        {"uid": "target", "trajectory": [[1, 0], [1, 0]], "certainty": 1},
        {"uid": "trajectory", "trajectory": [[1, 0], [0.8, 0.6]], "certainty": 1},
        {"uid": "last", "trajectory": [[-1, 0], [1, 0]], "certainty": 1},
    ]
    config = {"selection": {"context_count": 1}, "representation": {"mode": "trajectory"}}
    for row in rows:
        row["task"] = "topv2"
    rows.append({"uid": "foreign", "task": "code", "trajectory": [[1, 0], [1, 0]], "certainty": 1})
    rows.append({"uid": "foreign2", "task": "code", "trajectory": [[1, 0], [1, 0]], "certainty": 1})
    assert retrieve(rows, config)["target"] == ["trajectory"]
    config["representation"]["mode"] = "final_layer"
    assert retrieve(rows, config)["target"] == ["last"]


class FakeRuntime:
    calls = 0

    def __init__(self, config, **kwargs):
        pass

    def sample(self, chat, count, seed, settings):
        FakeRuntime.calls += 1
        assert all("reference" not in message for message in chat)
        return ["[IN:SET alarm [SL:TIME 7 am ] ]"] * count, [[1, 2, 3]] * count, [1, 2]

    def trajectory(self, prompt_ids, response):
        return [[1, 0], [0, 1]]

    def predictive_entropies(self, prompt_ids, responses):
        return [1.0] * len(responses)

    def partition_answers(self, chat, max_prompt_tokens):
        data = json.loads(chat[1]['content'])
        groups = {}
        for item in data['answers']:
            groups.setdefault(item['text'].strip().casefold(), []).append(item['index'])
        return json.dumps(list(groups.values()))

    def neuron_features(self, prompt_ids, response):
        return [1, 3, 4]


def make_fixture(tmp_path):
    data, output = tmp_path / "data", tmp_path / "run"
    rows = [{"uid": str(i), "task": "topv2", "split": "train", "question": f"set alarm {i}", "domain": "alarm"} for i in range(6)]
    write_jsonl(data / "train.jsonl", rows)
    dump(data / "schema.json", {"labels": ["IN:SET", "SL:TIME"]})
    config = {"seed": 42, "generation": {"num_samples": 1}, "teacher_generation": {"num_samples": 5}, "representation": {"projection_dim": 2, "mode": "trajectory"}, "selection": {"fraction": 0.5, "context_count": 1}, "evaluation": {"num_samples": 2}}
    return data, output, config


def test_preparation_resumes_and_selectors_share_teacher_cache(tmp_path):
    data, output, config = make_fixture(tmp_path)
    FakeRuntime.calls = 0
    prepare(config, data, output, runtime_factory=FakeRuntime)
    assert FakeRuntime.calls == 36
    prepare(config, data, output, runtime_factory=FakeRuntime)
    assert FakeRuntime.calls == 36
    select(config, output, ["full", "redundancy", "consistency", "neuron", "mutual", "random", "js", "hard_flip"])
    initial = records(output / "initial.jsonl")
    for method, count in [("full", 6), ("redundancy", 3), ("consistency", 3), ("neuron", 3), ("mutual", 3), ("random", 3), ("js", 3), ("hard_flip", 3)]:
        subset = list(read_jsonl(output / "subsets" / f"{method}.jsonl"))
        assert len(subset) == count
        assert all(row["uid"] not in row["demo_ids"] for row in subset)
        assert all(set(row["demo_ids"]) <= set(initial) for row in subset)
        if method == "consistency":
            assert all(not row["demo_ids"] and row["self_feedback"] is None for row in subset)
        else:
            assert all(len(row["demo_ids"]) == 1 for row in subset)
    changed = {**config, "seed": 9}
    with pytest.raises(ValueError, match="different data/config"):
        prepare(changed, data, output, runtime_factory=FakeRuntime)


def test_topv2_scoring_report_and_complete_prediction_check(tmp_path):
    data, output, config = make_fixture(tmp_path)
    reference = "[IN:SET alarm [SL:TIME 7 am ] ]"
    rows = [{"uid": "test", "task": "topv2", "split": "test", "reference": reference}]
    write_jsonl(data / "eval.jsonl", rows)
    predictions = [{"uid": "test", "sample_index": i, "text": reference if i == 0 else "invalid", "tokens": 3, "seconds": 1} for i in range(2)]
    write_jsonl(output / "predictions" / "original.jsonl", predictions)
    metrics = score(config, data, output, "original")
    assert metrics["exact_match"] == 50
    assert metrics["valid_parse_rate"] == 50
    report(output)
    assert (output / "results.tex").exists()
    with pytest.raises(ValueError):
        validate_predictions(rows, predictions[:1], 2)
    with pytest.raises(ValueError):
        validate_predictions(rows, [predictions[0], predictions[0]], 2)


def test_mbpp_scores_pass_at_one_not_any_success_at_five(tmp_path):
    data, output, config = make_fixture(tmp_path)
    config["evaluation"]["num_samples"] = 5
    rows = [{"uid": "mbpp:test:11", "task": "code", "split": "test"}]
    write_jsonl(data / "eval.jsonl", rows)
    predictions = [{"uid": "mbpp:test:11", "sample_index": i, "solution": f"def f(): return {i}\n", "tokens": 3, "seconds": 1} for i in range(5)]
    write_jsonl(output / "predictions" / "js.jsonl", predictions)
    items = [{"solution": p["solution"], "base_status": "pass" if i < 2 else "fail"} for i, p in enumerate(predictions)]
    evaluator = tmp_path / "evaluation.json"
    dump(evaluator, {"eval": {"mbpp:test:11": items}})
    metrics = score(config, data, output, "js", evaluator)
    assert metrics["pass_at_1"] == 40
    items[0]["solution"] = "changed"
    dump(evaluator, {"eval": {"mbpp:test:11": items}})
    with pytest.raises(ValueError, match="Stale"):
        score(config, data, output, "js", evaluator)


def test_question_paired_interval_keeps_candidate_means_grouped():
    result = paired_interval({"a": 0.5, "b": 1}, {"a": 0, "b": 0.5}, repetitions=100)
    assert result["difference_pp"] == 50
    assert result["ci95_low"] == 50


def test_code_probe_execution_uses_docker_and_cleans_up_on_timeout(monkeypatch):
    import subprocess
    import rig.open_tasks.sandbox as sandbox
    commands = []
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/docker")
    def fake_run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            assert command[command.index("--network") + 1] == "none"
            assert "--read-only" in command
            assert "--cap-drop" in command
            assert "expected" not in kwargs["input"]
            raise subprocess.TimeoutExpired(command, 1)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(sandbox.subprocess, "run", fake_run)
    runner = sandbox.DockerProbeRunner()
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(["def f(x): return x"], {"entry_point": "f", "probes": [{"args": [1], "kwargs": {}}]})
    assert commands[-1][:3] == ["docker", "rm", "-f"]
    assert all(command[0] == "docker" for command in commands)


@pytest.mark.parametrize("task,expected", [("topv2", (150, 150)), ("code", (150, 150)), ("coqa", (100, 100)), ("squad", (100, 100))])
def test_cli_defaults_keep_benchmark_size_small(monkeypatch, tmp_path, task, expected):
    import rig.open_tasks.data as data_module
    from rig.open_tasks.__main__ import main
    captured = []
    monkeypatch.setattr(data_module, "prepare_data", lambda args: captured.append((args.train_count, args.test_count)))
    main(["data", "--task", task, "--output", str(tmp_path)])
    assert captured == [expected]


def test_mbpp_interface_exposes_inputs_but_not_gold_outputs():
    from rig.open_tasks.data import public_interface
    entry, examples = public_interface(["assert sum(10, 15) == 6", "assert sum(4,6) == 3"])
    assert entry == "sum"
    assert examples == ["sum(10, 15)", "sum(4, 6)"]
    assert all("==" not in item for item in examples)


def test_coqa_keeps_turn_order_without_gold_history_and_predicts_own_history(tmp_path):
    from rig.open_tasks.qa import coqa_rows, qa_metrics
    from rig.open_tasks.evaluation import predict
    data, output, config = make_fixture(tmp_path)
    conversation = {"story": "A passage.", "source": "fiction", "questions": ["Who?", "What did they do?"], "answers": {"input_text": ["secret_gold_1", "secret_gold_2"]}}
    training = coqa_rows([conversation], "train", 2, 42)
    assert "secret_gold" not in json.dumps(training)
    assert training[1]["history_uids"] == [training[0]["uid"]]
    evaluation = coqa_rows([conversation], "test", 2, 43)
    write_jsonl(data / "eval.jsonl", evaluation)
    calls = []
    class DialogueRuntime(FakeRuntime):
        def sample(self, chat, count, seed, settings):
            calls.append(chat[-1]["content"])
            return [f"model_answer_{len(calls)}"], [[1]], [1]
    predict(config, data, output, "original", DialogueRuntime)
    assert "model_answer_1" in calls[2]
    assert "model_answer_2" in calls[3]
    assert all("secret_gold" not in prompt for prompt in calls)
    assert qa_metrics("The red car.", ["red car"])["exact"] == 1


def test_mixed_scoring_separates_tasks_and_weights_exact_accuracy(tmp_path):
    data, output, config = make_fixture(tmp_path)
    config["evaluation"]["num_samples"] = 1
    tree = "[IN:SET alarm ]"
    rows = [
        {"uid": "t", "task": "topv2", "reference": tree},
        {"uid": "c", "task": "code"},
        {"uid": "q", "task": "coqa", "references": ["red"]},
        {"uid": "s", "task": "squad", "references": ["blue"]},
    ]
    write_jsonl(data / "eval.jsonl", rows)
    predictions = [{"uid": row["uid"], "task": row["task"], "sample_index": 0, "text": text, "tokens": 1, "seconds": 1} for row, text in zip(rows, [tree, "code", "red", "blue"])]
    predictions[1]["solution"] = "def f(): return 0\n"
    write_jsonl(output / "predictions/js.jsonl", predictions)
    evaluator = tmp_path / "exec.json"
    dump(evaluator, {"eval": {"c": [{"solution": predictions[1]["solution"], "base_status": "fail"}]}})
    result = score(config, data, output, "js", evaluator)
    assert result["task"] == "mixed"
    assert set(result["by_task"]) == {"topv2", "code", "coqa", "squad"}
    assert result["overall_accuracy"] == 75
    report(output)
    assert "Overall" in (output / "results.md").read_text()
