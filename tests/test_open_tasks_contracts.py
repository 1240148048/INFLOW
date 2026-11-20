import json

import pytest

from rig.open_tasks.contracts import generate_contract


def response(inputs, signature="def f(x):"):
    return json.dumps({"entry_point": "f", "signature": signature, "probes": [{"args": [x], "kwargs": {}} for x in inputs]})


class Runtime:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts = []

    def sample(self, prompt, count, seed, settings):
        self.prompts.append(prompt[0]["content"])
        assert settings["max_new_tokens"] >= 4096
        return [next(self.replies)], [], []


def config():
    return {"seed": 42, "code": {"probe_count": 16}, "contract_generation": {"max_new_tokens": 2048}}


def test_completes_partial_inputs_without_duplicates_or_interface_changes(tmp_path):
    runtime = Runtime([response(range(10)), response(range(10, 16), "def f(x,y):"), response(range(8, 16))])
    result = generate_contract(runtime, {"uid": "task", "question": "Return x."}, config(), 0, tmp_path)
    assert [p["args"][0] for p in result["probes"]] == list(range(16))
    assert "Still need 6" in runtime.prompts[1]
    assert "previously specified" in runtime.prompts[2]


def test_recovers_truncated_json_and_saves_raw_attempts(tmp_path):
    runtime = Runtime(['{"probes": [', response(range(16))])
    assert len(generate_contract(runtime, {"uid": "task", "question": "Q"}, config(), 0, tmp_path)["probes"]) == 16
    assert "JSONDecodeError" in runtime.prompts[1]
    assert len((tmp_path / "contract_attempts.jsonl").read_text().splitlines()) == 2


def test_repeated_invalid_responses_fail_without_fabricating_inputs(tmp_path):
    runtime = Runtime(["bad"] * 6)
    with pytest.raises(ValueError, match="Existing preparation caches are preserved"):
        generate_contract(runtime, {"uid": "task", "question": "Q"}, config(), 0, tmp_path)
