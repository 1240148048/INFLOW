from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rig.open_tasks import process_execution as execution
from rig.open_tasks.data import dump
from rig.io import write_jsonl


@pytest.mark.parametrize("timeout", [False, True])
def test_process_worker_has_scratch_environment_limits_and_group_cleanup(monkeypatch, tmp_path, timeout):
    killed, observed = [], {}
    fake_os = SimpleNamespace(name="posix", environ={"PATH": "test-path", "API_SECRET": "must-not-propagate", "HF_TOKEN": "must-not-propagate"}, killpg=lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(execution, "os", fake_os)
    monkeypatch.setattr(execution, "signal", SimpleNamespace(SIGKILL=9))
    class Child:
        pid, returncode = 1234, 0
        def __init__(self, command, **kwargs):
            observed.update(kwargs)
            assert command[1] == "-I"
            assert kwargs["cwd"] != str(tmp_path)
            self.output = kwargs["stdout"]
            self.calls = 0
        def communicate(self, data=None, **kwargs):
            self.calls += 1
            if timeout and self.calls == 1:
                raise subprocess.TimeoutExpired("worker", 1)
            if data:
                self.output.write(json.dumps({"ok": True, "behavior": [1]}).encode())
            return None, None
    monkeypatch.setattr(execution, "subprocess", SimpleNamespace(Popen=Child, PIPE=-1, DEVNULL=-3, TimeoutExpired=subprocess.TimeoutExpired))
    result = execution.run_worker(tmp_path / "worker.py", {"code": "fixture"}, 1)
    assert observed["start_new_session"] is True
    assert "HF_TOKEN" not in observed["env"] and "API_SECRET" not in observed["env"]
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert killed
    assert result.get("error") == "timeout" if timeout else result["ok"]


def test_process_evaluation_uses_test_bundle_and_preserves_failures(monkeypatch, tmp_path):
    samples, tests, output = tmp_path / "samples.jsonl", tmp_path / "tests.jsonl", tmp_path / "result.json"
    write_jsonl(samples, [{"uid": "mbpp:test:1", "sample_index": i, "solution": f"program{i}"} for i in range(2)])
    write_jsonl(tests, [{"uid": "mbpp:test:1", "tests": ["assert f(1) == 2"], "setup": ""}])
    payloads = []
    def run(worker, payload, timeout):
        payloads.append(payload)
        return {"base_status": "pass" if len(payloads) == 1 else "fail"}
    monkeypatch.setattr(execution, "run_worker", run)
    execution.evaluate_mbpp(samples, tests, output)
    result = json.loads(output.read_text())
    assert [row["base_status"] for row in result["eval"]["mbpp:test:1"]] == ["pass", "fail"]
    assert payloads[0]["tests"] == ["assert f(1) == 2"]
    execution.evaluate_mbpp(samples, tests, output)
    assert len(payloads) == 2


@pytest.mark.skipif(__import__("os").name != "posix", reason="Real resource-limited child execution needs Linux")
def test_actual_linux_worker_and_timeout():
    worker = Path(execution.__file__).with_name("probe_worker.py")
    good = execution.run_worker(worker, {"code": "def f(x): return x + 1", "entry_point": "f", "probes": [{"args": [2], "kwargs": {}}]}, 5)
    assert good["ok"] and good["behavior"] == [["int", 3]]
    bad = execution.run_worker(worker, {"code": "while True: pass", "entry_point": "f", "probes": []}, 0.2)
    assert bad["error"] == "timeout"
