"""MBPP child worker: Docker or the explicit AutoDL process backend."""
import contextlib
import json
import os
import resource
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def single(payload):
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    scope = {"__name__": "generated_module"}
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(payload["setup"], "<setup>", "exec"), scope)
            exec(compile(payload["solution"], "<solution>", "exec"), scope)
            for test in payload["tests"]:
                exec(compile(test, "<official-test>", "exec"), scope)
        return {"base_status": "pass"}
    except BaseException as error:
        return {"base_status": "fail", "error": type(error).__name__}


def main():
    if "--single" in sys.argv:
        print(json.dumps(single(json.load(sys.stdin))))
        return
    tests = {row["uid"]: row for row in map(json.loads, Path("/inputs/tests.jsonl").read_text().splitlines()) if row}
    predictions = [json.loads(line) for line in Path("/inputs/samples.jsonl").read_text().splitlines() if line.strip()]
    if {row["uid"] for row in predictions} != set(tests):
        raise ValueError("Prediction and test task sets differ")
    results = defaultdict(list)
    for row in predictions:
        payload = {**tests[row["uid"]], "solution": row["solution"]}
        try:
            child = subprocess.run([sys.executable, "-I", __file__, "--single"], input=json.dumps(payload), text=True, capture_output=True, timeout=12)
            result = json.loads(child.stdout) if child.returncode == 0 and len(child.stdout) < 1_000_000 else {"base_status": "fail", "error": "worker_failed"}
        except subprocess.TimeoutExpired:
            result = {"base_status": "fail", "error": "timeout"}
        except ValueError:
            result = {"base_status": "fail", "error": "invalid_worker_output"}
        results[row["uid"]].append({**result, "solution": row["solution"]})
        print(f"Completed {row['uid']} sample {row['sample_index']}", flush=True)
    Path("/results/eval_results.json").write_text(json.dumps({"benchmark": "MBPP official test_list", "eval": results}))


if __name__ == "__main__":
    main()
