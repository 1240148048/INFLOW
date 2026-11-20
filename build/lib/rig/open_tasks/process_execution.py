"""Explicit Linux process backend for AutoDL containers without Docker.

Resource limits, scratch directories and process groups contain accidental runaway
benchmark code. This is not a security boundary equivalent to Docker.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from ..io import read_jsonl
from .data import dump


def run_worker(worker: Path, payload: dict, timeout: float):
    if os.name != "posix":
        raise RuntimeError("The process execution backend requires Linux/POSIX; use Docker elsewhere")
    environment = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "SYSTEMROOT") if key in os.environ}
    environment.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""})
    with tempfile.TemporaryDirectory(prefix="inflow-code-") as directory:
        with tempfile.TemporaryFile(mode="w+b") as captured:
            process = subprocess.Popen(
                [sys.executable, "-I", str(worker.resolve()), "--single"],
                stdin=subprocess.PIPE, stdout=captured, stderr=subprocess.DEVNULL,
                cwd=directory, env=environment, start_new_session=True,
            )
            try:
                process.communicate(json.dumps(payload).encode("utf-8"), timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
                return {"ok": False, "base_status": "fail", "error": "timeout"}
            finally:
                # Clean up descendants even if the entry function exited normally.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.returncode:
                return {"ok": False, "base_status": "fail", "error": "worker_failed"}
            captured.seek(0)
            raw = captured.read(2_000_001)
            try:
                if len(raw) > 2_000_000:
                    raise ValueError("worker output too large")
                return json.loads(raw)
            except (ValueError, UnicodeError):
                return {"ok": False, "base_status": "fail", "error": "invalid_worker_output"}


class ProcessProbeRunner:
    def __init__(self, timeout=5):
        if os.name != "posix":
            raise RuntimeError("AutoDL process execution requires Linux/POSIX")
        self.timeout = timeout

    def run(self, texts, contract):
        from .semantics import extract_code
        worker = Path(__file__).with_name("probe_worker.py")
        outputs = [run_worker(worker, {"code": extract_code(text), "entry_point": contract["entry_point"], "probes": contract["probes"]}, self.timeout) for text in texts]
        keys = ["behavior:" + json.dumps(row["behavior"], sort_keys=True, allow_nan=False) if row.get("ok") else None for row in outputs]
        return keys, outputs


def evaluate_mbpp(samples, tests, output, timeout=12):
    rows = list(read_jsonl(samples))
    test_rows = list(read_jsonl(tests))
    tasks = {row["uid"]: row for row in test_rows}
    if len(tasks) != len(test_rows) or not rows or {row["uid"] for row in rows} != set(tasks):
        raise ValueError("MBPP predictions/tests must contain exactly the same task IDs")
    seen = [(row["uid"], row["sample_index"]) for row in rows]
    if len(seen) != len(set(seen)):
        raise ValueError("Duplicate MBPP candidates")
    output = Path(output)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        from collections import Counter
        supplied = Counter((row["uid"], row["solution"]) for row in rows)
        cached = Counter((uid, row["solution"]) for uid, items in existing["eval"].items() for row in items)
        if supplied != cached:
            raise ValueError("Existing MBPP results are stale; use a new output directory")
        print(f"Reusing {output}")
        return
    worker = Path(__file__).with_name("mbpp_worker.py")
    results = defaultdict(list)
    for index, row in enumerate(rows):
        result = run_worker(worker, {**tasks[row["uid"]], "solution": row["solution"]}, timeout)
        results[row["uid"]].append({**result, "solution": row["solution"]})
        print(f"MBPP process evaluation: {index + 1}/{len(rows)}", flush=True)
    dump(output, {"benchmark": "MBPP official test_list", "backend": "process", "eval": results})
