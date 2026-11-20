"""Probe child worker, used by Docker or the explicit AutoDL process backend."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
import os
import random
import subprocess
import sys


def canonical(value):
    if value is None or isinstance(value, (bool, str, int)):
        return [type(value).__name__, value]
    if isinstance(value, float):
        return ["float", repr(value) if not math.isfinite(value) else value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical(x) for x in value]]
    if isinstance(value, (set, frozenset)):
        return ["set", sorted([canonical(x) for x in value], key=lambda x: json.dumps(x, sort_keys=True))]
    if isinstance(value, dict):
        pairs = [[canonical(k), canonical(v)] for k, v in value.items()]
        return ["dict", sorted(pairs, key=lambda x: json.dumps(x[0], sort_keys=True))]
    raise TypeError("Unsupported return type")


def single(payload):
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    scope = {"__name__": "generated_module"}
    try:
        random.seed(0)
        # Prevent an accidentally verbose program from filling the parent's pipe.
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(compile(payload["code"], "<generated>", "exec"), scope)
            function = scope[payload["entry_point"]]
            outputs = [canonical(function(*copy.deepcopy(p["args"]), **copy.deepcopy(p["kwargs"]))) for p in payload["probes"]]
        return {"ok": True, "behavior": outputs}
    except BaseException as error:
        return {"ok": False, "error": type(error).__name__}


def main():
    payload = json.load(sys.stdin)
    if "--single" in sys.argv:
        print(json.dumps(single(payload), allow_nan=False))
        return
    results = []
    for code in payload["codes"]:
        child = {"code": code, "entry_point": payload["entry_point"], "probes": payload["probes"]}
        try:
            process = subprocess.run([sys.executable, "-I", __file__, "--single"], input=json.dumps(child), text=True, capture_output=True, timeout=payload["timeout"])
            if process.returncode or len(process.stdout) > 2_000_000:
                result = {"ok": False, "error": "worker_failed"}
            else:
                result = json.loads(process.stdout)
        except subprocess.TimeoutExpired:
            result = {"ok": False, "error": "timeout"}
        except (ValueError, KeyError):
            result = {"ok": False, "error": "invalid_worker_output"}
        results.append(result)
    print(json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
