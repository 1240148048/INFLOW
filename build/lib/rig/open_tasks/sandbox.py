from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path


class DockerProbeRunner:
    def __init__(self, image="python:3.11-slim", timeout=5):
        self.image, self.timeout = image, timeout
        if not shutil.which("docker"):
            raise RuntimeError("Code belief grouping requires Docker; no host-execution fallback is provided")
        subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True)

    def run(self, texts, contract):
        from .semantics import extract_code
        worker = Path(__file__).with_name("probe_worker.py").resolve()
        container = "inflow-probes-" + uuid.uuid4().hex[:12]
        command = [
            "docker", "run", "--rm", "--name", container, "--network", "none",
            "--cpus", "1", "--memory", "512m", "--pids-limit", "64",
            "--read-only", "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--user", "65534:65534", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,source={worker},target=/worker.py,readonly",
            "-i", self.image, "python", "-I", "/worker.py",
        ]
        payload = {"codes": [extract_code(x) for x in texts], "entry_point": contract["entry_point"], "probes": contract["probes"], "timeout": self.timeout}
        try:
            result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True, timeout=30 + (self.timeout + 2) * len(texts))
            if result.returncode:
                raise RuntimeError("Docker probe batch failed: " + result.stderr[-2000:])
            outputs = json.loads(result.stdout)
            if len(outputs) != len(texts):
                raise RuntimeError("Incomplete Docker probe batch")
            return ["behavior:" + json.dumps(x["behavior"], sort_keys=True, allow_nan=False) if x.get("ok") else None for x in outputs], outputs
        finally:
            # A timed-out Docker client may leave its container alive.
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=20)
