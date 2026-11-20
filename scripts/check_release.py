#!/usr/bin/env python
"""Check the publishable source tree without reading datasets or model weights."""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("src", "scripts", "configs", "tests", "docs", ".github")
SKIP = {"__pycache__", ".pytest_cache", "build", "dist"}
TEXT = {".py", ".sh", ".json", ".toml", ".md", ".yml", ".yaml", ".txt"}
SECRETS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(r"(?:password|passwd|HF_TOKEN|GITHUB_TOKEN)\s*=\s*['\"][^'\"\s${}]{8,}['\"]", re.I),
)


def main():
    files = [path for path in ROOT.iterdir() if path.is_file()]
    for directory in SOURCE_DIRS:
        files.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and not SKIP.intersection(path.relative_to(ROOT).parts))
    failures = []
    counts = {"python": 0, "json": 0, "text": 0}
    for path in files:
        if path.suffix not in TEXT and path.name not in {"LICENSE", ".gitignore", ".gitattributes"}:
            continue
        relative = path.relative_to(ROOT)
        try:
            text = path.read_text(encoding="utf-8")
            if text.startswith("\ufeff") or "\ufffd" in text:
                raise ValueError("BOM or replacement character in UTF-8 text")
            counts["text"] += 1
            if path.suffix == ".py":
                ast.parse(text, filename=str(relative))
                counts["python"] += 1
            elif path.suffix == ".json":
                json.loads(text)
                counts["json"] += 1
            if any(pattern.search(text) for pattern in SECRETS):
                failures.append(f"{relative}: credential-like value; inspect locally")
            if re.search(r"[A-Za-z]:[/\\]Users[/\\][^/\\\s]+", text) or re.search(r"connect\.[\w.-]+\.seetacloud\.com", text):
                failures.append(f"{relative}: private workstation/server path")
            if path.suffix == ".md":
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    if re.match(r"(?:https?://|mailto:|#)", target):
                        continue
                    target = target.split("#")[0]
                    if target and not (path.parent / target).exists():
                        failures.append(f"{relative}: missing link target {target}")
        except (UnicodeError, ValueError, SyntaxError) as error:
            failures.append(f"{relative}: {error}")
    environment = os.environ.copy()
    for name in ("MODEL", "CONFIG", "OUTPUT", "SEED", "METHODS", "UPDATES", "EPOCHS",
                 "BATCH_SIZE", "EVAL_EVERY_UPDATES", "DATA_FILE", "DATA_DIR", "SOURCE_RUN",
                 "FULL_EPOCH_SCALE", "CURVE", "DRY_RUN", "WALL_BUDGET_SECONDS"):
        environment.pop(name, None)
    environment.update(PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8")
    for family in ("mc", "open", "mc-epochs"):
        result = subprocess.run([sys.executable, "-m", "rig.launch", family, "--dry-run"],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                                env=environment)
        if result.returncode:
            failures.append(f"{family} dry-run: {result.stderr.strip()}")
        else:
            json.loads(result.stdout)
    print(f"Checked {counts['python']} Python files, {counts['json']} JSON files, "
          f"{counts['text']} UTF-8 text files and 3 experiment plans.")
    if failures:
        print("\n".join(failures))
        return 1
    print("Release source checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
