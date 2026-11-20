"""Resolve backbone-specific module names without separate experiment programs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

STANDARD_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DELTA_TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
SUPPORTED_TYPES = {"qwen3", "qwen3_5", "qwen3_5_text", "mistral3", "llama"}


def resolve_profile(config, model):
    result = copy.deepcopy(config)
    path = Path(model) / "config.json"
    if not path.is_file():
        raise ValueError(f"MODEL must point to a downloaded local model directory: {model}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    model_type = metadata.get("model_type")
    if model_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported backbone type {model_type!r}; supported types: {sorted(SUPPORTED_TYPES)}")
    result["model"]["name"] = str(model)
    result["model"]["resolved_type"] = model_type
    result["training"]["lora_targets"] = STANDARD_TARGETS + (DELTA_TARGETS if model_type in {"qwen3_5", "qwen3_5_text"} else [])
    return result
