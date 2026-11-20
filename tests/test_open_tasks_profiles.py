import copy
import json

import pytest

from rig.open_tasks.profiles import DELTA_TARGETS, STANDARD_TARGETS, resolve_profile


@pytest.mark.parametrize("model_type", ["qwen3", "qwen3_5", "qwen3_5_text", "mistral3", "llama"])
def test_shared_protocol_is_preserved(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    base = {"model": {"name": "placeholder"}, "training": {"updates": 250, "lora_targets": []}, "seed": 42}
    before = copy.deepcopy(base)
    resolved = resolve_profile(base, tmp_path)
    assert base == before
    assert resolved["model"]["name"] == str(tmp_path)
    assert resolved["training"]["updates"] == 250
    assert resolved["seed"] == 42
    expected = STANDARD_TARGETS + (DELTA_TARGETS if model_type in {"qwen3_5", "qwen3_5_text"} else [])
    assert resolved["training"]["lora_targets"] == expected


def test_missing_or_unsupported_model_fails_early(tmp_path):
    with pytest.raises(ValueError, match="local model directory"):
        resolve_profile({}, tmp_path)
    (tmp_path / "config.json").write_text('{"model_type":"unknown"}')
    with pytest.raises(ValueError, match="Unsupported backbone"):
        resolve_profile({}, tmp_path)
