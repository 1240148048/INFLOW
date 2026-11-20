from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForConditionalGeneration

from rig.modeling import load_causal_model
from rig.open_tasks.profiles import resolve_profile


def test_qwen35_composite_checkpoint_loads_text_weights_and_lora_targets(tmp_path):
    text = Qwen3_5TextConfig(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=16, layer_types=["linear_attention", "full_attention"], linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8)
    config = Qwen3_5Config(text_config=text.to_dict(), vision_config={"depth": 1, "hidden_size": 16, "num_heads": 2, "intermediate_size": 32, "out_hidden_size": 32, "num_position_embeddings": 16})
    composite = Qwen3_5ForConditionalGeneration(config)
    expected = composite.get_input_embeddings().weight.detach().clone()
    folder = tmp_path / "qwen3.5-tiny"
    composite.save_pretrained(folder)
    loaded = load_causal_model(str(folder), torch.float32)
    torch.testing.assert_close(loaded.get_input_embeddings().weight, expected)
    base = json.loads((Path(__file__).parents[1] / "configs/open_mixed_autodl.json").read_text())
    profile = resolve_profile(base, folder)
    suffixes = {name.rsplit(".", 1)[-1] for name, _ in loaded.named_modules()}
    assert set(profile["training"]["lora_targets"]) <= suffixes
