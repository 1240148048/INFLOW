from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from rig.modeling import (
    hidden_size,
    language_backbone,
    lm_head,
    mlp_activation_modules,
    mlp_down_projections,
    transformer_layers,
)


class Olmo3LikeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(8, 16, bias=False)
        self.up_proj = nn.Linear(8, 16, bias=False)
        self.down_proj = nn.Linear(16, 8, bias=False)
        self.act_fn = nn.SiLU()


class Olmo3LikeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = Olmo3LikeMlp()


class Olmo3LikeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Olmo3LikeLayer(), Olmo3LikeLayer()])


class Olmo3LikeCausalLm(nn.Module):
    """Minimal mirror of Olmo3ForCausalLM's public module layout."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="olmo3", hidden_size=8)
        self.model = Olmo3LikeBackbone()
        self.lm_head = nn.Linear(8, 32, bias=False)


def test_olmo3_public_layout_is_supported() -> None:
    model = Olmo3LikeCausalLm()

    assert hidden_size(model) == 8
    assert language_backbone(model) is model.model
    assert lm_head(model) is model.lm_head
    assert transformer_layers(model) == list(model.model.layers)
    assert mlp_down_projections(model) == [
        layer.mlp.down_proj for layer in model.model.layers
    ]
    assert mlp_activation_modules(model) == [
        layer.mlp.act_fn for layer in model.model.layers
    ]
