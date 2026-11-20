from __future__ import annotations

import torch

from .modeling import lm_head, mlp_down_projections


class NeuronActivationCapture:
    """Early-unembedding activation sets for short answer-only traces."""

    def __init__(
        self,
        model,
        prediction_positions: list[int],
        per_layer_candidates: int = 256,
        per_token_global_topk: int = 5000,
    ):
        self.projections = mlp_down_projections(model)
        self.head = lm_head(model)
        self.positions = prediction_positions
        self.per_layer_candidates = per_layer_candidates
        self.per_token_global_topk = per_token_global_topk
        self.values: dict[int, torch.Tensor] = {}
        self.hooks = []

    def __enter__(self):
        position_index = torch.tensor(self.positions, dtype=torch.long)
        for layer_index, projection in enumerate(self.projections):
            def hook(_module, inputs, index=layer_index):
                values = inputs[0]
                positions = position_index.to(values.device)
                self.values[index] = values.index_select(1, positions).detach()

            self.hooks.append(projection.register_forward_pre_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for hook in self.hooks:
            hook.remove()

    @torch.inference_mode()
    def activation_set(self, target_tokens: list[int]) -> list[int]:
        if not target_tokens:
            return []
        selected: set[int] = set()
        intermediate = self.projections[0].in_features
        unembedding = self.head.weight[
            torch.tensor(target_tokens, device=self.head.weight.device)
        ].float()
        for token_position in range(len(target_tokens)):
            scored: list[tuple[float, int]] = []
            for layer_index, projection in enumerate(self.projections):
                activation = self.values[layer_index][0, token_position].float()
                candidate_count = min(
                    self.per_layer_candidates, activation.numel()
                )
                candidates = torch.topk(
                    activation, k=candidate_count, sorted=False
                ).indices
                down = projection.weight[:, candidates].float()
                token_direction = unembedding[token_position].to(down.device) @ down
                contribution = (activation[candidates] * token_direction).clamp_min(0)
                scored.extend(
                    (
                        float(score),
                        layer_index * intermediate + int(neuron_index),
                    )
                    for score, neuron_index in zip(
                        contribution.tolist(), candidates.tolist()
                    )
                )
            scored.sort(key=lambda pair: pair[0], reverse=True)
            positive = [pair for pair in scored if pair[0] > 0.0]
            selected.update(
                identifier
                for _score, identifier in positive[: self.per_token_global_topk]
            )
        return sorted(selected)
