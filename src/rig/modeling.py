from __future__ import annotations

from typing import Iterable

import torch
import transformers
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.versions import require_version


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def require_model_runtime(name: str) -> None:
    """Fail early with an actionable dependency error for recent model families."""
    normalized = name.replace("\\", "/").lower()
    if "olmo-3" in normalized or "olmo3" in normalized:
        require_version(
            "transformers>=4.57.0",
            "OLMo 3 requires transformers>=4.57.0. Upgrade Transformers before "
            "running the OLMo experiment.",
        )
    if "qwen3.5" in normalized or "qwen3_5" in normalized:
        if getattr(transformers, "Qwen3_5ForCausalLM", None) is None:
            raise ImportError(
                "Qwen3.5 text-only experiments require a Transformers build "
                "that exposes Qwen3_5ForCausalLM. Upgrade Transformers before "
                "running the Qwen3.5 experiment."
            )


def _resolve_attr(root: object, candidates: Iterable[str]):
    for candidate in candidates:
        current = root
        try:
            for component in candidate.split("."):
                current = getattr(current, component)
            return current
        except AttributeError:
            continue
    raise AttributeError(f"Could not resolve any of {list(candidates)}")


def load_tokenizer(name: str, trust_remote_code: bool = False):
    require_model_runtime(name)
    config = AutoConfig.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    if getattr(config, "model_type", "") == "mistral3":
        backend = getattr(transformers, "MistralCommonBackend", None)
        if backend is None:
            raise ImportError(
                "Ministral-3 requires a recent Transformers build exposing "
                "MistralCommonBackend and mistral-common>=1.8.6."
            )
        return backend.from_pretrained(name, local_files_only=True)
    return AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )


def load_causal_model(
    name: str,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
):
    require_model_runtime(name)
    config = AutoConfig.from_pretrained(
        name,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )
    model_type = getattr(config, "model_type", "")
    if model_type == "mistral3":
        model_class = getattr(transformers, "Mistral3ForConditionalGeneration")
        fp8_config = getattr(transformers, "FineGrainedFP8Config")(
            dequantize=True
        )
    elif model_type in {"qwen3_5", "qwen3_5_text"}:
        # The published Qwen3.5 checkpoint is a multimodal composite checkpoint.
        # Loading its explicit causal-LM view avoids constructing the vision tower
        # and avoids dtype propagation bugs seen in some AutoModel releases.
        model_class = getattr(transformers, "Qwen3_5ForCausalLM", None)
        if model_class is None:
            raise ImportError(
                "This Transformers build does not expose Qwen3_5ForCausalLM. "
                "Install a recent Transformers release with Qwen3.5 support."
            )
        fp8_config = None
    else:
        model_class = AutoModelForCausalLM
        fp8_config = None
    major = int(transformers.__version__.split(".", 1)[0])
    dtype_key = "dtype" if major >= 5 else "torch_dtype"
    kwargs = {
        dtype_key: dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
        "local_files_only": True,
    }
    if fp8_config is not None:
        kwargs["quantization_config"] = fp8_config
    model = model_class.from_pretrained(
        name,
        **kwargs,
    )
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        actual_dtype = next(
            parameter.dtype
            for parameter in model.parameters()
            if parameter.is_floating_point()
        )
        if actual_dtype != dtype:
            raise RuntimeError(
                f"Qwen3.5 loaded in {actual_dtype}, but {dtype} was requested. "
                "Upgrade Transformers or load the explicit text-only checkpoint."
            )
    if fp8_config is not None and hasattr(model, "generation_config"):
        # Ministral ships max_length=262144; all RIG generation uses
        # max_new_tokens, so remove the redundant total-length limit.
        model.generation_config.max_length = None
    return model


def hidden_size(model: nn.Module) -> int:
    config = model.config
    for candidate in (config, getattr(config, "text_config", None)):
        if candidate is not None and hasattr(candidate, "hidden_size"):
            return int(candidate.hidden_size)
    raise AttributeError("The model config does not expose a text hidden size.")


def language_backbone(model: nn.Module) -> nn.Module:
    return _resolve_attr(
        model,
        (
            "language_model",
            "model.language_model",
            "language_model.model",
            "model.language_model.model",
            "model",
            "transformer",
        ),
    )


def lm_head(model: nn.Module) -> nn.Module:
    return _resolve_attr(
        model,
        ("lm_head", "language_model.lm_head", "model.language_model.lm_head"),
    )


def mlp_down_projections(model: nn.Module) -> list[nn.Module]:
    layers = transformer_layers(model)
    output = []
    for layer in layers:
        mlp = _resolve_attr(layer, ("mlp", "feed_forward"))
        projection = _resolve_attr(mlp, ("down_proj", "c_proj", "wo", "w2"))
        if not hasattr(projection, "weight") or not hasattr(projection, "in_features"):
            raise TypeError(f"Unsupported MLP down projection: {type(projection)!r}")
        output.append(projection)
    return output


def transformer_layers(model: nn.Module) -> list[nn.Module]:
    return list(
        _resolve_attr(
            model,
            (
                "language_model.layers",
                "model.language_model.layers",
                "language_model.model.layers",
                "model.language_model.model.layers",
                "model.layers",
                "model.model.layers",
                "transformer.h",
            ),
        )
    )


def mlp_activation_modules(model: nn.Module) -> list[nn.Module]:
    output = []
    for layer in transformer_layers(model):
        mlp = _resolve_attr(layer, ("mlp", "feed_forward"))
        output.append(_resolve_attr(mlp, ("act_fn", "activation_fn", "act")))
    return output


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
