import os
import torch
import logging
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, VocabParallelEmbedding)

logger = logging.getLogger(__name__)

@register_quantization_config("bf16_last_layer_fp32")
class BF16WithLastLayerFP32(QuantizationConfig):
    """
    A custom mixed-precision configuration for vLLM.

    This configuration keeps the last layer in float32 for maximum precision
    while running all other layers using bfloat16 for improved performance
    and reduced memory usage.
    """

    def __init__(self, config: object | None = None):
        super().__init__()
        self.default_dtype = self._resolve_default_dtype(config)
        logger.info(
            "Initialized quantization config '%s' with default_dtype=%s; last layer forced to %s",
            self.get_name(),
            self.default_dtype,
            torch.float32,
        )

    @staticmethod
    def _resolve_default_dtype(config: object | None) -> torch.dtype:
        """Best-effort extraction of the default dtype for non-final layers."""

        if config is None:
            return torch.bfloat16

        # Passed a torch dtype directly.
        if isinstance(config, torch.dtype):
            return config

        # Allow string representations.
        if isinstance(config, str):
            return _string_to_dtype(config)

        # HuggingFace model config or similar objects typically expose `dtype`.
        dtype = getattr(config, "dtype", None)
        if dtype is not None:
            if isinstance(dtype, torch.dtype):
                return dtype
            return _string_to_dtype(dtype)

        # Dictionary based quantization configs can also specify the dtype.
        if isinstance(config, dict):
            for key in ("default_dtype", "activation_dtype", "dtype"):
                value = config.get(key)
                if value is None:
                    continue
                if isinstance(value, torch.dtype):
                    return value
                return _string_to_dtype(value)

        return torch.bfloat16

    def get_supported_activations(self) -> list[str]:
        # vLLM does not currently query this helper, but keep it for completeness.
        return ["relu", "gelu", "silu"]

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> LinearMethodBase | None:
        """Return quantization method for the given layer."""

        is_last_layer = prefix.endswith("lm_head") or prefix.endswith(
            ".lm_head") or (".lm_head." in prefix) or ("lm_head." in prefix)

        # Heuristic: many models (e.g., LLaMA/Qwen) tie the output projection
        # (unembedding) to the input token embedding. When weight tying is used,
        # there is no distinct Linear lm_head module. In that case, force the
        # embedding weights to float32 so the final logits use FP32 params.
        tied_unembed = False
        if isinstance(layer, VocabParallelEmbedding):
            name = prefix.lower()
            if any(s in name for s in ["embed_tokens", "tok_embeddings", "word_embeddings", "wte"]):
                tied_unembed = True

        if isinstance(layer, VocabParallelEmbedding):
            if is_last_layer:
                logger.info(
                    "Quant config forcing FP32 embedding for %s (%s)",
                    prefix,
                    layer.__class__.__name__,
                )
                return _ForcedDTypeEmbeddingMethod(torch.float32)
            if tied_unembed:
                # Keep BF16 parameters for input embedding, but force logits
                # matmul to FP32 via a custom apply() used by LogitsProcessor.
                logger.info(
                    "Detected tied embedding %s (%s); using FP32 unembedding matmul while keeping BF16 params.",
                    prefix,
                    layer.__class__.__name__,
                )
                return _FP32UnembedEmbeddingMethod()
            logger.info("Quant config leaving embedding %s (%s) unmodified", prefix, layer.__class__.__name__)
            return None

        if isinstance(layer, LinearBase):
            if is_last_layer:
                logger.info("Quant config forcing FP32 linear layer for %s (%s)", prefix, layer.__class__.__name__)
                # Also pin activations to FP32 via a pre-hook for clarity.
                try:
                    def _cast_input_to_fp32(_mod, _inp):
                        if not _inp:
                            return _inp
                        args = list(_inp)
                        if isinstance(args[0], torch.Tensor) and args[0].dtype != torch.float32:
                            logger.info("Casting input activation to FP32 for %s (%s)", prefix, layer.__class__.__name__)
                            args[0] = args[0].to(torch.float32)
                        return tuple(args)

                    # Avoid duplicate hooks on reload
                    if not hasattr(layer, "_pipelinerl_fp32_hook_installed"):
                        layer.register_forward_pre_hook(_cast_input_to_fp32)
                        layer._pipelinerl_fp32_hook_installed = True
                except Exception as _hook_err:  # pragma: no cover - defensive logging
                    logger.warning("Failed to install FP32 input hook for %s: %s", prefix, _hook_err)
                return _ForcedDTypeLinearMethod(torch.float32)
            logger.info("Quant config setting dtype %s for %s (%s)", self.default_dtype, prefix, layer.__class__.__name__)
            return _ForcedDTypeLinearMethod(self.default_dtype)

        logger.info("Quant config leaving module %s (%s) unmodified", prefix, layer.__class__.__name__)
        return None

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        dtypes = [self.default_dtype]
        if torch.float32 not in dtypes:
            dtypes.append(torch.float32)
        return dtypes

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_name(cls) -> str:
        return "bf16_last_layer_fp32"

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict | None = None):
        return cls(config)

    def get_tp_size(self) -> int:
        return 1
    
    def get_tp_group(self, tp_size: int):
        return None

    def get_supported_dtypes(self) -> list[torch.dtype]:
        dtypes = [self.default_dtype]
        if torch.float32 not in dtypes:
            dtypes.append(torch.float32)
        return dtypes



class _ForcedDTypeLinearMethod(UnquantizedLinearMethod):
    """Linear method that enforces a specific parameter dtype."""

    def __init__(self, target_dtype: torch.dtype):
        super().__init__()
        self._target_dtype = target_dtype

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        logger.info("Creating linear weights for %s with dtype %s", layer.__class__.__name__, self._target_dtype)
        return super().create_weights(layer,
                                      input_size_per_partition,
                                      output_partition_sizes,
                                      input_size,
                                      output_size,
                                      params_dtype=self._target_dtype,
                                      **extra_weight_attrs)


class _ForcedDTypeEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Embedding method that enforces a specific parameter dtype."""

    def __init__(self, target_dtype: torch.dtype):
        super().__init__()
        self._target_dtype = target_dtype

    def create_weights(self, layer: torch.nn.Module,
                       input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        logger.info("Creating embedding weights for %s with dtype %s", layer.__class__.__name__, self._target_dtype)
        return super().create_weights(layer,
                                      input_size_per_partition,
                                      output_partition_sizes,
                                      input_size,
                                      output_size,
                                      params_dtype=self._target_dtype,
                                      **extra_weight_attrs)


class _FP32UnembedEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Use BF16 weights in module state, but compute logits matmul in FP32.

    This method does NOT change the dtype of the embedding weights stored in
    the module (so input token embeddings remain in BF16). It only casts the
    hidden states and the weight to FP32 inside apply(), which is used by the
    LogitsProcessor to compute final logits via F.linear.
    """

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        # Log once to avoid spamming per token.
        if not hasattr(layer, "_pipelinerl_fp32_unembed_logged"):
            logger.info("Computing unembedding (logits) in FP32 for %s", layer.__class__.__name__)
            layer._pipelinerl_fp32_unembed_logged = True

        # Upcast activations to FP32 for the final matmul.
        x32 = x if x.dtype == torch.float32 else x.to(torch.float32)
        target_device = x32.device

        # Use the existing FP32 parameter directly if already suitable to
        # avoid extra copies. Otherwise, keep a cached FP32 copy per device.
        w_param = layer.weight
        if w_param.dtype == torch.float32 and w_param.device == target_device:
            w32 = w_param
        else:
            w_attr = "_pipelinerl_fp32_unembed_weight"
            w32 = getattr(layer, w_attr, None)
            needs_new_copy = (
                w32 is None or
                w32.dtype != torch.float32 or
                w32.device != target_device or
                w32.shape != w_param.shape
            )
            if needs_new_copy:
                # Only copy when necessary.
                w32 = w_param.to(device=target_device, dtype=torch.float32)
                setattr(layer, w_attr, w32)

        b32 = None
        if bias is not None:
            b32 = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
            if b32.device != target_device:
                b32 = b32.to(target_device)

        return torch.nn.functional.linear(x32, w32, b32)


def _string_to_dtype(value: str) -> torch.dtype:
    normalized = value.lower().replace("torch.", "")
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported dtype string: {value}") from exc


if __name__ == "__main__":
    get_quantization_config("bf16_last_layer_fp32")
