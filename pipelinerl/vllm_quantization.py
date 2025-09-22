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
            ".lm_head")

        if isinstance(layer, VocabParallelEmbedding):
            if is_last_layer:
                logger.info("Quant config forcing FP32 embedding for %s (%s)", prefix, layer.__class__.__name__)
                return _ForcedDTypeEmbeddingMethod(torch.float32)
            logger.info("Quant config leaving embedding %s (%s) unmodified", prefix, layer.__class__.__name__)
            return None

        if isinstance(layer, LinearBase):
            if is_last_layer:
                logger.info("Quant config forcing FP32 linear layer for %s (%s)", prefix, layer.__class__.__name__)
                return _ForcedDTypeLinearMethod(torch.float32)
            logger.info("Quant config setting dtype %s for %s (%s)", self.default_dtype, prefix, layer.__class__.__name__)
            return _ForcedDTypeLinearMethod(self.default_dtype)

        logger.debug("Quant config has no override for %s (%s)", prefix, layer.__class__.__name__)
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
        logger.debug("Creating linear weights for %s with dtype %s", layer.__class__.__name__, self._target_dtype)
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
        logger.debug("Creating embedding weights for %s with dtype %s", layer.__class__.__name__, self._target_dtype)
        return super().create_weights(layer,
                                      input_size_per_partition,
                                      output_partition_sizes,
                                      input_size,
                                      output_size,
                                      params_dtype=self._target_dtype,
                                      **extra_weight_attrs)


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
