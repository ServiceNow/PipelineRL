import logging
import os
import threading

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

logger = logging.getLogger(__name__)

_FP32_LAYER_PREFIX_ENV = "PIPELINERL_FP32_LAYER_PREFIX"
_DEFAULT_FP32_LAYER_PREFIX = "lm_head"

_weight_version: int = 0
_weight_version_lock = threading.Lock()

_FP32_CACHE_ACTIVE = False


def activate_fp32_cache() -> None:
    """Enable FP32 cache mode. Called when bf16_last_layer_fp32 quant config is instantiated."""
    global _FP32_CACHE_ACTIVE
    _FP32_CACHE_ACTIVE = True


def increment_weight_version() -> int:
    """Increment weight version counter to invalidate cached FP32 copies. Returns new version."""
    global _weight_version
    with _weight_version_lock:
        _weight_version += 1
        logger.debug(f"Weight version incremented to {_weight_version}")
        return _weight_version


def get_weight_version() -> int:
    """Get the current weight version."""
    return _weight_version


def _get_fp32_layer_prefix() -> str:
    """Get the FP32 layer prefix from environment or use default."""
    return os.environ.get(_FP32_LAYER_PREFIX_ENV, _DEFAULT_FP32_LAYER_PREFIX)


def _is_fp32_target_layer(prefix: str, target_prefix: str) -> bool:
    """Check if a layer prefix matches the target FP32 layer.

    Matches if:
    - prefix ends with target_prefix (e.g., "model.lm_head" matches "lm_head")
    - prefix ends with ".{target_prefix}" for nested paths
    """
    if not target_prefix:
        return False
    prefix_lower = prefix.lower()
    target_lower = target_prefix.lower()
    return prefix_lower.endswith(target_lower) or prefix_lower.endswith(f".{target_lower}")


def string_to_dtype(value: str) -> torch.dtype:
    """Convert string representation to torch dtype."""
    normalized = value.lower().replace("torch.", "").strip()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if normalized not in mapping:
        raise ValueError(
            f"Unsupported dtype string: '{value}'. "
            f"Supported values: {list(mapping.keys())}"
        )
    return mapping[normalized]


def _resolve_dtype_from_config(config: object | None) -> torch.dtype:
    """Best-effort extraction of the default dtype from various config formats."""
    if config is None:
        return torch.bfloat16

    # direct torch dtype
    if isinstance(config, torch.dtype):
        return config

    # string representation
    if isinstance(config, str):
        return string_to_dtype(config)

    # HuggingFace model config
    dtype = getattr(config, "dtype", None)
    if dtype is not None:
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            return string_to_dtype(dtype)

    # check torch_dtype
    torch_dtype = getattr(config, "torch_dtype", None)
    if torch_dtype is not None:
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str):
            return string_to_dtype(torch_dtype)

    # dict based configs
    if isinstance(config, dict):
        for key in ("default_dtype", "torch_dtype", "activation_dtype", "dtype"):
            value = config.get(key)
            if value is None:
                continue
            if isinstance(value, torch.dtype):
                return value
            if isinstance(value, str):
                return string_to_dtype(value)

    return torch.bfloat16


@register_quantization_config("bf16_last_layer_fp32")
class BF16WithLastLayerFP32(QuantizationConfig):
    """Mixed-precision config: BF16 for most layers, FP32 for target layer.
    """

    def __init__(self, config: object | None = None):
        super().__init__()
        activate_fp32_cache()  # Enable cache invalidation for this mode
        self.default_dtype = _resolve_dtype_from_config(config)
        self.fp32_layer_prefix = _get_fp32_layer_prefix()
        self._layer_count = 0
        self._fp32_layer_count = 0
        logger.info(
            f"Initialized {self.get_name()}: default_dtype={self.default_dtype}, "
            f"fp32_layer_prefix='{self.fp32_layer_prefix}'"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"default_dtype={self.default_dtype}, "
            f"fp32_layer_prefix='{self.fp32_layer_prefix}', "
            f"layers={self._layer_count}, "
            f"fp32_layers={self._fp32_layer_count})"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        """Return the appropriate quantization method for the given layer."""
        self._layer_count += 1
        is_fp32_target = _is_fp32_target_layer(prefix, self.fp32_layer_prefix)

        if isinstance(layer, VocabParallelEmbedding):
            if is_fp32_target:
                # Target layer as embedding (e.g., tied weights)
                logger.debug(f"Layer {prefix}: FP32 embedding (target: {self.fp32_layer_prefix})")
                self._fp32_layer_count += 1
                return _FP32UnembedEmbeddingMethod()

            # Regular embedding, leave unmodified
            logger.debug(f"Layer {prefix}: unmodified embedding")
            return None

        if isinstance(layer, LinearBase):
            if is_fp32_target:
                logger.debug(f"Layer {prefix}: FP32 linear (target: {self.fp32_layer_prefix})")
                self._fp32_layer_count += 1
                return _FP32LinearMethod()

            # Regular linear layer, use default dtype
            logger.debug(f"Layer {prefix}: {self.default_dtype} linear")
            return _ForcedDTypeLinearMethod(self.default_dtype)

        # Unknown layer type, leave unmodified
        logger.debug(f"Layer {prefix} ({layer.__class__.__name__}): unmodified (unknown type)")
        return None

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        dtypes = [self.default_dtype]
        if torch.float32 not in dtypes:
            dtypes.append(torch.float32)
        return dtypes

    def get_supported_dtypes(self) -> list[torch.dtype]:
        return self.get_supported_act_dtypes()

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
    def from_config(cls, config: dict | None = None) -> "BF16WithLastLayerFP32":
        return cls(config)


class _ForcedDTypeLinearMethod(UnquantizedLinearMethod):
    """Linear method that enforces a specific parameter dtype."""

    def __init__(self, target_dtype: torch.dtype):
        super().__init__()
        self._target_dtype = target_dtype

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        return super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype=self._target_dtype,
            **extra_weight_attrs,
        )


class _FP32LinearMethod(UnquantizedLinearMethod):
    """Linear method for lm_head: FP32 weights and FP32 computation."""

    def __init__(self):
        super().__init__()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        return super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype=torch.float32,
            **extra_weight_attrs,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply linear transformation with FP32 upcast for inputs."""
        # upcast input to FP32 if needed
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # weight should already be FP32 here from create_weights
        return super().apply(layer, x, bias)


class _FP32UnembedEmbeddingMethod(UnquantizedEmbeddingMethod):
    """Embedding method for tied weights: BF16 storage, FP32 logits computation."""

    # Class-level cache for FP32 weights, keyed by (layer_id, device)
    _fp32_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, int]] = {}
    _cache_lock = threading.Lock()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply embedding lookup or unembedding matmul with FP32 precision."""
        # upcast activations to FP32
        x32 = x if x.dtype == torch.float32 else x.to(torch.float32)
        target_device = x32.device

        # get FP32 weights, use cache if valid
        w32 = self._get_fp32_weight(layer, target_device)

        # Handle bias
        b32 = None
        if bias is not None:
            b32 = bias if bias.dtype == torch.float32 else bias.to(torch.float32)
            if b32.device != target_device:
                b32 = b32.to(target_device)

        return torch.nn.functional.linear(x32, w32, b32)

    def _get_fp32_weight(
        self, layer: torch.nn.Module, target_device: torch.device
    ) -> torch.Tensor:
        """Get FP32 weight tensor, using cache if valid."""
        w_param = layer.weight

        # fast path: weight is already FP32 on correct device
        if w_param.dtype == torch.float32 and w_param.device == target_device:
            return w_param

        layer_id = id(layer)
        cache_key = (layer_id, target_device)
        current_version = get_weight_version()

        with self._cache_lock:
            cached = self._fp32_cache.get(cache_key)
            if cached is not None:
                cached_weight, cached_version = cached
                # Validate cache: same version and shape
                if (
                    cached_version == current_version
                    and cached_weight.shape == w_param.shape
                ):
                    return cached_weight

            # cache miss or invalid - create new FP32 copy
            w32 = w_param.to(device=target_device, dtype=torch.float32)
            self._fp32_cache[cache_key] = (w32, current_version)

            logger.debug(f"Created FP32 weight cache for layer {layer_id} on {target_device} (version {current_version})")
            return w32

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached FP32 weights."""
        with cls._cache_lock:
            cls._fp32_cache.clear()
        logger.debug("FP32 weight cache cleared")


def invalidate_fp32_cache() -> None:
    """Invalidate FP32 weight caches. No-op if bf16_last_layer_fp32 not active."""
    if not _FP32_CACHE_ACTIVE:
        return
    increment_weight_version()
    _FP32UnembedEmbeddingMethod.clear_cache()
    logger.debug("FP32 caches invalidated")
