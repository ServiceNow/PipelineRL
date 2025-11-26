"""
Mixed-precision quantization configuration for vLLM.

This module provides a custom quantization config that keeps most layers in bfloat16
for efficiency while forcing the final lm_head layer to float32 for numerical precision
in logits computation. This is particularly important for RL training where small
precision errors in log probabilities can accumulate.

Usage:
    Launch vLLM with: --quantization bf16_last_layer_fp32

    The module must be imported before vLLM initializes to register the config:
        import pipelinerl.vllm_quantization
"""

import logging
import threading
from typing import Callable

import torch
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

logger = logging.getLogger(__name__)

# Global weight version counter for cache invalidation
_weight_version: int = 0
_weight_version_lock = threading.Lock()


def increment_weight_version() -> int:
    """Increment the global weight version counter.

    Call this after weight updates to invalidate cached FP32 copies.
    Returns the new version number.
    """
    global _weight_version
    with _weight_version_lock:
        _weight_version += 1
        logger.debug("Weight version incremented to %d", _weight_version)
        return _weight_version


def get_weight_version() -> int:
    """Get the current weight version."""
    return _weight_version


# Known lm_head layer name patterns across different model architectures
LM_HEAD_PATTERNS = frozenset({
    "lm_head",
    "output",
    "cls.predictions",  # BERT-style
    "classifier",
    "head",
})

# Known embedding layer name patterns for tied weight detection
EMBEDDING_PATTERNS = frozenset({
    "embed_tokens",
    "tok_embeddings",
    "word_embeddings",
    "wte",
    "wpe",  # GPT-2 position embeddings (not tied, but worth knowing)
    "token_embedding",
})


def _is_lm_head_layer(prefix: str) -> bool:
    """Check if a layer prefix indicates an lm_head / output projection layer."""
    prefix_lower = prefix.lower()
    for pattern in LM_HEAD_PATTERNS:
        # Check for exact match at end or as a component
        if prefix_lower.endswith(pattern) or prefix_lower.endswith(f".{pattern}"):
            return True
        # Check for pattern as a path component (with dots on both sides or at start)
        if f".{pattern}." in prefix_lower or prefix_lower.startswith(f"{pattern}."):
            return True
    return False


def _is_tied_embedding(prefix: str) -> bool:
    """Check if a layer prefix indicates an embedding that might be tied to lm_head."""
    prefix_lower = prefix.lower()
    for pattern in EMBEDDING_PATTERNS:
        if pattern in prefix_lower:
            return True
    return False


def _string_to_dtype(value: str) -> torch.dtype:
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

    # Direct torch dtype
    if isinstance(config, torch.dtype):
        return config

    # String representation
    if isinstance(config, str):
        return _string_to_dtype(config)

    # HuggingFace model config or similar objects
    dtype = getattr(config, "dtype", None)
    if dtype is not None:
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            return _string_to_dtype(dtype)

    # Also check torch_dtype (common in HF configs)
    torch_dtype = getattr(config, "torch_dtype", None)
    if torch_dtype is not None:
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        if isinstance(torch_dtype, str):
            return _string_to_dtype(torch_dtype)

    # Dictionary-based configs
    if isinstance(config, dict):
        for key in ("default_dtype", "torch_dtype", "activation_dtype", "dtype"):
            value = config.get(key)
            if value is None:
                continue
            if isinstance(value, torch.dtype):
                return value
            if isinstance(value, str):
                return _string_to_dtype(value)

    return torch.bfloat16


@register_quantization_config("bf16_last_layer_fp32")
class BF16WithLastLayerFP32(QuantizationConfig):
    """
    Mixed-precision configuration: BF16 for most layers, FP32 for lm_head.

    This configuration improves numerical stability in the final logits computation
    while maintaining memory efficiency for the rest of the model. Particularly
    useful for RL training where log probability precision matters.

    Features:
        - Automatic detection of lm_head layers across architectures
        - Support for tied embeddings (shared input/output embeddings)
        - Cache invalidation for weight updates during online RL
        - Configurable default dtype for non-final layers
    """

    def __init__(self, config: object | None = None):
        super().__init__()
        self.default_dtype = _resolve_dtype_from_config(config)
        self._layer_count = 0
        self._fp32_layer_count = 0
        logger.info(
            "Initialized %s: default_dtype=%s, lm_head forced to fp32",
            self.get_name(),
            self.default_dtype,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"default_dtype={self.default_dtype}, "
            f"layers={self._layer_count}, "
            f"fp32_layers={self._fp32_layer_count})"
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> LinearMethodBase | None:
        """Return the appropriate quantization method for the given layer."""
        self._layer_count += 1
        is_lm_head = _is_lm_head_layer(prefix)
        is_tied_embed = _is_tied_embedding(prefix)

        if isinstance(layer, VocabParallelEmbedding):
            if is_lm_head:
                # Explicit lm_head embedding (rare but possible)
                logger.debug(
                    "Layer %s (%s): FP32 embedding (lm_head)",
                    prefix, layer.__class__.__name__
                )
                self._fp32_layer_count += 1
                return _FP32UnembedEmbeddingMethod()

            if is_tied_embed:
                # Tied embedding: keep BF16 storage but compute logits in FP32
                logger.debug(
                    "Layer %s (%s): tied embedding, FP32 logits matmul",
                    prefix, layer.__class__.__name__
                )
                self._fp32_layer_count += 1
                return _FP32UnembedEmbeddingMethod()

            # Regular embedding, leave unmodified
            logger.debug(
                "Layer %s (%s): unmodified embedding",
                prefix, layer.__class__.__name__
            )
            return None

        if isinstance(layer, LinearBase):
            if is_lm_head:
                logger.debug(
                    "Layer %s (%s): FP32 linear (lm_head)",
                    prefix, layer.__class__.__name__
                )
                self._fp32_layer_count += 1
                return _FP32LinearMethod()

            # Regular linear layer, use default dtype
            logger.debug(
                "Layer %s (%s): %s linear",
                prefix, layer.__class__.__name__, self.default_dtype
            )
            return _ForcedDTypeLinearMethod(self.default_dtype)

        # Unknown layer type, leave unmodified
        logger.debug(
            "Layer %s (%s): unmodified (unknown type)",
            prefix, layer.__class__.__name__
        )
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
        # Upcast input to FP32 if needed
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # Weight should already be FP32 from create_weights
        return super().apply(layer, x, bias)


class _FP32UnembedEmbeddingMethod(UnquantizedEmbeddingMethod):
    """
    Embedding method for tied weights: BF16 storage, FP32 logits computation.

    This method keeps embedding weights in their original dtype (typically BF16)
    for memory efficiency during forward embedding lookups, but performs the
    final logits matmul (unembedding) in FP32 for numerical precision.

    The FP32 weight cache is automatically invalidated when the global weight
    version changes (call increment_weight_version() after weight updates).
    """

    # Class-level cache for FP32 weights, keyed by (layer_id, device)
    # Using a class attribute allows proper cleanup and avoids memory leaks
    _fp32_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, int]] = {}
    _cache_lock = threading.Lock()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply embedding lookup or unembedding matmul with FP32 precision."""
        # Upcast activations to FP32
        x32 = x if x.dtype == torch.float32 else x.to(torch.float32)
        target_device = x32.device

        # Get FP32 weights, using cache if valid
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

        # Fast path: weight is already FP32 on correct device
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

            # Cache miss or invalid - create new FP32 copy
            w32 = w_param.to(device=target_device, dtype=torch.float32)
            self._fp32_cache[cache_key] = (w32, current_version)

            logger.debug(
                "Created FP32 weight cache for layer %d on %s (version %d)",
                layer_id, target_device, current_version
            )
            return w32

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached FP32 weights. Call after model changes."""
        with cls._cache_lock:
            cls._fp32_cache.clear()
        logger.debug("FP32 weight cache cleared")


def invalidate_fp32_cache() -> None:
    """Invalidate FP32 weight caches after weight updates.

    Call this function after receiving weight updates in online RL to ensure
    the next forward pass uses fresh FP32 copies of the updated weights.
    """
    increment_weight_version()
    _FP32UnembedEmbeddingMethod.clear_cache()
    logger.debug("FP32 caches invalidated")
