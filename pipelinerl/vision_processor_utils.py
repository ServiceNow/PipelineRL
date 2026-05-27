"""
Vision processor utilities for multimodal models.

This module provides processor caching and management for vision-language models.

Supported models:
- Qwen2.5-VL: Uses image_grid_thw (B, 3) and flattened pixel_values
- Pixtral/Apriel: Uses image_sizes (B, 2) and standard pixel_values (B, C, H, W)
"""
import logging
from typing import Dict
import torch
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

# Processor cache
_processors: Dict[str, AutoProcessor] = {}


def get_mm_processor(model_name: str, mm_processor_kwargs: dict | None = None) -> AutoProcessor:
    """
    Get or create an AutoProcessor for multimodal models.

    Args:
        model_name: HuggingFace model identifier
        mm_processor_kwargs: Optional kwargs to pass to AutoProcessor.from_pretrained()

    Returns:
        AutoProcessor instance
    """
    if model_name not in _processors:
        if mm_processor_kwargs is None:
            mm_processor_kwargs = {}

        logger.info(f"Loading processor for model: {model_name} with kwargs: {mm_processor_kwargs}")
        _processors[model_name] = AutoProcessor.from_pretrained(
            model_name, **mm_processor_kwargs
        )
    return _processors[model_name]


def clear_cache() -> None:
    """Clear all cached processors."""
    _processors.clear()


def collate_visual_features(visual_features_list: list[dict]) -> dict[str, torch.Tensor]:
    """
    Collate visual features from multiple samples into batched tensors.

    Handles different formats:
    - Metadata (image_grid_thw, image_sizes): Concatenate along image dimension
    - Qwen pixel_values (2D): Concatenate flattened features
    - Pixtral pixel_values (4D): Pad to max_num_images
    - Other features: Pad to max_num_images

    Args:
        visual_features_list: List of visual feature dicts from individual samples

    Returns:
        Dict mapping feature names to batched tensors
    """
    if not visual_features_list or visual_features_list[0] is None:
        return {}

    first_vf = visual_features_list[0]
    batched_visual_features = {}

    for key in first_vf.keys():
        if key in ("image_grid_thw", "image_sizes"):
            # Concatenate metadata arrays (image_grid_thw or image_sizes)
            # Each sample has shape (num_images, 2 or 3), concatenate along image dimension
            all_metadata = [torch.as_tensor(vf[key]) for vf in visual_features_list]
            batched_visual_features[key] = torch.cat(all_metadata, dim=0)

        elif key == "pixel_values":
            # Handle pixel_values - format differs by model:
            # - Qwen: (total_pixels, hidden_dim) - flattened, concatenate along pixel dimension
            # - Pixtral: (num_images, C, H, W) - standard, needs padding to max_num_images
            all_tensors = [torch.as_tensor(vf[key]) for vf in visual_features_list]

            # Check if this is flattened format (2D) or image format (4D)
            if all_tensors[0].ndim == 2:
                # Qwen format: (total_pixels, hidden_dim) - just concatenate
                batched_visual_features[key] = torch.cat(all_tensors, dim=0)
            elif all_tensors[0].ndim == 4:
                # Pixtral format: (num_images, C, H, W) - pad to max_num_images
                max_num_images = max(t.shape[0] for t in all_tensors)
                single_shape = all_tensors[0].shape[1:]  # (C, H, W)
                dtype = all_tensors[0].dtype

                # Pre-allocate: (batch_size, max_num_images, C, H, W)
                batch_shape = (len(all_tensors), max_num_images) + single_shape
                batched = torch.zeros(batch_shape, dtype=dtype)

                # Fill in actual data
                for i, tensor in enumerate(all_tensors):
                    num_images = tensor.shape[0]
                    batched[i, :num_images] = tensor

                batched_visual_features[key] = batched
            else:
                raise ValueError(f"Unexpected pixel_values shape: {all_tensors[0].shape}")

        else:
            # Other visual features - assume they need padding like Pixtral pixel_values
            all_tensors = [torch.as_tensor(vf[key]) for vf in visual_features_list]
            max_num_images = max(t.shape[0] for t in all_tensors)
            single_shape = all_tensors[0].shape[1:]
            dtype = all_tensors[0].dtype

            batch_shape = (len(all_tensors), max_num_images) + single_shape
            batched = torch.zeros(batch_shape, dtype=dtype)

            for i, tensor in enumerate(all_tensors):
                num_images = tensor.shape[0]
                batched[i, :num_images] = tensor

            batched_visual_features[key] = batched

    return batched_visual_features
