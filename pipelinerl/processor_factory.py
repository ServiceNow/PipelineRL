"""Simple cache for AutoProcessor instances."""
from typing import Dict
from transformers import AutoProcessor
import logging
logger = logging.getLogger(__name__)

_processors: Dict[str, AutoProcessor] = {}

def get_processor(model_name: str) -> AutoProcessor:
    """Get or create an AutoProcessor for the given model."""
    if model_name not in _processors:
        logger.info(f"Loading processor for model: {model_name}")
        #TODO: should be args
        _processors[model_name] = AutoProcessor.from_pretrained(model_name, min_pixels=28*28, max_pixels=1280*28*28)
    return _processors[model_name]

def clear_cache() -> None:
    """Clear all cached processors."""
    _processors.clear()