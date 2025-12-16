"""
Experiment configurations for PipelineRL.

Each config is a dictionary with:
- exp_name: Base experiment name
- config_name: Hydra config file to use (miniwob, workarena, etc.)
- n_gpus: Number of GPUs
- params: Hyperparameters to override
- base_overrides: Pipeline configuration overrides
"""

from .debug import DEBUG_CONFIG
from .main import MAIN_CONFIG


__all__ = [
    "DEBUG_CONFIG",
    "MAIN_CONFIG",

]
