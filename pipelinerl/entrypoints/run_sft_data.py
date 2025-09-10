#!/usr/bin/env python3
"""
Entrypoint for running SFT data processing.

This script processes existing datasets into TrainingText format for supervised fine-tuning.
"""

import hydra
from omegaconf import DictConfig

from pipelinerl.sft_data import run_sft_data_loop


@hydra.main(
    config_path="../../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for SFT data processing."""
    run_sft_data_loop(cfg)


if __name__ == "__main__":
    main()
