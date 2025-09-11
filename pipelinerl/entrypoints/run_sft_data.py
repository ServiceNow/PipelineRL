#!/usr/bin/env python3
"""
Entrypoint for running SFT data processing.

This script processes existing datasets into TrainingText format for supervised fine-tuning.
"""

import logging
import hydra
from omegaconf import DictConfig

from pipelinerl.sft_data import run_sft_data_loop

logger = logging.getLogger(__name__)


@hydra.main(
    config_path="../../conf/",
    config_name="base",
    version_base="1.3.2",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for SFT data processing."""
    logger.info("Starting SFT data processing...")
    logger.info(f"Output: {cfg.output_dir}, Model: {cfg.model_path}")
    
    try:
        run_sft_data_loop(cfg)
        logger.info("SFT data processing completed successfully")
    except Exception as e:
        logger.error(f"SFT data processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
