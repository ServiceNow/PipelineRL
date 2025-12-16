#!/usr/bin/env python3
"""
Launch PipelineRL experiments on EAI cluster.

Usage:
    python launch/launch_job.py
    
Edit CONFIG and DRY_RUN below to change what gets launched.
"""

import sys
from pathlib import Path

# Add PipelineRL root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from launch.remote import launch_run
from launch.training_configs import DEBUG_CONFIG, MAIN_CONFIG

# ============================================================
# CONFIGURATION - Edit these to change what gets launched
# ============================================================

CONFIG = MAIN_CONFIG  # Options: DEBUG_CONFIG, MAIN_CONFIG
DRY_RUN = False        # Set to True to preview without launching


def main():
    print(f"\n{'='*60}")
    print(f"Launching: {CONFIG['exp_name']}")
    print(f"Dry run: {DRY_RUN}")
    print(f"{'='*60}")
    
    launch_run(
        exp_name=CONFIG["exp_name"],
        params=CONFIG["params"],
        config_name=CONFIG["config_name"],
        base_overrides=CONFIG["base_overrides"],
        n_gpus=CONFIG["n_gpus"],
        dry_run=DRY_RUN,
    )


if __name__ == "__main__":
    main()
