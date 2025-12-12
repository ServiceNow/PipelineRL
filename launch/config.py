"""
Configuration for EAI cluster launching.

Edit personal_config.py to set your personal settings.
"""

import os
from pathlib import Path

# Import personal settings
from .personal_config import (
    PERSONAL_ACCOUNT_NAME_SUFFIX,
    TEAM_ACCOUNT_NAME_SUFFIX,
    DATAS_TO_MOUNT,
    HOME_DATA_NAME_SUFFIX,
    HOME_TEAM_ACCOUNT_NAME_SUFFIX,
    HOME_MOUNT_PATH,
    PIPELINERL_PATH as PIPELINERL_PATH_STR,
)

# ============================================================
# DERIVED SETTINGS (usually don't need to change)
# ============================================================

# Team account
TEAM_ACCOUNT_NAME = f"snow.research.{TEAM_ACCOUNT_NAME_SUFFIX}"
HOME_TEAM_ACCOUNT_NAME = f"snow.research.{HOME_TEAM_ACCOUNT_NAME_SUFFIX}"
HOME_DATA_NAME = f"{HOME_TEAM_ACCOUNT_NAME}.{HOME_DATA_NAME_SUFFIX}"

# Output directory for experiment results (on the read-write data mount)
OUTPUT_DIR_ROOT = f"/mnt/{TEAM_ACCOUNT_NAME_SUFFIX}/data_rw/finetuning/{PERSONAL_ACCOUNT_NAME_SUFFIX}"

# Data paths and volumes
DATA_ROOT_PATHS = [f"/mnt/{suffix}/data" for suffix in DATAS_TO_MOUNT]
DATA_ROOT_RW_PATHS = [f"/mnt/{suffix}/data_rw" for suffix in DATAS_TO_MOUNT]
DATA_VOLUMES = [f"snow.research.{suffix}.data" for suffix in DATAS_TO_MOUNT]
DATA_VOLUMES_RW = [f"snow.research.{suffix}.data" for suffix in DATAS_TO_MOUNT]

# Build data mounts list: [(volume, mount_path, mode), ...]
DATA_MOUNTS = []
for data_path, data_volume in zip(DATA_ROOT_PATHS, DATA_VOLUMES):
    DATA_MOUNTS.append((data_volume, data_path, "ro"))
for data_rw_path, data_volume_rw in zip(DATA_ROOT_RW_PATHS, DATA_VOLUMES_RW):
    DATA_MOUNTS.append((data_volume_rw, data_rw_path, "rw"))

# Docker image (same format as toolkit_configs.py)
RUN_IMAGE = f"registry.toolkit-sp.yul201.service-now.com/{TEAM_ACCOUNT_NAME}/ui_copilot_playwright"

# Paths
SNOW_INSTANCE_POOL="/mnt/adea/data_rw/finetuning/snow_instance_pool.json"
PIPELINERL_PATH = Path(PIPELINERL_PATH_STR)
TMP_SCRIPT_DIR = Path(HOME_MOUNT_PATH) / "tmp" / "pipelinerl_launch"

# Default resource settings (24 CPUs per GPU, ~188GB mem per GPU)
DEFAULT_RESOURCES = {
    "1gpu": {"n_gpus": 1, "gpu_mem": 80, "n_cpus": 24, "mem": 188},
    "2gpu": {"n_gpus": 2, "gpu_mem": 80, "n_cpus": 48, "mem": 375},
    "4gpu": {"n_gpus": 4, "gpu_mem": 80, "n_cpus": 96, "mem": 750},
    "8gpu": {"n_gpus": 8, "gpu_mem": 80, "n_cpus": 192, "mem": 1500},
}

# Default max run time (4 days in seconds)
DEFAULT_MAX_RUN_TIME = 4 * 24 * 60 * 60
