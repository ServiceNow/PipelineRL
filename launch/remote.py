"""
Remote job launching for PipelineRL experiments on EAI cluster.
"""

import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .config import (
    TEAM_ACCOUNT_NAME,
    PERSONAL_ACCOUNT_NAME_SUFFIX,
    RUN_IMAGE,
    DATA_MOUNTS,
    HOME_DATA_NAME,
    HOME_MOUNT_PATH,
    PIPELINERL_PATH,
    TMP_SCRIPT_DIR,
    DEFAULT_RESOURCES,
    DEFAULT_MAX_RUN_TIME,
    SNOW_INSTANCE_POOL,
    OUTPUT_DIR_ROOT,
)
from .utils import run_command, get_job_status, kill_job, get_job_id_by_name

logger = logging.getLogger(__name__)


# Fix the miniwob env launcher!!!


# Base command defaults for experiments (no debug.mode = full pipeline)
DEFAULT_BASE_OVERRIDES = {
    "world.preprocessor_fraction": 0,
    "world.actor_fraction": 2,
    "world.finetune_fraction": 6,
    "actor.rollout_workers": 128,
}


def create_launch_script(
    exp_name: str,
    config_name: str = "miniwob",
    config_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
) -> Path:
    """
    Create a bash script to launch PipelineRL experiment.
    
    Args:
        exp_name: Name of the experiment (should already include timestamp)
        config_name: Hydra config name (e.g., 'miniwob', 'workarena')
        config_overrides: Dictionary of config overrides
        output_dir: Output directory for results
    
    Returns:
        Path to the created script
    """
    TMP_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    
    script_name = f"{exp_name}.sh"
    script_path = TMP_SCRIPT_DIR / script_name
    
    # Build config overrides string
    overrides_str = ""
    if config_overrides:
        for key, value in config_overrides.items():
            # Skip empty string values (Hydra doesn't like debug.mode=)
            if value == "":
                continue
            if isinstance(value, str):
                overrides_str += f' {key}="{value}"'
            else:
                overrides_str += f" {key}={value}"
    
    # Set output directory
    if output_dir is None:
        output_dir = f"{OUTPUT_DIR_ROOT}/results/{config_name}/{exp_name}"
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    script_content = f"""#!/bin/bash
set -e

# Change to PipelineRL directory
cd {PIPELINERL_PATH}

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pipeline-rl

# Fix for "corrupted size vs. prev_size" memory corruption with DeepSpeed/Ray/PyTorch
export DS_ACCELERATOR=cuda
export MALLOC_CHECK_=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Environment-specific setup
if [[ "{config_name}" == *"miniwob"* ]]; then
    # MiniWoB: Set URL and start HTTP server
    export MINIWOB_URL="http://0.0.0.0:8000/miniwob/"
    echo "Starting MiniWoB HTTP server..."
    cd /mnt/adea/data/finetuning/.miniwob-plusplus/miniwob/html
    python -m http.server &
    HTTP_SERVER_PID=$!
    echo "MiniWoB HTTP server started with PID: $HTTP_SERVER_PID"
    sleep 2
    cd {PIPELINERL_PATH}
    trap "echo 'Killing MiniWoB HTTP server...'; kill $HTTP_SERVER_PID 2>/dev/null" EXIT
elif [[ "{config_name}" == *"workarena"* ]]; then
    # WorkArena: Set ServiceNow instance pool path
    export SNOW_INSTANCE_POOL="{SNOW_INSTANCE_POOL}"
    echo "SNOW_INSTANCE_POOL set to: $SNOW_INSTANCE_POOL"
fi

echo "Starting PipelineRL experiment: {exp_name}"
echo "Config: {config_name}"
echo "Output: {output_dir}"

# Launch PipelineRL
python -m pipelinerl.launch \\
    --config-name {config_name} \\
    output_dir={output_dir} \\
    wandb.wandb_workspace_root=results/{config_name}{overrides_str}

echo "Experiment completed!"
"""
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    logger.info(f"Created launch script: {script_path}")
    
    return script_path


def launch_experiment_remote(
    exp_name: str,
    config_name: str = "miniwob",
    config_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    n_gpus: int = 1,
    gpu_mem: int = 80,
    n_cpus: Optional[int] = None,
    mem: Optional[int] = None,
    max_run_time: int = DEFAULT_MAX_RUN_TIME,
) -> str:
    """
    Launch a PipelineRL experiment on the EAI cluster.
    
    Args:
        exp_name: Name of the experiment (should already include timestamp)
        config_name: Hydra config name (e.g., 'miniwob', 'workarena')
        config_overrides: Dictionary of config overrides
        output_dir: Output directory for results
        n_gpus: Number of GPUs
        gpu_mem: GPU memory in GB
        n_cpus: Number of CPUs (auto-calculated if None)
        mem: Memory in GB (auto-calculated if None)
        max_run_time: Maximum run time in seconds
    
    Returns:
        Job ID of the launched job
    """
    # Auto-calculate CPUs and memory based on GPUs (24 CPUs per GPU, ~188GB mem per GPU)
    if n_cpus is None:
        n_cpus = n_gpus * 24
    if mem is None:
        mem = n_gpus * 188
    
    # Create the launch script
    script_path = create_launch_script(
        exp_name=exp_name,
        config_name=config_name,
        config_overrides=config_overrides,
        output_dir=output_dir,
    )
    
    # Build job name (exp_name already has timestamp)
    job_name = f"{PERSONAL_ACCOUNT_NAME_SUFFIX}_{exp_name}"
    
    # Build the eai job command
    launch_command = f"""eai job new \\
        --name {job_name} \\
        --account {TEAM_ACCOUNT_NAME} \\
        --image {RUN_IMAGE} \\
        --gpu {n_gpus} \\
        --gpu-mem {gpu_mem} \\
        --cpu {n_cpus} \\
        --mem {mem} \\
        --max-run-time {max_run_time}"""
    
    # Add data volume mounts (read-only and read-write)
    for volume_name, mount_path, mode in DATA_MOUNTS:
        if mode == "ro":
            launch_command += f" \\\n        --data {volume_name}:{mount_path}:ro"
        else:
            launch_command += f" \\\n        --data {volume_name}:{mount_path}"
    
    # Add home directory mount
    launch_command += f" \\\n        --data {HOME_DATA_NAME}:{HOME_MOUNT_PATH}"
    
    # Add environment variables (config-specific)
    launch_command += f" \\\n        --env HOME={HOME_MOUNT_PATH}"
    if "miniwob" in config_name:
        launch_command += f" \\\n        --env MINIWOB_URL=http://0.0.0.0:8000/miniwob/"
    elif "workarena" in config_name:
        launch_command += f" \\\n        --env SNOW_INSTANCE_POOL={SNOW_INSTANCE_POOL}"
    
    launch_command += f""" \\
        --restartable \\
        -- bash {script_path}"""
    
    # Print the command being executed
    print(f"\nEAI Launch Command:\n{launch_command}\n")
    
    # Execute the command
    result = run_command(launch_command)
    
    # Get the job ID
    import time
    time.sleep(2)
    job_id = get_job_id_by_name(job_name)
    
    if job_id:
        print(f"\n✓ Successfully launched job: {job_id}")
        print(f"\nCheck the logs at: eai job logs -f {job_id}")
        return job_id
    else:
        raise RuntimeError("Failed to get job ID after launching")





def launch_run(
    exp_name: str,
    params: Dict[str, Any],
    config_name: str = "miniwob",
    base_overrides: Optional[Dict[str, Any]] = None,
    n_gpus: int = 8,
    dry_run: bool = False,
) -> Optional[str]:
    """
    Simple function to launch a single run with given parameters.
    
    Args:
        exp_name: Name of the experiment
        params: Dictionary of hyperparameters to override
        config_name: Hydra config name
        base_overrides: Base config overrides (defaults to DEFAULT_BASE_OVERRIDES)
        n_gpus: Number of GPUs
        dry_run: If True, print command but don't execute
    
    Returns:
        Job ID if launched, None if dry_run
    
    Example:
        launch_run(
            exp_name="my_exp",
            params={
                "finetune.learning_rate": 5e-6,
                "llm.parameters.temperature": 0.3,
            }
        )
    """
    # Merge base overrides with params
    if base_overrides is None:
        base_overrides = DEFAULT_BASE_OVERRIDES.copy()
    
    config_overrides = {**base_overrides, **params}
    
    # Create timestamp for unique naming (alphanumeric lowercase with underscores only)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_exp_name = f"{exp_name}_{timestamp}".lower()
    
    # Set output directory and wandb workspace root to match
    workspace_root = f"{OUTPUT_DIR_ROOT}/{config_name}"
    output_dir = f"{workspace_root}/{full_exp_name}"
    config_overrides["wandb.wandb_workspace_root"] = workspace_root
    
    print(f"\n{'='*60}")
    print(f"Launching: {full_exp_name}")
    print(f"Config: {config_name}")
    print(f"Output: {output_dir}")
    print(f"GPUs: {n_gpus}")
    print(f"Parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")
    
    if dry_run:
        print("[DRY RUN] Would launch with these settings")
        # Still show what command would be run
        from .config import TEAM_ACCOUNT_NAME, DATA_MOUNTS, HOME_DATA_NAME, HOME_MOUNT_PATH, RUN_IMAGE, PERSONAL_ACCOUNT_NAME_SUFFIX
        job_name = f"{PERSONAL_ACCOUNT_NAME_SUFFIX}_{full_exp_name}"
        print(f"\nJob name would be: {job_name}")
        return None
    
    job_id = launch_experiment_remote(
        exp_name=full_exp_name,
        config_name=config_name,
        config_overrides=config_overrides,
        output_dir=output_dir,
        n_gpus=n_gpus,
    )
    
    return job_id

