#!/usr/bin/env python3
"""
Launch PipelineRL experiments on EAI cluster.

Usage:
    python launch/run_sweep.py              # Dry run (preview)
    python launch/run_sweep.py --execute    # Actually launch jobs
"""

import sys
from pathlib import Path

# Add PipelineRL root to path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from launch.remote import launch_run

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Set to True to actually launch jobs, False for dry run
execute = True

# Base experiment name
exp_name = "miniwob_baseline"

# Config file to use (miniwob, workarena, etc.)
config_name = "miniwob"

# Number of GPUs
n_gpus = 8

# ============================================================
# HYPERPARAMETERS - Edit these to configure your experiment
# ============================================================

# Model path (used for conditional vLLM config)
model_path = "meta-llama/Llama-3.1-8B-Instruct"

params = {
    # -------------------- MODEL --------------------
    # Base model to finetune
    "model_path": model_path,
    
    # -------------------- DATASET --------------------
    # Dataset splits to use (train/test, easy_train/easy_test, debug_train/debug_test)
    "train_dataset_names": ["train"],
    "test_dataset_names": ["test"],
    # Seeds for dataset loading
    # "dataset_loader_params.train_seeds": [0],
    # "dataset_loader_params.test_seeds": [0],
    
    # -------------------- TRAINING --------------------
    # Learning rate
    "finetune.learning_rate": 1e-6,
    # Max training steps
    # "finetune.max_train_steps": 1000,
    # Gradient accumulation (effective batch size = train_batch_size * gradient_accumulation_passes)
    "finetune.gradient_accumulation_passes": 512,
    # Train batch size
    # Sequence length (input + output tokens)
    # "finetune.seq_length": 16384,
    # Checkpoint saving interval
    # "finetune.save_checkpoint_steps": 100,
    
    # -------------------- RL SPECIFIC --------------------
    # Number of attempts per problem (GRPO group size)
    "finetune.attempts": 8,
    # Divide advantage by std
    "finetune.rl.divide_advantage_by_std": True,
    # Filter out groups where all raw_advantages are zero
    "finetune.rl.filter_zero_advantage_groups": True,

    
    # -------------------- ACTOR/ROLLOUTS --------------------
    # Discount factor for rewards
    "actor.discount_factor": 0.95,
    # Number of rollout workers
    # "actor.rollout_workers": 32,  # Set in base_overrides
    # Max concurrent LLM rollouts
    # "actor.llm_max_rollouts": 256,
    
    # -------------------- LLM GENERATION --------------------
    # Temperature for sampling during training
    "llm.parameters.temperature": 0.25,
    # Max output tokens
    # "llm.parameters.max_tokens": 4096,
    # Test-time temperature (usually lower)
    # "test_llm.parameters.temperature": 0.1,
    
    # -------------------- AGENT --------------------
    # Use GenericWebAgent (true) vs TapeAgents Agent (false)
    "use_generic_agent": True,
    "generic_agent.include_think_in_history": True,
    # Rollout timeout in seconds
    # "rollout_timeout": 600,
    
    # -------------------- EVALUATION --------------------
    # Evaluate every N gradient updates (effective batch size units)
    # "eval_every_n_versions": 1024,
    
    # -------------------- VLLM --------------------
    # Max sequences for vLLM
    # "vllm_config.vllm_kwargs.max-num-seqs": 256,
    # GPU memory utilization
    # "vllm_config.vllm_kwargs.gpu-memory-utilization": 0.9,
}

# Add reasoning parser for Qwen models
if "qwen" in model_path.lower():
    params["vllm_config.vllm_kwargs.reasoning-parser"] = "qwen3"

# ============================================================
# BASE OVERRIDES (pipeline configuration, usually don't change)
# ============================================================

base_overrides = {
    # World fractions (how to distribute GPUs)
    "world.preprocessor_fraction": 0,
    "world.actor_fraction": 2,
    "world.finetune_fraction": 6,
    # Rollout workers
    "actor.rollout_workers": 128,
}

# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Actually launch (default: dry run)")
    args = parser.parse_args()
    
    dry_run = not (execute or args.execute)
    
    launch_run(
        exp_name=exp_name,
        params=params,
        config_name=config_name,
        base_overrides=base_overrides,
        n_gpus=n_gpus,
        dry_run=dry_run,
    )
