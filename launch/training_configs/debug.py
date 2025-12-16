"""
Debug configuration for quick testing.

This config uses debug_train/debug_test datasets and minimal training steps
to verify the entire pipeline is working. Should complete in ~5 minutes.
"""

# Model path (used for conditional vLLM config)
model_path = "Qwen/Qwen3-8B"

DEBUG_CONFIG = {
    "exp_name": "debug_run",
    "config_name": "miniwob",
    "n_gpus": 8,
    
    "params": {
        # -------------------- MODEL --------------------
        "model_path": model_path,
        
        # -------------------- DATASET --------------------
        # Use debug splits (small subset for quick testing)
        "train_dataset_names": ["debug_train"],
        "test_dataset_names": ["debug_test"],
        
        # -------------------- TRAINING --------------------
        # Minimal training for quick validation
        "finetune.learning_rate": 1e-6,
        "finetune.gradient_accumulation_passes": 16,  # Small batch for quick iterations
        "finetune.max_train_steps": 2,  # Just 2 steps to verify pipeline
        "finetune.save_checkpoint_steps": 1,  # Save after each step
        
        # -------------------- RL SPECIFIC --------------------
        "finetune.attempts": 4,  # Fewer attempts for speed
        "finetune.rl.divide_advantage_by_std": True,
        "finetune.rl.filter_zero_advantage_groups": True,
        
        # -------------------- ACTOR/ROLLOUTS --------------------
        "actor.discount_factor": 0.95,
        
        # -------------------- LLM GENERATION --------------------
        "llm.parameters.temperature": 0.25,
        
        # -------------------- AGENT --------------------
        "use_generic_agent": True,
        "generic_agent.include_think_in_history": True,
    },
    
    "base_overrides": {
        # World fractions (how to distribute GPUs)
        "world.preprocessor_fraction": 0,
        "world.actor_fraction": 2,
        "world.finetune_fraction": 6,
        # Fewer rollout workers for debug
        "actor.rollout_workers": 32,
    },
}

# Add reasoning parser for Qwen models
if "qwen" in model_path.lower():
    DEBUG_CONFIG["params"]["vllm_config.vllm_kwargs.reasoning-parser"] = "qwen3"


