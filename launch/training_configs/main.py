"""
Main training configuration for full experiments.
"""

# Model path (used for conditional vLLM config)
model_path = "Qwen/Qwen3-8B"

MAIN_CONFIG = {
    "exp_name": "workarena_baseline",
    "config_name": "workarena",
    "n_gpus": 8,
    
    "params": {
        # -------------------- MODEL --------------------
        "model_path": model_path,
        
        # -------------------- INFRASTRUCTURE --------------------
        "use_ray": True,
        
        # -------------------- DATASET --------------------
        # Full train/test splits
        "train_dataset_names": ["train"],
        "test_dataset_names": ["test"],
        
        # -------------------- TRAINING --------------------
        "finetune.seq_length": 40_000,  # input + output tokens
        "finetune.max_train_steps": 1000,
        "finetune.learning_rate": 1e-6,
        "finetune.gradient_accumulation_passes": 512,
        # "finetune.save_checkpoint_steps": 100,
        
        # -------------------- RL SPECIFIC --------------------
        "finetune.attempts": 8,
        "finetune.rl.divide_advantage_by_std": False,
        "finetune.rl.filter_zero_advantage_groups": True,
        
        # -------------------- ACTOR/ROLLOUTS --------------------
        "actor.discount_factor": 0.95,
        
        # -------------------- LLM GENERATION (Training) --------------------
        "llm.parameters.max_tokens": 4096,  # output tokens
        "llm.parameters.temperature": 0.7,
        
        # -------------------- LLM GENERATION (Evaluation) --------------------
        "test_llm.parameters.max_tokens": 4096,
        "test_llm.parameters.temperature": 0.7,
        "test_llm.parameters.top_p": 1.0,
        "test_llm.parameters.top_k": 50,
        
        # -------------------- VLLM CONFIG --------------------
        "vllm_config.use_v1": False,
        "vllm_config.vllm_kwargs.max-num-seqs": 256,
        "vllm_config.vllm_kwargs.max-num-batched-tokens": 32000,
        "vllm_config.vllm_kwargs.max_model_len": 16384,
        "vllm_config.vllm_kwargs.gpu-memory-utilization": 0.9,
        
        # -------------------- AGENT --------------------
        "use_generic_agent": True,
        "generic_agent.max_iterations": 15,
        "generic_agent.use_examples": True,
        "generic_agent.max_chars_page_observation": 10000,
        "generic_agent.max_retries": 3,
        "generic_agent.include_think_in_history": False,
        
        # -------------------- EVALUATION --------------------
        "eval_every_n_versions": 1024,
    },
    
    "base_overrides": {
        # World fractions (how to distribute GPUs)
        "world.preprocessor_fraction": 0,
        "world.actor_fraction": 2,
        "world.finetune_fraction": 6,
        # Rollout workers
        "actor.rollout_workers": 128,
    },
}

# Add reasoning parser for Qwen models
if "qwen" in model_path.lower():
    MAIN_CONFIG["params"]["vllm_config.vllm_kwargs.reasoning-parser"] = "qwen3"
