#!/usr/bin/env bash
set -euo pipefail

seq_len=32000
max_tokens=16000
temperature=0.7
seq_parallel=4
output_dir_base=results
job_name=cube-ft
filter_zero_advantage_groups=true
max_train_steps=50
train_subset_end=-1
test_subset_end=-1
gradient_accumulation_passes=1024
attempts=8
wandb=true

uv run python -m pipelinerl.launch --config-name cube_math_tool.yaml \
output_dir=$output_dir_base/$job_name \
actor.llm_max_rollouts=32 \
actor.cube_actor_num_cpus=0.5 \
force_restart=true \
fp32_lm_head=true \
finetune.learning_rate=1e-6 \
finetune.attempts=$attempts \
finetune.rl.policy_loss=gspo \
finetune.rl.epsilon_low=3e-3 \
finetune.rl.epsilon_high=4e-3 \
+finetune.rl.filter_zero_advantage_groups=$filter_zero_advantage_groups \
finetune.max_train_steps=$max_train_steps \
finetune.seq_length=$seq_len \
finetune.seq_parallel=$seq_parallel \
finetune.gradient_accumulation_passes=$gradient_accumulation_passes \
vllm_config.vllm_kwargs.max_model_len=$seq_len \
llm.parameters.max_tokens=$max_tokens \
llm.parameters.temperature=$temperature \
llm.parameters.max_completion_tokens=$max_tokens \
+llm.parameters.max_model_len=$seq_len \
test_llm.parameters.max_tokens=$max_tokens \
test_llm.parameters.temperature=$temperature \
test_llm.parameters.max_completion_tokens=$max_tokens \
+test_llm.parameters.max_model_len=$seq_len \
world.actor_fraction=4 \
world.preprocessor_fraction=0 \
world.finetune_fraction=4 \
streams=files \
eval_every_n_versions=1 \
model_path=Qwen/Qwen3-4B-Instruct-2507 \
vllm_config.vllm_kwargs.served_model_name=Qwen3-4B-Instruct-2507 \
wandb.wandb_workspace_root=$output_dir_base \
wandb.wandb_project_name=watermelon \
wandb.use_wandb=$wandb \
+train_subset.begin=0 \
+train_subset.end=$train_subset_end \
+test_subset.begin=0 \
+test_subset.end=$test_subset_end \