#!/bin/bash
# 4-node interactive smoke run: DeepSpeed ZeRO-3 trainer + vLLM v1 + PPO loss
# -----------------------------------------------------------------------------
# Mirrors submit_eai_math_7b_multinode_ds_vllm_v1.sh but runs in your current
# shell instead of submitting an `eai job new`. Use this as the reference path
# when comparing fast-llm behavior against the established DeepSpeed trainer.
#
# Prereqs (one-time, see ../../README.md "Install FastLLM+PipelineRL"):
#   - Image: registry.toolkit-sp.yul201.service-now.com/snow.research.afm/
#            interactive-toolkit:25.12-py3-vllm014rc1redis
#   - PipelineRL editable-installed in /home/toolkit/code/PipelineRL/.venv
#   - Qwen2.5-7B at /home/toolkit/Qwen2.5-7B
#
# Success looks like:
#   - finetune/stderr_node0.log shows
#       "pipelinerl.finetune_loop - Completed steps 1: {...}"
#     followed by "Completed steps 2" and "Reached final step 2, stopping."
#   - With MAX_TRAIN_STEPS=2 (default) the run finishes in ~10 min.
#
# Where logs go:
#   $RESULTS_DIR/$EXP_NAME/{launch.log, finetune/stderr_node*.log,
#                          actor/info.log, actor_vllm_*/stdout.log}
#
# NOTE: DS uses streams=files (default) and prints step metrics to STDERR.
# Don't confuse the empty stdout with a stalled trainer — check stderr.
#
# Override knobs (env vars):
#   NODES             default 4
#   MAX_TRAIN_STEPS   default 2
#   MODEL_PATH        default /home/toolkit/Qwen2.5-7B
#   RESULTS_DIR       default /mnt/shared/denis/math_7b_results
#   WANDB_ENTITY      default denisko-se
#   WANDB_PROJECT     default watermelon
# -----------------------------------------------------------------------------

set -euo pipefail

NODES="${NODES:-4}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2}"
MODEL_PATH="${MODEL_PATH:-/home/toolkit/Qwen2.5-7B}"
RESULTS_DIR="${RESULTS_DIR:-/mnt/shared/denis/math_7b_results}"
WANDB_ENTITY="${WANDB_ENTITY:-denisko-se}"
WANDB_PROJECT="${WANDB_PROJECT:-watermelon}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="math_7b_${NODES}node_ds_interactive_${TIMESTAMP}"
EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"

mkdir -p "${EXP_DIR}"
cd /home/toolkit/code/PipelineRL
# shellcheck disable=SC1091
source /home/toolkit/code/PipelineRL/.venv/bin/activate

echo "=== DeepSpeed 4-node interactive smoke ==="
echo "  NODES=${NODES}  MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "  EXP_DIR=${EXP_DIR}"
echo "==========================================="

PYTHONHASHSEED=42 python -m pipelinerl.launch \
  --config-path /home/toolkit/code/PipelineRL/conf \
  --config-name math \
  "output_dir=${EXP_DIR}" \
  "wandb.wandb_workspace_root=${RESULTS_DIR}" \
  "wandb.wandb_entity_name=${WANDB_ENTITY}" \
  "wandb.wandb_project_name=${WANDB_PROJECT}" \
  wandb.wandb_group=eai_math7b_ds_fastllm \
  "+wandb.wandb_run_name=math7b_ds_interactive_${NODES}node_${TIMESTAMP}" \
  use_fast_llm=false \
  actor.llm_max_rollouts=128 \
  force_restart=true \
  finetune.learning_rate=1e-6 \
  finetune.attempts=8 \
  finetune.rl.policy_loss=ppo \
  finetune.rl.epsilon_low=2e-2 \
  finetune.rl.epsilon_high=2e-2 \
  '+finetune.rl.filter_zero_advantage_groups=true' \
  "finetune.max_train_steps=${MAX_TRAIN_STEPS}" \
  finetune.seq_length=20000 \
  finetune.gradient_accumulation_passes=1024 \
  'vllm_config.vllm_kwargs.max_model_len=20000' \
  'llm.parameters.max_tokens=16000' \
  'llm.parameters.temperature=0.7' \
  'test_llm.parameters.max_tokens=16000' \
  'test_llm.parameters.temperature=0.7' \
  world.actor_fraction=4 \
  world.preprocessor_fraction=0 \
  world.finetune_fraction=4 \
  "world.run_id=\${MASTER_ADDR}" \
  streams=files \
  eval_every_n_versions=0 \
  "model_path=${MODEL_PATH}"
