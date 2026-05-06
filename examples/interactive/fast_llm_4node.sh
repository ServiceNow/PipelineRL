#!/bin/bash
# 4-node interactive smoke run: fast-llm trainer + vLLM v1 + GSPO loss
# -----------------------------------------------------------------------------
# Mirrors submit_eai_math_7b_multinode.sh but runs in your current shell instead
# of submitting an `eai job new`. Use this from inside an interactive EAI
# session that already has 4 nodes attached.
#
# Prereqs (one-time, see ../../README.md "Install FastLLM+PipelineRL"):
#   - Image: registry.toolkit-sp.yul201.service-now.com/snow.research.afm/
#            interactive-toolkit:25.12-py3-vllm014rc1redis
#   - Fast-LLM checked out on the `gspo` branch, editable-installed in
#     /home/toolkit/code/PipelineRL/.venv (alongside PipelineRL on `fast-llm`)
#   - Qwen2.5-7B at /home/toolkit/Qwen2.5-7B
#   - WandB credentials configured for the entity below
#
# Success looks like:
#   - finetune/stdout_node0.log shows "[Rank 00] training @ step 1/N | ... | grad norm: 0.1-0.3"
#   - actor/info.log shows weights_ready events and rollouts being collected
#   - With MAX_TRAIN_STEPS=2 (default) the run finishes in ~10 min and saves a
#     checkpoint at iteration 2.
#
# Where logs go:
#   $RESULTS_DIR/$EXP_NAME/{launch.log, finetune/stdout_node*.log,
#                          actor/info.log, actor_vllm_*/stdout.log}
#
# Override knobs (env vars):
#   NODES             default 4
#   MAX_TRAIN_STEPS   default 2  (smoke run; bump for real training)
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
EXP_NAME="math_7b_${NODES}node_fastllm_gspo_interactive_${TIMESTAMP}"
EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"

mkdir -p "${EXP_DIR}"
cd /home/toolkit/code/PipelineRL
# shellcheck disable=SC1091
source /home/toolkit/code/PipelineRL/.venv/bin/activate

echo "=== fast-llm 4-node interactive smoke ==="
echo "  NODES=${NODES}  MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "  EXP_DIR=${EXP_DIR}"
echo "=========================================="

PYTHONHASHSEED=42 python -m pipelinerl.launch \
  --config-path /home/toolkit/code/PipelineRL/conf \
  --config-name math \
  streams=redis \
  world.actor_fraction=4 \
  world.preprocessor_fraction=0 \
  world.finetune_fraction=4 \
  "world.run_id=\${MASTER_ADDR}" \
  "model_path=${MODEL_PATH}" \
  "output_dir=${EXP_DIR}" \
  force_restart=true \
  actor.llm_max_rollouts=128 \
  finetune.attempts=8 \
  "finetune.max_train_steps=${MAX_TRAIN_STEPS}" \
  '+finetune.rl.filter_zero_advantage_groups=true' \
  eval_every_n_versions=0 \
  "wandb.wandb_workspace_root=${RESULTS_DIR}" \
  "wandb.wandb_entity_name=${WANDB_ENTITY}" \
  "wandb.wandb_project_name=${WANDB_PROJECT}" \
  wandb.wandb_group=eai_math7b_fastllm_gspo \
  "+wandb.wandb_run_name=math7b_fastllm_gspo_interactive_${NODES}node_${TIMESTAMP}" \
  'vllm_config.vllm_kwargs.gpu-memory-utilization=0.85' \
  'vllm_config.vllm_kwargs.max-num-batched-tokens=8192' \
  'vllm_config.vllm_kwargs.max_model_len=20000' \
  'llm.parameters.max_tokens=16000' \
  'llm.parameters.temperature=0.7' \
  'test_llm.parameters.max_tokens=16000' \
  'test_llm.parameters.temperature=0.7' \
  'fast_llm.data.micro_batch_size=20000' \
  '+fast_llm.schedule.docs_per_step=1024' \
  "fast_llm.training.train_iters=${MAX_TRAIN_STEPS}" \
  'fast_llm.training.num_workers=1' \
  'fast_llm.training.checkpoint.interval=20' \
  'fast_llm.model.distributed.sequence_data_parallel=2' \
  '+fast_llm.model.distributed.timeout=3600' \
  '+fast_llm.model.base_model.decoder.block.mlp.recompute_level=full' \
  '+fast_llm.model.base_model.head.fp32_lm_head=true' \
  '+fast_llm.model.base_model.head.losses.grpo.policy_loss=gspo' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_low=3e-3' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_high=4e-3' \
  '+fast_llm.model.base_model.head.losses.grpo.normalize_by_documents=true' \
  '+fast_llm.model.base_model.head.losses.grpo.temperature=0.7' \
  '+fast_llm.model.base_model.head.losses.grpo.metrics=with_entropy' \
  '+fast_llm.optimizer.learning_rate.base=1e-6' \
  '+fast_llm.optimizer.learning_rate.warmup_iterations=50' \
  '+fast_llm.optimizer.learning_rate.decay_style=cosine' \
  '+fast_llm.optimizer.learning_rate.decay_iterations=400' \
  '+fast_llm.optimizer.gradient_norm_clipping=0.3'
