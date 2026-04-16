#!/bin/bash
# Submit an 8-GPU eai job for math task with Qwen2.5-7B-Instruct:
# 2 vLLM actors (1 GPU each, TP=1) + 6-GPU fast-llm trainer (DP=3, ZeRO-2, SDP=2)
# 16K/14K sequences, depth_first_micro_batches=1024, full recompute, prefetch=1024
# Run `eai login` before executing this script.

IMAGE="registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1redis"
RESULTS_DIR="/mnt/shared/denis/math_7b_results"
MODEL_PATH="${MODEL_PATH:-/home/toolkit/Qwen2.5-7B-Instruct}"
MICROBATCHES="${1:-32}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_DIR="${RESULTS_DIR}/math_7b_8gpu_mb${MICROBATCHES}_${TIMESTAMP}"
JOB_NAME="math_7b_8gpu_mb${MICROBATCHES}_${TIMESTAMP}"

CMD="
set -e
mkdir -p ${EXP_DIR}
source /home/toolkit/code/PipelineRL/.venv/bin/activate
PYTHONHASHSEED=42 python -m pipelinerl.launch \
  --config-path /home/toolkit/code/PipelineRL/conf \
  --config-name math \
  'streams=redis' \
  world.replicas=1 \
  world.actor_fraction=2 \
  world.preprocessor_fraction=0 \
  world.finetune_fraction=6 \
  model_path=${MODEL_PATH} \
  output_dir=${EXP_DIR} \
  wandb.wandb_workspace_root=${RESULTS_DIR} \
  wandb.wandb_entity_name=denisko-se \
  wandb.wandb_project_name=watermelon \
  wandb.wandb_group=eai_math7b_16k_sdp2_fastllm_integration \
  'vllm_config.vllm_kwargs.gpu-memory-utilization=0.85' \
  'vllm_config.vllm_kwargs.max-num-batched-tokens=8192' \
  'vllm_config.vllm_kwargs.max_model_len=16000' \
  'fast_llm.data.micro_batch_size=16000' \
  'llm.parameters.max_tokens=14000' \
  'test_llm.parameters.max_tokens=14000' \
  'eval_every_n_versions=0' \
  'fast_llm.training.num_workers=1' \
  '+fast_llm.training.prefetch_factor=${MICROBATCHES}' \
  'fast_llm.schedule.depth_first_micro_batches=${MICROBATCHES}' \
  'fast_llm.model.distributed.sequence_data_parallel=2' \
  '+fast_llm.model.distributed.timeout=3600' \
  '+fast_llm.model.base_model.decoder.block.mlp.recompute_level=full' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_low=0.02' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_high=0.02' \
  '+fast_llm.optimizer.learning_rate.base=1e-5' \
  '+fast_llm.optimizer.learning_rate.warmup_iterations=10' \
  '+fast_llm.optimizer.learning_rate.decay_style=cosine' \
  '+fast_llm.optimizer.learning_rate.decay_iterations=100000' \
  '+fast_llm.optimizer.beta_2=0.95' \
  '+fast_llm.optimizer.gradient_norm_clipping=0.3' \
  '+wandb.wandb_run_name=math7b_16k_sdp2_mb${MICROBATCHES}_lr1e5'
"

eai job new \
  --preemptable \
  --gpu 8 \
  --cpu 128 \
  --mem 800 \
  --name "$JOB_NAME" \
  -i "$IMAGE" \
  --data "snow.home.denis_kocetkov:/home/toolkit:rw" \
  --data "snow.research.afm.shared_fml:/mnt/shared:rw" \
  --env "HOME=/home/toolkit" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- /bin/bash -c "$CMD"
