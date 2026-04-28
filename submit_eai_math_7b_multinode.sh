#!/bin/bash
# Multi-node fast-llm finetuner math run with DS-matched params (GSPO, docs_per_step).
# Topology: actor_fraction=4 (16 GPUs / 2 nodes) + finetune_fraction=4 (16 GPUs / 2 nodes).
# Usage: bash submit_eai_math_7b_multinode.sh [NODES] [TIMESTAMP]
# Example (fresh):  bash submit_eai_math_7b_multinode.sh 4
# Example (resume): bash submit_eai_math_7b_multinode.sh 4 20260428_132330
# Run `eai login` before executing this script.

IMAGE="registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1redis"
RESULTS_DIR="/mnt/shared/denis/math_7b_results"
MODEL_PATH="${MODEL_PATH:-/home/toolkit/Qwen2.5-7B}"
NODES="${1:-4}"
TIMESTAMP="${2:-$(date +%Y%m%d_%H%M%S)}"

EXP_NAME="math_7b_${NODES}node_fastllm_gspo_${TIMESTAMP}"
EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"

if [ -n "${2:-}" ]; then
  RESUME_TS=$(date +%Y%m%d_%H%M%S)
  JOB_NAME="${EXP_NAME}_resume_${RESUME_TS}"
  echo "RESUMING: ${EXP_DIR} (job: ${JOB_NAME})"
else
  JOB_NAME="${EXP_NAME}"
fi

echo "Config: ${NODES} nodes, actor_fraction=4, finetune_fraction=4, docs_per_step=1024, max_train_steps=400"

CMD="
set -e
mkdir -p ${EXP_DIR}
cd /home/toolkit/code/PipelineRL
source /home/toolkit/code/PipelineRL/.venv/bin/activate
PYTHONHASHSEED=42 python -m pipelinerl.launch \
  --config-path /home/toolkit/code/PipelineRL/conf \
  --config-name math \
  streams=redis \
  world.actor_fraction=4 \
  world.preprocessor_fraction=0 \
  world.finetune_fraction=4 \
  world.run_id=\${MASTER_ADDR} \
  model_path=${MODEL_PATH} \
  output_dir=${EXP_DIR} \
  force_restart=true \
  fp32_lm_head=true \
  actor.llm_max_rollouts=128 \
  finetune.attempts=8 \
  finetune.max_train_steps=400 \
  '+finetune.rl.filter_zero_advantage_groups=true' \
  eval_every_n_versions=0 \
  wandb.wandb_workspace_root=${RESULTS_DIR} \
  wandb.wandb_entity_name=denisko-se \
  wandb.wandb_project_name=watermelon \
  wandb.wandb_group=eai_math7b_fastllm_gspo \
  '+wandb.wandb_run_name=math7b_fastllm_gspo_${NODES}node_${TIMESTAMP}' \
  'vllm_config.vllm_kwargs.gpu-memory-utilization=0.85' \
  'vllm_config.vllm_kwargs.max-num-batched-tokens=8192' \
  'vllm_config.vllm_kwargs.max_model_len=20000' \
  'llm.parameters.max_tokens=16000' \
  'llm.parameters.temperature=0.7' \
  'test_llm.parameters.max_tokens=16000' \
  'test_llm.parameters.temperature=0.7' \
  'fast_llm.data.micro_batch_size=20000' \
  '+fast_llm.schedule.docs_per_step=1024' \
  'fast_llm.training.train_iters=400' \
  'fast_llm.training.num_workers=1' \
  'fast_llm.training.checkpoint.interval=20' \
  'fast_llm.model.distributed.sequence_data_parallel=2' \
  '+fast_llm.model.distributed.timeout=3600' \
  '+fast_llm.model.base_model.decoder.block.mlp.recompute_level=full' \
  '+fast_llm.model.base_model.head.losses.grpo.policy_loss=gspo' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_low=3e-3' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_high=4e-3' \
  '+fast_llm.model.base_model.head.losses.grpo.compute_extra_metrics=true' \
  '+fast_llm.optimizer.learning_rate.base=1e-6' \
  '+fast_llm.optimizer.learning_rate.warmup_iterations=10' \
  '+fast_llm.optimizer.learning_rate.decay_style=cosine' \
  '+fast_llm.optimizer.learning_rate.decay_iterations=100000' \
  '+fast_llm.optimizer.beta_2=0.95' \
  '+fast_llm.optimizer.gradient_norm_clipping=0.3'
"

SPEC_YAML=$(mktemp /tmp/eai_job_spec_XXXXXX.yaml)
cat > "$SPEC_YAML" << 'YAML_EOF'
options:
    internal-dns:
        name: ""
        ports:
            - port: 29501
            - port: 11000
            - port: 9000
            - port: 7777
            - port: 8080
            - port: 8081
            - port: 8082
            - port: 8083
            - port: 8084
            - port: 8085
            - port: 8086
            - port: 8087
YAML_EOF

eai job new \
  --file "$SPEC_YAML" \
  --non-preemptable \
  --replicas "$NODES" \
  --gpu 8 \
  --cpu 128 \
  --mem 800 \
  --name "$JOB_NAME" \
  -i "$IMAGE" \
  --data "snow.home.denis_kocetkov:/home/toolkit:rw" \
  --data "snow.research.afm.shared_fml:/mnt/shared:rw" \
  --env "HOME=/home/toolkit" \
  --env "GPUS_PER_NODE=8" \
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  --env "TRITON_CACHE_DIR=/tmp/triton_cache" \
  -- /bin/bash -c "$CMD"

rm -f "$SPEC_YAML"
