#!/bin/bash
# Submit a multi-node EAI job for math task with Qwen2.5-7B-Instruct.
# Topology: 1 actor node (vLLM) + (NODES-1) fast-llm trainer nodes.
# Usage: bash submit_eai_math_7b_multinode.sh [NODES] [TARGET_TOTAL_MB] [TIMESTAMP]
# Example (fresh):  bash submit_eai_math_7b_multinode.sh 4 60
#   -> 1 actor node, 3 fast-llm nodes (BDP=12, depth_first=5, total_MBs=60)
# Example (resume): bash submit_eai_math_7b_multinode.sh 4 60 20260427_144646
#   -> resumes experiment math_7b_4node_mb5x12_20260427_144646 (checkpoint + wandb run preserved)
# Run `eai login` before executing this script.

IMAGE="registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1redis"
RESULTS_DIR="/mnt/shared/denis/math_7b_results"
MODEL_PATH="${MODEL_PATH:-/home/toolkit/Qwen2.5-7B-Instruct}"
NODES="${1:-2}"
TARGET_TOTAL_MB="${2:-1024}"   # target total microbatches/step across all DP ranks
TIMESTAMP="${3:-$(date +%Y%m%d_%H%M%S)}"  # experiment ID; omit to start fresh

FINETUNE_NODES=$((NODES - 1))
if [ "$FINETUNE_NODES" -lt 1 ]; then
  echo "ERROR: NODES must be >= 2 (got $NODES)" >&2
  exit 1
fi

# SDP=2 throughout; batch_data_parallel = finetune_gpus / SDP
FINETUNE_GPUS=$((FINETUNE_NODES * 8))
SDP=2
BDP=$((FINETUNE_GPUS / SDP))

# depth_first_micro_batches and prefetch_factor are per-rank (per DP group)
# Round up so total >= TARGET_TOTAL_MB
DEPTH_FIRST=$(( (TARGET_TOTAL_MB + BDP - 1) / BDP ))
PREFETCH=$DEPTH_FIRST

EXP_NAME="math_7b_${NODES}node_mb${DEPTH_FIRST}x${BDP}_${TIMESTAMP}"
EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"

if [ -n "${3:-}" ]; then
  RESUME_TS=$(date +%Y%m%d_%H%M%S)
  JOB_NAME="${EXP_NAME}_resume_${RESUME_TS}"
  echo "RESUMING: ${EXP_DIR} (job: ${JOB_NAME})"
else
  JOB_NAME="${EXP_NAME}"
fi

echo "Config: ${NODES} nodes, ${FINETUNE_NODES} fast-llm nodes, BDP=${BDP}, depth_first=${DEPTH_FIRST}, total_MBs=$((DEPTH_FIRST * BDP))"

CMD="
set -e
mkdir -p ${EXP_DIR}
source /home/toolkit/code/PipelineRL/.venv/bin/activate
PYTHONHASHSEED=42 python -m pipelinerl.launch \
  --config-path /home/toolkit/code/PipelineRL/conf \
  --config-name math \
  'streams=redis' \
  world.replicas=1 \
  world.actor_fraction=1 \
  world.preprocessor_fraction=0 \
  world.finetune_fraction=${FINETUNE_NODES} \
  model_path=${MODEL_PATH} \
  output_dir=${EXP_DIR} \
  wandb.wandb_workspace_root=${RESULTS_DIR} \
  wandb.wandb_entity_name=denisko-se \
  wandb.wandb_project_name=watermelon \
  wandb.wandb_group=eai_math7b_multinode \
  'vllm_config.vllm_kwargs.gpu-memory-utilization=0.85' \
  'vllm_config.vllm_kwargs.max-num-batched-tokens=8192' \
  'vllm_config.vllm_kwargs.max_model_len=16000' \
  'fast_llm.data.micro_batch_size=16000' \
  'llm.parameters.max_tokens=14000' \
  'test_llm.parameters.max_tokens=14000' \
  'eval_every_n_versions=0' \
  'fast_llm.training.num_workers=1' \
  '+fast_llm.training.prefetch_factor=${PREFETCH}' \
  'fast_llm.schedule.depth_first_micro_batches=${DEPTH_FIRST}' \
  'fast_llm.model.distributed.sequence_data_parallel=${SDP}' \
  '+fast_llm.model.distributed.timeout=3600' \
  '+fast_llm.model.base_model.decoder.block.mlp.recompute_level=full' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_low=0.02' \
  'fast_llm.model.base_model.head.losses.grpo.epsilon_high=0.02' \
  '+fast_llm.model.base_model.head.losses.grpo.compute_extra_metrics=true' \
  '+fast_llm.model.base_model.head.losses.grpo.compute_entropy_metric=true' \
  'fast_llm.training.checkpoint.interval=20' \
  '+fast_llm.optimizer.learning_rate.base=1e-5' \
  '+fast_llm.optimizer.learning_rate.warmup_iterations=10' \
  '+fast_llm.optimizer.learning_rate.decay_style=cosine' \
  '+fast_llm.optimizer.learning_rate.decay_iterations=100000' \
  '+fast_llm.optimizer.beta_2=0.95' \
  '+fast_llm.optimizer.gradient_norm_clipping=0.3'
"

# Generate a job spec YAML with all ports exposed in the Kubernetes Service.
# Ports: 29501 (EAI replica master), 11000 (Redis), 9000 (TCPStore weight-broadcast),
#        8080-8087 (vLLM HTTP servers, one per GPU on the actor node),
#        7777 (environment server, for actor→environment HTTP).
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
  --preemptable \
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
