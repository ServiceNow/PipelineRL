#!/bin/bash
# Multi-node EAI DeepSpeed math run on vllm_v1 branch.
# Topology: 1 actor node (vLLM) + (NODES-1) DeepSpeed trainer nodes.
# Usage: bash submit_eai_math_7b_multinode_ds_vllm_v1.sh [NODES]
# Example: bash submit_eai_math_7b_multinode_ds_vllm_v1.sh 4
# Run `eai login` before executing this script.

IMAGE="registry.toolkit-sp.yul201.service-now.com/snow.research.afm/interactive-toolkit:25.12-py3-vllm014rc1redis"
RESULTS_DIR="/mnt/shared/denis/math_7b_results"
MODEL_PATH="${MODEL_PATH:-/home/toolkit/Qwen2.5-7B}"
NODES="${1:-4}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="math_7b_ds_fastllm_${NODES}node_${TIMESTAMP}"
EXP_DIR="${RESULTS_DIR}/${EXP_NAME}"
JOB_NAME="${EXP_NAME}"

echo "Config: ${NODES} nodes, actor_fraction=4, finetune_fraction=4, max_train_steps=400"

CMD="
set -e
mkdir -p ${EXP_DIR}
cd /home/toolkit/code/PipelineRL
source /home/toolkit/code/PipelineRL/.venv/bin/activate
PYTHONHASHSEED=42 python -m pipelinerl.launch \
  --config-path /home/toolkit/code/PipelineRL/conf \
  --config-name math \
  output_dir=${EXP_DIR} \
  wandb.wandb_workspace_root=${RESULTS_DIR} \
  wandb.wandb_entity_name=denisko-se \
  wandb.wandb_project_name=watermelon \
  wandb.wandb_group=eai_math7b_ds_fastllm \
  '+wandb.wandb_run_name=math7b_ds_fastllm_${NODES}node_${TIMESTAMP}' \
  use_fast_llm=false \
  actor.llm_max_rollouts=128 \
  force_restart=true \
  fp32_lm_head=true \
  finetune.learning_rate=1e-6 \
  finetune.attempts=8 \
  finetune.rl.policy_loss=gspo \
  finetune.rl.epsilon_low=3e-3 \
  finetune.rl.epsilon_high=4e-3 \
  '+finetune.rl.filter_zero_advantage_groups=true' \
  finetune.max_train_steps=400 \
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
  world.run_id=\${MASTER_ADDR} \
  streams=files \
  eval_every_n_versions=0 \
  model_path=${MODEL_PATH}
"

SPEC_YAML=$(mktemp /tmp/eai_job_spec_XXXXXX.yaml)
cat > "$SPEC_YAML" << 'YAML_EOF'
options:
    internal-dns:
        name: ""
        ports:
            - port: 29501
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
  --env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
  -- /bin/bash -c "$CMD"

rm -f "$SPEC_YAML"
