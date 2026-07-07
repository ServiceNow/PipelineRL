#!/bin/bash
# Shuffled-events control arm for the event-credit experiment: identical to
# qwen35_9b_event_credit.sh except --config-name terminal_event_credit_shuffle
# (event_error flags permuted within each rollout before shaping) and
# ecs01-suffixed fleet DNS. Requires the credit-instruments merge on terminal.

TIMESTAMP=$(date +%s)
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/rafa/terminal
JOB_NAME=${JOB_NAME:-terminal_qwen35_9b_event_credit_shuffle_01}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}

COMMAND="python -m pipelinerl.launch \
output_dir=${OUTPUT_DIR_BASE}/${JOB_NAME} \
--config-name terminal_event_credit_shuffle \
--config-dir /home/toolkit/PipelineRL/conf"

make multi-replica-job \
    REPLICAS=4 \
    ENV=${CONDA_ENV} \
    CONDA_EXE=${CONDA_EXE} \
    SNAPSHOT=0 \
    NPROC=8 \
    BID=999 \
    JOB_NAME=${JOB_NAME}_${TIMESTAMP} \
    HOME_DATA_NAME="snow.research.tapes.rafael_pardinas_home" \
    TRANSFORMERS_CACHE_DATA="snow.research.tapes.transformers_cache" \
    DATA_OBJ="snow.research.tapes.data" \
    RESULTS_OBJ="snow.research.tapes.results" \
    BASE_MODELS_OBJ="snow.research.tapes.base_models" \
    COMMAND="${COMMAND}"
