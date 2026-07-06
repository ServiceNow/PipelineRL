#!/bin/bash
# Phase 1 event-credit run (A_t = z - c|z|(e_t - mean_e)), parallel to the gspo_N mainline.
# Same 4 nodes x 8 H100 recipe as qwen35_9b_terminal.sh but with
# --config-name terminal_event_credit (event_credit_coef 0.5 + ec01-suffixed
# fleet DNS so it never clashes with the mainline run's fleets).

TIMESTAMP=$(date +%s)
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/rafa/terminal
JOB_NAME=${JOB_NAME:-terminal_qwen35_9b_event_credit_01}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}

COMMAND="python -m pipelinerl.launch \
output_dir=${OUTPUT_DIR_BASE}/${JOB_NAME} \
--config-name terminal_event_credit \
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
