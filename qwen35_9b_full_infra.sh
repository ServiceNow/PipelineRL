#!/bin/bash
# Full-infra run: --config-name terminal_full_infra (event_credit 0.5, session
# resurrection on, max_turns 40, memory eviction on, camouflage/replay/context-margin
# inherited; fi02 fleet DNS). Same 4x8 H100 recipe as the event-credit launchers.
# _02: clean verifier on (P1a), the single variable vs full_infra_01.

TIMESTAMP=$(date +%s)
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/rafa/terminal
JOB_NAME=${JOB_NAME:-terminal_qwen35_9b_full_infra_02}
CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}

COMMAND="python -m pipelinerl.launch \
output_dir=${OUTPUT_DIR_BASE}/${JOB_NAME} \
--config-name terminal_full_infra \
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
