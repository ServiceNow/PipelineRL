#!/bin/bash
# TMax terminal-agent RL run, Qwen3.5-9B, 4 nodes x 8 H100.
#
# Prerequisites (already staged on /mnt/llmd/data, mounted into the job):
#   - proot binary:        /mnt/llmd/data/terminal_bin/proot
#   - base rootfs trees:   /mnt/llmd/data/terminal_bases/base_<domain>
#       (base_software_engineering acts as a universal fallback; build proper
#        per-domain trees with pipelinerl.domains.terminal.base_builder before
#        enabling rust/service domains or the full corpus).
# Override locations via TERMINAL_BASES_DIR / PROOT_BIN if you move them.

TIMESTAMP=$(date +%s)
OUTPUT_DIR_BASE=/mnt/llmd/results/exps/rafa/terminal
JOB_NAME=terminal_qwen35_9b_gspo_32
CONDA_ENV=${CONDA_ENV:-pipeline-rl}
CONDA_EXE=${CONDA_EXE:-/opt/conda/bin/conda}

COMMAND="python -m pipelinerl.launch \
output_dir=${OUTPUT_DIR_BASE}/${JOB_NAME} \
--config-name terminal \
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
