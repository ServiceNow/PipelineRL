#!/bin/bash
OUTPUT_DIR="results/workarena"
DATE_ID=$(date +%Y_%m_%d__%H_%M_%S)

python -m pipelinerl.launch \
    output_dir=${OUTPUT_DIR}/debug_${DATE_ID} \
    wandb.wandb_workspace_root=${OUTPUT_DIR} \
    actor.rollout_workers=2 \
    debug.mode=actor \
    world.actor_fraction=8 \
    world.finetune_fraction=0 \
    world.preprocessor_fraction=0 \
    --config-name workarena