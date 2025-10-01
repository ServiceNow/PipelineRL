#!/bin/bash
python -m pipelinerl.launch \
    output_dir=results/actor_debug1 \
    force_restart=true \
    world.env_replicas_per_actor=1 \
    actor.llm_max_rollouts=16 \
    finetune.seq_parallel=8 \
    eval_every_n_versions=0 \
    actor.rollout_workers=1 \
    debug.mode=actor \
    world.actor_fraction=8 \
    world.finetune_fraction=0 \
    world.preprocessor_fraction=0 \
    --config-name mcp

    # environment.n_envs=4 \
    # environment.mcp_read_timeout_seconds=300 \
    # environment.env_call_timeout=300 \