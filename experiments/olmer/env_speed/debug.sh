#!/bin/bash
echo "Run 38 workers"
DEBUG_FILE=timing_debug_workers38_4.jsonl python -m pipelinerl.launch \
    output_dir=results/actor_debug38_4 \
    force_restart=true \
    actor.llm_max_rollouts=256 \
    finetune.seq_parallel=8 \
    eval_every_n_versions=0 \
    actor.rollout_workers=38 \
    debug.mode=actor \
    world.actor_fraction=8 \
    world.finetune_fraction=0 \
    world.preprocessor_fraction=0 \
    --config-name mcp

# echo "Run 10 workers"
# DEBUG_FILE=timing_debug_gpt_workers10.jsonl python -m pipelinerl.launch \
#     output_dir=results/actor_debug2 \
#     force_restart=true \
#     actor.llm_max_rollouts=16 \
#     finetune.seq_parallel=8 \
#     eval_every_n_versions=0 \
#     actor.rollout_workers=10 \
#     debug.mode=actor \
#     world.actor_fraction=8 \
#     world.finetune_fraction=0 \
#     world.preprocessor_fraction=0 \
#     --config-name mcp


# echo "Run 5 workers"
# DEBUG_FILE=timing_debug_gpt_workers5.jsonl python -m pipelinerl.launch \
#     output_dir=results/actor_debug2 \
#     force_restart=true \
#     actor.llm_max_rollouts=16 \
#     finetune.seq_parallel=8 \
#     eval_every_n_versions=0 \
#     actor.rollout_workers=5 \
#     debug.mode=actor \
#     world.actor_fraction=8 \
#     world.finetune_fraction=0 \
#     world.preprocessor_fraction=0 \
#     --config-name mcp

# echo "Run 40 workers"
# DEBUG_FILE=timing_debug_gpt_workers40.jsonl python -m pipelinerl.launch \
#     output_dir=results/actor_debug2 \
#     force_restart=true \
#     actor.llm_max_rollouts=16 \
#     finetune.seq_parallel=8 \
#     eval_every_n_versions=0 \
#     actor.rollout_workers=40 \
#     debug.mode=actor \
#     world.actor_fraction=8 \
#     world.finetune_fraction=0 \
#     world.preprocessor_fraction=0 \
#     --config-name mcp

# echo "Run 30 workers"
# DEBUG_FILE=timing_debug_gpt_workers30.jsonl python -m pipelinerl.launch \
#     output_dir=results/actor_debug2 \
#     force_restart=true \
#     actor.llm_max_rollouts=16 \
#     finetune.seq_parallel=8 \
#     eval_every_n_versions=0 \
#     actor.rollout_workers=30 \
#     debug.mode=actor \
#     world.actor_fraction=8 \
#     world.finetune_fraction=0 \
#     world.preprocessor_fraction=0 \
#     --config-name mcp