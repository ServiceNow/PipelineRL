#!/bin/bash
echo "Run LLM only"

# python -m pipelinerl.launch \
#     output_dir=results/llm_debug1 \
#     force_restart=true \
#     actor.llm_max_rollouts=16 \
#     finetune.seq_parallel=8 \
#     eval_every_n_versions=0 \
#     debug.mode=llm \
#     world.actor_fraction=8 \
#     world.finetune_fraction=0 \
#     world.preprocessor_fraction=0 \
#     --config-name mcp


python -m pipelinerl.entrypoints.run_vllm0 \
    --model Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8080 \
    --seed 42 \
    --actor-llm-idx 0 \
    --weight-update-group-init-method tcp://localhost:9000 \
    --weight-update-group-world-size 2 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --num-scheduler-steps 1 \
    --disable-log-requests \
    --disable-frontend-multiprocessing \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32000 \
    --enable-chunked-prefill \
    --return-tokens-as-token-ids \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --generation-config vllm \
    --max_model_len 32000 \
    --enable-auto-tool-choice \
    --tool-call-parser rl_tool \
    --tool-parser-plugin /home/toolkit/PipelineRL/pipelinerl/rl_tool_parser_plugin.py \
    --disable-weight-update


# python -m pipelinerl.entrypoints.run_vllm0 \
#     --model Qwen/Qwen2.5-7B \
#     --host 0.0.0.0 \
#     --port 8080 \
#     --seed 13 \
#     --actor-llm-idx 0 \
#     --weight-update-group-init-method tcp://localhost:9000 \
#     --weight-update-group-world-size 2 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 \
#     --num-scheduler-steps 1 \
#     --disable-log-requests \
#     --disable-frontend-multiprocessing \
#     --max-num-seqs 64 \
#     --max-num-batched-tokens 1024 \
#     --enable-chunked-prefill \
#     --return-tokens-as-token-ids \
#     --tensor-parallel-size 1 \
#     --pipeline-parallel-size 1 \
#     --generation-config vllm \
#     --max_model_len 64000 \
#     --disable-weight-update
 
# python -m pipelinerl.entrypoints.run_vllm0 --model /mnt/llmd/base_models/Mistral-Small-24B-Base-2501 --host 0.0.0.0 --port 8080 --seed 78 --actor-llm-idx 36 --weight-update-group-init-method tcp://dns-99833624-2133-43c0-a112-07520ffee505-0:9000 --weight-update-group-world-size 49 --dtype bfloat16 --gpu-memory-utilization 0.9 --num-scheduler-steps 1 --disable-log-requests --disable-frontend-multiprocessing --max-num-seqs 256 --max-num-batched-tokens 1024 --enable-chunked-prefill --return-tokens-as-token-ids --tensor-parallel-size 1 --pipeline-parallel-size 1 --generation-config vllm --max_model_len 32768

 