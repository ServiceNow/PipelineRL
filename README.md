Pipeline RL
=================

[![Github](https://img.shields.io/badge/Blog%20Post-000000)](https://huggingface.co/blog/ServiceNow/pipelinerl/)

A scalable asynchronous reinforcement learning implementation with in-flight weight updates. Designed to maximize GPU utilization while staying as on-policy as possible.

<p align="center">
    <img src="assets/figure1.jpg" alt="Pipeline-RL Architecture" width="600">
</p>

# Setup

Clone the repository and change the directory to `pipelinerl`
```bash
git clone git@github.com:ServiceNow/PipelineRL.git
cd pipelinerl
```

Create the environments with dependencies.
```bash
conda create -n pipeline-rl -y python=3.11
conda run --no-capture-output -n pipeline-rl pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 
conda run --no-capture-output -n pipeline-rl pip install -r requirements.txt --no-build-isolation
```

By default Pipeline-RL will use the file system as the medium for streaming the generated data to the trainer processes. This works on one node, but the files can get quite large. To use Redis instead you will need to install the Redis server in the same conda environment:
```bash
conda install redis-server==7.4.0 -c conda-forge 
```

# Run experiments

First, activate the conda environment:
```bash
conda activate pipeline-rl
```

Single node with 8 H00 GPUs:

```bash
python -m pipelinerl.launch output_dir=results/base1
```

If you only have 4 H100 GPUs:
```bash
python -m pipelinerl.launch --config-name base_4gpu output_dir=results/base1 
```

To use Redis instead of the filesystem for data streaming:
```
python -m pipelinerl.launch streams=redis output_dir=results/base1
```

Multi node: coming soon.

## PipelineRL configuration explained

Some key hyperparameters:
- `attempts`: the number of attempts per prompt / reasoning problem
- `finetune.seq_length`: maximum number of tokens per micro-batch, if you are using `finetune.seq_packing=true` which is the default
- `finetune.train_batch_size`: 
    - if `finetune.seq_packing=false`, the number of samples in each micro-batch.
    - if `finetune.seq_packing=true`, see the explanation of `finetune.gradient_accumulation_passes` below
- `finetune.gradient_accumulation_passes`
    -  if `finetune.seq_packing=false`: the total number of micro-batches per batch for all training workers
    -  if `finetune.seq_packing=true`: take this number and multiply it by `finetune.train_batch_size` to get the total batch size per optimizer step for all training workers 
