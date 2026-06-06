# Privacy HopQA Training

This domain trains the MosaicLeaks agent used for the paper experiments. It consumes the materialized JSONL splits produced by MosaicProject and uses the Privacy HopQA helper service for private-document retrieval and BrowseComp search.

The default config is `conf/privacy_hopqa.yaml`. It expects:

- `privacy_hopqa.final_train_dataset_path` and `privacy_hopqa.final_val_dataset_path`: MosaicProject JSONL splits.
- `privacy_hopqa.task_data_root`: the DRBench-style task directory used by the helper service.
- `privacy_hopqa.helper_service_url`: the helper endpoint for local-document and BrowseComp retrieval.
- `model_path`, `output_dir`, and the usual PipelineRL launch settings.

Run it through the normal PipelineRL launcher:

```bash
python -m pipelinerl.launch --config-name privacy_hopqa \
  model_path=Qwen/Qwen3-4B-Instruct-2507 \
  output_dir=/path/to/output \
  privacy_hopqa.final_train_dataset_path=/path/to/train.jsonl \
  privacy_hopqa.final_val_dataset_path=/path/to/val.jsonl \
  privacy_hopqa.task_data_root=/path/to/drbench/data/tasks \
  privacy_hopqa.helper_service_url=http://localhost:8012
```

Supported training variants:

- Outcome reward: the default. It trains each captured planning trace against the final hop-accuracy reward.
- Situational hop-step reward: set `privacy_hopqa.training_reward_mode=hop_step_situational`, `finetune.rl.step_reward_advantages=true`, and `finetune.rl.filter_zero_advantage_groups=true`.
- Privacy-penalized situational reward: use the situational settings, then set `privacy_hopqa.privacy_reward_weight=1.0` and `privacy_hopqa.privacy_reward_service_url=<reward-helper-url>`.

The current defaults match the latest MosaicLeaks training runs: planning-only capture, hop-accuracy evaluation reward, rollout-level loss normalization, 4128 gradient accumulation passes, and 1 GB shared-memory queue entries for actor and preprocess workers.
