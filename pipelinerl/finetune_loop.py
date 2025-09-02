import contextlib
import json
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
import numpy as np
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Literal

import deepspeed
import requests
import torch
import torch.distributed as dist
from accelerate.utils import FullyShardedDataParallelPlugin
from omegaconf import DictConfig
from pydantic import BaseModel
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import MixedPrecision
from transformers import PreTrainedTokenizerFast, get_scheduler, set_seed
from ring_flash_attn import substitute_hf_flash_attn, update_ring_flash_attn_params

from pipelinerl.finetune.value_model import AutoModelForCausalLMWithValueHead
import pipelinerl.torch_utils
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.checkpoints import (
    load_model,
    load_tokenizer,
    load_training_state,
    remove_results,
    save_model_and_tokenizer,
    save_training_state,
)
from pipelinerl.finetune.context import get_accelerator
from pipelinerl.finetune.data import collate, collate_packed
from pipelinerl.finetune.logging_ import log_metrics, log_time, setup_logging
from pipelinerl.finetune.optim import get_optimizer
from pipelinerl.finetune.rl import (
    RLConfig,
    rl_step,
)
from pipelinerl.finetune.rl.utils import aggregate_rl_stats
from pipelinerl.finetune.types import TrainingMetrics
from pipelinerl.finetune.utils import create_sentinel_batch
from pipelinerl.streams import (
    SingleStreamSpec,
    read_stream,
    set_streams_backend,
    write_to_streams,
)
from pipelinerl.utils import wait_for_inference_servers

logger = logging.getLogger(__name__)


def gather_rl_metrics(rl_metrics: Dict[str, List]) -> Dict[str, List]:
    """
    Gather RL metrics from all processes using torch.distributed.all_gather_object.

    Args:
        rl_metrics: Dictionary mapping metric names to lists of values

    Returns:
        Dictionary with gathered metrics from all processes
    """
    # First, gather all metrics dictionaries from all processes
    all_metrics = [None] * dist.get_world_size()
    dist.all_gather_object(all_metrics, rl_metrics)
    
    # Now aggregate the gathered metrics
    aggregated_metrics = defaultdict(list)
    
    # Process each gathered dictionary
    for process_metrics in all_metrics:
        #if process_metrics is None:
        #    continue
        for key, values in process_metrics.items():
            if values:
                aggregated_metrics[key].extend([v for v in values if np.isfinite(v)])
    
    return aggregated_metrics




def run_data_loader(
    data_stream: SingleStreamSpec,
    batch_queue: Queue[PipelineBatchEncoding | Exception],
):
    """Load BatchEncoding objects directly from preprocessor."""
    with read_stream(data_stream) as stream_reader:
        logger.info(f"Starting data loader for stream {data_stream.topic} on partition {data_stream.partition}")
        while True:
            try:
                for batch_encoding in stream_reader.read():
                    if isinstance(batch_encoding, Exception):
                        batch_queue.put(batch_encoding)
                        logger.info(f"Received exception from stream: {batch_encoding}")
                        break

                    # Convert to PipelineBatchEncoding and move to device
                    pipeline_batch = PipelineBatchEncoding(**batch_encoding)
                    pipeline_batch = pipeline_batch.to_device(get_accelerator().device)
                    batch_queue.put(pipeline_batch)

            except Exception as e:
                logger.error(f"Error in stream reader: {e}")
                batch_queue.put(e)
                break


#TODO: should the topic be renamed to trainer_messages since it contains more than just weight updates?
TRAINER_TOPIC = "weight_update_request"


class ParameterInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str


class WeightUpdateRequest(BaseModel):
    kind: Literal["weight_update_request"] = "weight_update_request"
    version: int
    parameters_info: list[ParameterInfo]
    timestamp: float = time.time()


class WeightUpdateSuccess(BaseModel):
    kind: Literal["weight_update_success"] = "weight_update_success"
    version: int
    timestamp: float = time.time()

 
class SamplesProcessed(BaseModel):
    kind: Literal["samples_processed"] = "samples_processed"
    samples_processed: int
    timestamp: float = time.time()

TrainerMessage = WeightUpdateRequest | WeightUpdateSuccess | SamplesProcessed


class WeightUpdateManager:
    def __init__(self, llm_urls: list[str], accelerated_model, update_stream, actor_update_group):
        self.llm_urls = llm_urls
        self.accelerated_model = accelerated_model
        self.update_stream = update_stream
        self.actor_update_group = actor_update_group
        self.thread_pool = ThreadPoolExecutor(max_workers=len(llm_urls))

    def _request_weight_update(self, url: str, message: WeightUpdateRequest):
        response = None
        try:
            response = requests.post(url + "/receive_weight_update", json=message.model_dump())
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error sending weight update request to {url}: {e}")
            # print response details
            if response is not None:
                logger.error(f"Response: {response.status_code} - {response.text}")

    def request_weight_updates(self, message: WeightUpdateRequest):
        futures = []
        for url in self.llm_urls:
            futures.append(self.thread_pool.submit(self._request_weight_update, url, message))
        return futures

    def send_weight_update(
        self,
        version: int,
    ):
        if (
            isinstance(self.accelerated_model, deepspeed.DeepSpeedEngine)
            and self.accelerated_model.zero_optimization_stage() == 3
        ):
            module = self.accelerated_model.module
            logger.info("Start gathering and sending ZeRO Stage 3 weights")

            # Filter out value head parameters and get only the pretrained model parameters
            named_parameters = (
                dict(module.pretrained_model.named_parameters())
                if isinstance(module, AutoModelForCausalLMWithValueHead)
                else dict(module.named_parameters())
            )
            
            if get_accelerator().is_main_process:
                parameters_info = [
                    # assume DeepSpeed Stage 3
                    ParameterInfo(name=name, shape=list(parameter.ds_shape), dtype=str(torch.bfloat16))
                    for name, parameter in named_parameters.items()
                ]
                message = WeightUpdateRequest(version=version, parameters_info=parameters_info)
                futures = self.request_weight_updates(message)
                logger.info(f"Published weight update request for version {version}")

            for name, parameter in named_parameters.items():
                with deepspeed.zero.GatheredParameters([parameter]):
                    if get_accelerator().is_main_process:
                        dist.broadcast(parameter.data.bfloat16(), src=0, group=self.actor_update_group)
            if get_accelerator().is_main_process:
                logger.info("Wait for HTTP requests")
                for future in futures:  # type: ignore
                    future.result()
            logger.info("Finished broadcasting weights")

            if get_accelerator().is_main_process:
                assert self.update_stream is not None
                with write_to_streams(self.update_stream) as writer:
                    writer.write(WeightUpdateSuccess(version=version))
        else:
            if isinstance(self.accelerated_model, AutoModelForCausalLMWithValueHead):
                raise ValueError(
                    "AutoModelForCausalLMWithValueHead is not supported with non-Deepspeed training. "
                    "Please use Deepspeed Stage 3 for weight updates."
                )
            logger.info("Gather all weights at rank 0")
            if isinstance(self.accelerated_model, FSDP):
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=False, rank0_only=True)
                with FSDP.state_dict_type(
                    self.accelerated_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
                ):
                    named_parameters = self.accelerated_model.state_dict()
                if "lm_head.weight" in named_parameters:
                    logger.info("Removing lm_head.weight from gathered parameters, because it's not a parameter.")
                    logger.info("See https://github.com/huggingface/transformers/issues/9753")
                    del named_parameters["lm_head.weight"]
            else:
                unwrapped = get_accelerator().unwrap_model(self.accelerated_model)
                named_parameters = dict(unwrapped.named_parameters())
            if get_accelerator().is_main_process:
                assert self.update_stream is not None
                parameters_info = [
                    # assume DeepSpeed Stage 3
                    ParameterInfo(name=name, shape=list(parameter.shape), dtype=str(torch.bfloat16))
                    for name, parameter in named_parameters.items()
                ]
                messages = WeightUpdateRequest(version=version, parameters_info=parameters_info)
                futures = self.request_weight_updates(messages)
                logger.info(f"Published weight update request for version {version}")
                for _, parameter in named_parameters.items():
                    dist.broadcast(parameter.data.bfloat16(), src=0, group=self.actor_update_group)
                dist.barrier(self.actor_update_group)
                for future in futures:
                    future.result()
                logger.info("Finished broadcasting weights")
                with write_to_streams(self.update_stream) as writer:
                    writer.write(WeightUpdateSuccess(version=version))
            else:
                pass
            get_accelerator().wait_for_everyone()


def get_batch_token_count(batch):
    """Count actual tokens in batch (excluding padding)"""
    attention_mask = batch.attention_mask
    assert attention_mask is not None, "We need attention_mask for accurate token counting"
    return attention_mask.sum().item()


def get_batch_sequence_count(batch: PipelineBatchEncoding):
    """Count actual sequences in a batch, accounting for sequence packing"""
    if batch.position_ids is not None:
        assert batch.seq_boundaries is not None 
        if batch.padding == 0:
            return len(batch.seq_boundaries) - 1
        else:
            return len(batch.seq_boundaries) - 2
    else:
        # For unpacked sequences, each row is one sample
        return batch.input_ids.size(0)


def validate_packing_config(args):
    if not args.seq_packing:
        return
    if not args.use_flash_attention:
        raise ValueError(
            "Sequence packing requires flash attention. Either:\n"
            "- Enable flash attention (use_flash_attention=true), or\n"
            "- Disable sequence packing (seq_packing=false)"
        )


def run_finetuning_loop(
    cfg: DictConfig,
):
    set_streams_backend(**cfg.streams)

    dt = time.perf_counter()
    time_stats = {}

    exp_root_dir = Path(cfg.output_dir)
    output_dir = Path(cfg.finetune.output_dir)
    num_processes = get_accelerator().state.num_processes  # type: ignore
    args = cfg.finetune if "finetune" in cfg else cfg
    validate_packing_config(args)

    if not args.gradient_accumulation_passes % num_processes == 0:
        raise ValueError("gradient_accumulation_passes must be divisible by num_processes")
    gradient_accumulation_passes_per_gpu = args.seq_parallel * (args.gradient_accumulation_passes // num_processes)
    if (ds_plugin := get_accelerator().state.deepspeed_plugin) is not None:
        logger.info("Manual inform Deepspeed about micro batch size and gradient accumulation")
        ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.train_batch_size
        ds_plugin.deepspeed_config["gradient_accumulation_steps"] = gradient_accumulation_passes_per_gpu
        if args.gradient_clipping_threshold:
            ds_plugin.deepspeed_config["gradient_clipping"] = args.gradient_clipping_threshold

    set_seed(args.seed)

    # using a subfolder makes "safe" overwriting possible
    current_dir = output_dir / "current"
    intermediate_root_dir = output_dir / "intermediate"
    training_state_dir = output_dir / "training_state"
    log_dir = output_dir / "logs"

    if args.force_restart and get_accelerator().is_main_process:
        remove_results(current_dir, intermediate_root_dir, training_state_dir, log_dir)

    # Rendez-vous with inference servers
    #TODO: rename the stream to be more general
    weight_update_stream = SingleStreamSpec(exp_path=exp_root_dir, topic="weight_update_request")

    # Logging
    if get_accelerator().is_main_process:
        setup_logging(cfg, output_dir)
    else:
        logger.info(f"Last logging message from {get_accelerator().process_index}, will be quiet from now on")
        logging.disable(logging.INFO)

    logger.info(get_accelerator().state)
    logger.info(f"Saving experiment to {output_dir}")
    dt = log_time(dt, time_stats, "finetune/startup")

    tokenizer = load_tokenizer(args.config_name)
    logger.info("About to load model")
    model = load_model(args, args.model_class, current_dir)
    logger.info(f"Model loaded in dtype {model.dtype}")

    dt = log_time(dt, time_stats, "finetune/model_load")

    data_stream = SingleStreamSpec(
        exp_path=exp_root_dir,
        topic=args.input,
        instance=0,
        partition=get_accelerator().process_index,
    )

    logger.info(f"Using {'packed' if args.seq_packing else 'unpacked'} collate function")

    optimizer = get_optimizer(args.optim, model, args.learning_rate, args.weight_decay)
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer,
        args.num_warmup_steps,
        args.max_train_steps,
    )
    dtypes = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            dtypes.add(param.dtype)
    logger.info(f"Before accelerator.prepare() the optimizer's parameters had dtypes {dtypes}")

    if isinstance(plugin := getattr(get_accelerator().state, "fsdp_plugin", None), FullyShardedDataParallelPlugin):
        assert isinstance(plugin, FullyShardedDataParallelPlugin)
        value_to_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}
        if plugin.mixed_precision_policy is not None:
            logger.info("Will override FSDP mixed precision policy with settings from now-reasoner config")
            plugin.mixed_precision_policy = MixedPrecision(
                param_dtype=value_to_dtype[cfg.fsdp.param_dtype],
                reduce_dtype=value_to_dtype[cfg.fsdp.reduce_dtype],
                buffer_dtype=value_to_dtype[cfg.fsdp.buffer_dtype],
            )
        else:
            logger.info("No mixed precision for FSDP")
    (
        model,
        optimizer,
        lr_scheduler,
    ) = get_accelerator().prepare(model, optimizer, lr_scheduler)
    logger.info("Model, optimizer and lr_scheduler prepared")
    logger.info(
        f"Model class is {model.__class__}, optimizer class is {optimizer.__class__}, lr_scheduler class is {lr_scheduler.__class__}"
    )
    if get_accelerator().is_main_process and isinstance(model, FSDP):
        logger.info(f"FSDP internal mixed precision config: {model.mixed_precision}")
    after_dtype = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            after_dtype.add(param.dtype)
    logger.info(f"After accelerator.prepare() the optimizer's parameters had dtypes {after_dtype}")

    get_accelerator().wait_for_everyone()

    if get_accelerator().is_main_process and args.send_weight_updates:
        logger.info("Initializing actor process group")
        actor_update_group = pipelinerl.torch_utils.init_extra_process_group(
            group_name="actor",
            backend="nccl",
            init_method=cfg.me.weight_update_group_init_method,
            rank=0,
            world_size=cfg.me.weight_update_group_world_size,
        )
        logger.info("Actor process group initialized")
    else:
        actor_update_group = None
        #weight_update_stream = None
    get_accelerator().wait_for_everyone()

    training_metrics = TrainingMetrics()
    if os.path.exists(training_state_dir):
        # WARNING: In case of deepspeed this will overwrite model weights too
        training_metrics = load_training_state(training_state_dir, model, optimizer, lr_scheduler, training_metrics)
        training_metrics.lr = optimizer.param_groups[0]["lr"]
        logger.info("LR after loading training state: %.2E" % training_metrics.lr)
        dt = log_time(dt, time_stats, "finetune/training_state_load")

    if get_accelerator().is_main_process:
        trigger_message = SamplesProcessed(samples_processed=training_metrics.samples)
        with write_to_streams(weight_update_stream) as writer:
            writer.write(trigger_message)
        

    if args.send_weight_updates:
        llm_urls = cfg.me.llm_urls.split("+")
        if get_accelerator().is_main_process:
            wait_for_inference_servers(llm_urls)
        get_accelerator().wait_for_everyone()
        weight_update_manager = WeightUpdateManager(
            llm_urls=llm_urls,
            accelerated_model=model,
            update_stream=weight_update_stream,
            actor_update_group=actor_update_group,
        )
        logger.info("Load the first version of the model into inference LLMs")
        weight_update_manager.send_weight_update(training_metrics.samples)
    else:
        weight_update_manager = None

    batch_queue = Queue(maxsize=1)
    data_loader_worker_fn = partial(
        run_data_loader,
        data_stream=data_stream,
        batch_queue=batch_queue,
    )
    data_loader_thread = threading.Thread(target=data_loader_worker_fn, args=())

    get_accelerator().wait_for_everyone()
    model.train()
    data_loader_thread.start()

    seq_parallel_group = None
    if cfg.finetune.seq_parallel > 1:
        assert get_accelerator().state.num_processes % cfg.finetune.seq_parallel == 0
        for leader_rank in range(0, get_accelerator().state.num_processes, cfg.finetune.seq_parallel):
            group_ranks = [leader_rank + i for i in range(cfg.finetune.seq_parallel)]
            logger.info(f"Creating sequence parallel group with ranks: {group_ranks}")
            group = dist.new_group(group_ranks, backend="nccl")
            if get_accelerator().process_index in group_ranks:
                seq_parallel_group = group
        assert seq_parallel_group is not None
        substitute_hf_flash_attn(seq_parallel_group, heads_k_stride=1)

    try:
        logger.info("Start training")
        rl_finetuning_worker(
            args,
            model,
            optimizer,
            lr_scheduler,
            weight_update_manager,
            tokenizer,
            training_metrics,
            batch_queue,
            weight_update_stream,
            seq_parallel_group, 
        )
    finally:
        if actor_update_group:
            dist.destroy_process_group(actor_update_group)


def rl_finetuning_worker(
    args: DictConfig,
    # model, optimizer and scheduler can be of different types depending on what Accelerate backend we use (DeepSpeed vs FSDP)
    model: Any,
    optimizer: Any,
    lr_scheduler: Any,
    weight_update_manager: WeightUpdateManager | None,
    tokenizer: PreTrainedTokenizerFast,
    training_metrics: TrainingMetrics,
    batch_queue: Queue[PipelineBatchEncoding | Exception],
    weight_update_stream: SingleStreamSpec | None = None,
    seq_parallel_group = None,
):
    local_samples = torch.tensor([0], device=get_accelerator().device)
    # Create a list of tensors with matching dtype (int64)
    all_samples = [
        torch.zeros(1, dtype=torch.int64, device=get_accelerator().device)
        for _ in range(get_accelerator().state.num_processes)  
    ]
    total_samples = 0

    def batch_generator_fn():
        while True:
            timeout = 0.1
            while True:
                try:
                    batch_or_exc = batch_queue.get(timeout=timeout)
                    break
                except Empty:
                    logger.info(f"Batch queue is empty, retrying with timeout {timeout}")
                    timeout = min(timeout * 1.5, 5.0)
            if isinstance(batch_or_exc, Exception):
                raise batch_or_exc
            assert isinstance(batch_or_exc, PipelineBatchEncoding)
            yield batch_or_exc

    data_generator = batch_generator_fn()

    dt = time.perf_counter()

    output_dir = Path(args.output_dir)
    current_dir = output_dir / "current"
    intermediate_root_dir = output_dir / "intermediate"
    training_state_dir = output_dir / "training_state"

    final_train_steps = calculate_train_steps(args, args.interrupt_train_steps)
    if training_metrics.completed_steps == final_train_steps:
        logger.info("Training is already completed")
        return

    first_pass = True
    do_optimizer_step = False

    time_stats = {}
    rl_metrics = defaultdict(list)
    lag_stats = {}

    time_waiting_for_data = 0.0

    num_trainers = get_accelerator().state.num_processes
    num_lead_trainers = num_trainers // args.seq_parallel
    gradient_accumulation_passes_per_lead = args.gradient_accumulation_passes // num_lead_trainers
    samples_per_lead_per_step = gradient_accumulation_passes_per_lead * args.train_batch_size
    samples_per_step = samples_per_lead_per_step * num_lead_trainers
    start_samples = training_metrics.samples
    logger.info(
        f"Starting training with {start_samples} samples, "
        f"{samples_per_lead_per_step} samples per lead trainer, "
        f"{samples_per_step} samples per step, "
        f"{gradient_accumulation_passes_per_lead} gradient accumulation passes per lead trainer"
    )
    tokens_processed = []
    passes_took = []
    micro_batches_size = []
    target_samples_per_lead = samples_per_lead_per_step
    target_samples = samples_per_step
    rl_config = RLConfig(**args.rl)
    # samples_per_step will be used to normalize the loss
    rl_config.batch_size = samples_per_step
    while training_metrics.completed_steps < final_train_steps:
        # We include time waiting for data in the step time
        if first_pass:
            first_pass = False
            step_start_time = time.time()
            logger.info(f"Start step at {step_start_time}")

        before_getting_next_batch = time.time()

        batch = next(data_generator)
        is_sentinel_batch = batch.sentinel
        if local_samples[0] == target_samples_per_lead:
            assert is_sentinel_batch, "We should get a sentinel batch"
            logger.info("next batch should be a sentinel batch")

        time_waiting_for_data += time.time() - before_getting_next_batch
        # check if too old, don't drop but count
        if (
            args.max_lag is not None
            and training_metrics.last_broadcasted_version - batch.model_version > args.max_lag
        ):
            training_metrics.samples_too_old_to_train += args.train_batch_size
        lag_stats["min_version"] = min(
            lag_stats.get("min_version", batch.model_version), batch.model_version
        )
        lag_stats["max_version"] = max(
            lag_stats.get("max_version", batch.model_version), batch.model_version
        )

        if not is_sentinel_batch:
            # We exclude time waiting for data from the pass time
            time_before_pass = time.time()
            training_metrics.passes += 1

            # Get sequence count using the helper function
            micro_batch_size = get_batch_sequence_count(batch)
            micro_batches_size.append(micro_batch_size)

            local_samples[0] += micro_batch_size

            # track token count
            tokens_processed.append(get_batch_token_count(batch))

        if args.cuda_empty_cache:
            torch.cuda.empty_cache()

        dist.all_gather(all_samples, local_samples)
        total_samples_overcounted = sum(int(tensor.item()) for tensor in all_samples)
        assert total_samples_overcounted % args.seq_parallel == 0
        total_samples = total_samples_overcounted // args.seq_parallel
        do_optimizer_step = total_samples == target_samples
        using_deepspeed = isinstance(model, deepspeed.DeepSpeedEngine)

        def backward(loss, is_final_micro_batch=False):
            """Perform backward pass with appropriate gradient accumulation boundary"""
            if using_deepspeed:
                # Tell DeepSpeed whether this is a boundary for gradient accumulation
                model.set_gradient_accumulation_boundary(is_final_micro_batch)
                # DeepSpeed's backward
                model.backward(loss)
            else:
                # accelerator's backward
                get_accelerator().backward(loss)

        def optimizer_step_and_zero_grad():
            """Perform optimizer step and zero gradients"""
            if using_deepspeed:
                # Final boundary before optimizer step
                model.set_gradient_accumulation_boundary(True)
                model.step()
                grad_norm = model.get_global_grad_norm() if hasattr(model, "get_global_grad_norm") else None
                if isinstance(training_metrics.grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()
                training_metrics.grad_norm = grad_norm if grad_norm is not None else -1.0
            else:
                max_grad_norm = args.get("gradient_clipping_threshold", None)
                training_metrics.grad_norm = get_accelerator().clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        
        @contextlib.contextmanager
        def toggle_sync(sync: bool):
            """Wrap accelerate.no_sync() if sync is False."""
            if sync:
                yield  # do not enforce no_sync mode
            else:
                with get_accelerator().no_sync(model):
                    yield

        with toggle_sync(do_optimizer_step):
            # Choose RL step function based on seq_packing config
            if seq_parallel_group is not None:
                assert batch.seq_boundaries is not None
                update_ring_flash_attn_params(batch.seq_boundaries, seq_parallel_group)
            loss, this_step_rl_metrics = rl_step(
                model, batch, training_metrics.completed_steps, final_train_steps, rl_config
            )
            if is_sentinel_batch:
                # zero out the loss and do not update the metrics
                loss = loss * 0.0
            else:
                # update the metrics
                for k, v in this_step_rl_metrics.items():
                    rl_metrics[k].append(v)

            backward(loss, is_final_micro_batch=do_optimizer_step)

        if not is_sentinel_batch:
            passes_took.append(time.time() - time_before_pass)

        if get_accelerator().is_main_process:
            trigger_message = SamplesProcessed(samples_processed=start_samples + total_samples)
            with write_to_streams(weight_update_stream) as writer:
                writer.write(trigger_message)

        if not do_optimizer_step:
            continue

        target_samples_per_lead += samples_per_lead_per_step
        target_samples += samples_per_step

        logger.info(f"Stop step at {time.time()}")
        step_took = time.time() - step_start_time
        first_pass = True

        # All gradients have been accumulated, we can now do an optimizer step
        training_metrics.completed_steps += 1
        training_metrics.samples = start_samples + total_samples
        this_worker_tokens = sum(tokens_processed)
        training_metrics.tokens += this_worker_tokens * get_accelerator().state.num_processes
        
        try:
            # Synchronize workers before optimizer step
            logger.info("Waiting for all workers to synchronize...")
            torch.cuda.synchronize()  # Ensure CUDA operations are complete
            get_accelerator().wait_for_everyone()
            logger.info("All workers synchronized successfully")
        except Exception as e:
            logger.warning(f"Synchronization error: {e}. Continuing anyway...")

        optimizer_step_and_zero_grad()
        lr_scheduler.step()

        metrics_dict = {}
        time_to_stop = training_metrics.completed_steps >= final_train_steps
        time_to_log = training_metrics.completed_steps % args.log_each_n_steps == 0
        time_to_save = (training_metrics.completed_steps % args.save_checkpoint_steps == 0) or (
            len(args.also_save_steps) and training_metrics.completed_steps in args.also_save_steps
        )
        time_to_save = time_to_save and not time_to_stop
        assert sum(micro_batches_size) == samples_per_lead_per_step
        training_metrics.time_waiting_for_data += time_waiting_for_data
        
        
        if time_to_log or time_to_save:
            dt = log_time(dt, time_stats, "finetune/interim_eval")
            training_metrics.lr = optimizer.param_groups[0]["lr"]
            metrics_dict.update(
                {
                    "stats/lr": training_metrics.lr,
                    "stats/grad_norm": training_metrics.grad_norm,
                    "stats/samples": training_metrics.samples,
                    "stats/tokens": training_metrics.tokens,
                    "stats/samples_too_old_to_queue": training_metrics.samples_too_old_to_queue,
                    "stats/samples_too_old_to_train": training_metrics.samples_too_old_to_train,
                    "stats/passes": training_metrics.passes,
                    "stats/completed_steps": training_metrics.completed_steps,
                    "stats/epoch": training_metrics.epoch,
                    "stats/min_actor_version": lag_stats["min_version"],
                    "stats/max_actor_version": lag_stats["max_version"],
                    "stats/queue/batches": batch_queue.qsize(),
                    "stats/time_waiting_for_data": training_metrics.time_waiting_for_data,
                    "stats/lag": training_metrics.last_broadcasted_version - lag_stats["min_version"],
                    "throughput/tokens_perGPU_per_sec": this_worker_tokens / sum(passes_took) if passes_took else 0,
                    "throughput/tokens_per_step": this_worker_tokens * get_accelerator().state.num_processes,
                    "throughput/micro_batches_per_step": len(tokens_processed),
                    "throughput/min_tokens_per_micro_batch": min(tokens_processed) if tokens_processed else 0,
                    "throughput/max_tokens_per_micro_batch": max(tokens_processed) if tokens_processed else 0,
                    "throughput/tokens_per_micro_batch": this_worker_tokens / len(tokens_processed)
                    if tokens_processed
                    else 0,
                    "throughput/tokens_per_sec": this_worker_tokens
                    * get_accelerator().state.num_processes
                    / sum(passes_took)
                    if passes_took
                    else 0,
                    "throughput/real_tokens_per_sec": this_worker_tokens / step_took,
                    "throughput/sec_per_pass": sum(passes_took) / len(passes_took) if passes_took else 0,
                    "throughput/steps_per_sec": 1 / step_took if step_took else 0,
                    "throughput/samples_per_sec": samples_per_step / sum(passes_took) if passes_took else 0,
                    "throughput/sec_per_step": step_took,
                    "throughput/max_sequences_per_micro_batch": max(micro_batches_size) if micro_batches_size else 0,
                    "throughput/min_sequences_per_micro_batch": min(micro_batches_size) if micro_batches_size else 0,
                    "throughput/sequences_per_micro_batch": sum(micro_batches_size) / len(micro_batches_size)
                    if micro_batches_size
                    else 0,
                }
            )

            gathered_rl_metrics = gather_rl_metrics(rl_metrics)
            time_waiting_for_data = 0.0

            average_rl_metrics = aggregate_rl_stats(gathered_rl_metrics, samples_per_step)
            ess = (
                average_rl_metrics["rl/ratio_new_old_sum"] ** 2
                / average_rl_metrics["rl/ratio_new_old_squared_sum"]
                / average_rl_metrics["rl/num_output_tokens_sum"]
            )
            metrics_dict.update(average_rl_metrics)
            metrics_dict.update(
                {
                    "rl/ess": ess,
                }
            )

            rl_metrics = defaultdict(list)
            time_stats = {}
            lag_stats = {}

            # Reset token counting for next step
            tokens_processed = []
            passes_took = []
            micro_batches_size = []

        if len(metrics_dict):
            log_metrics(logger, training_metrics.completed_steps, metrics_dict)

        if (
            args.send_weight_updates
            and training_metrics.samples - training_metrics.last_broadcasted_version >= args.weight_update_interval
        ):
            assert weight_update_manager is not None
            weight_update_manager.send_weight_update(training_metrics.samples)
            training_metrics.last_broadcasted_version = training_metrics.samples
        get_accelerator().wait_for_everyone()

        if time_to_save:
            save_model_and_tokenizer(
                current_dir,
                model,
                tokenizer,
                args.lora.enabled,
                safe_serialization=args.use_safetensors,
            )
            # Save training state to training_state.pt (for resuming).
            save_training_state(
                training_state_dir,
                model,
                optimizer,
                lr_scheduler,
                asdict(training_metrics),
            )

            if args.keep_intermediate_checkpoints:
                intermediate_dir = intermediate_root_dir / str(training_metrics.completed_steps)
                save_model_and_tokenizer(
                    intermediate_dir,
                    model,
                    tokenizer,
                    args.lora.enabled,
                    safe_serialization=args.use_safetensors,
                )
                dt = log_time(dt, time_stats, "finetune/interim_save")

                if args.cuda_empty_cache:
                    torch.cuda.empty_cache()

            get_accelerator().wait_for_everyone()  # wait for the main process that saves the model

        if training_metrics.completed_steps >= final_train_steps:
            logger.info(f"Reached final step {final_train_steps}, stopping.")
            break
    dt = log_time(dt, time_stats, "finetune/train_loop")

    logger.info("Final model saving")
    save_model_and_tokenizer(
        current_dir,
        model,
        tokenizer,
        args.lora.enabled,
        safe_serialization=args.use_safetensors,
    )
    dt = log_time(dt, time_stats, "finetune/final_save")
    if args.save_final_training_state:
        save_training_state(
            training_state_dir,
            model,
            optimizer,
            lr_scheduler,
            asdict(training_metrics),
        )
        dt = log_time(dt, time_stats, "finetune/final_training_state_save")

    if get_accelerator().is_main_process:
        with open(output_dir / "summary.json", "w") as wf:
            json.dump(asdict(training_metrics), wf, indent=4, sort_keys=True)
        with open(output_dir / "rl_summary.json", "w") as wf:
            json.dump(rl_metrics, wf, indent=4, sort_keys=True)

    torch.cuda.empty_cache()


def calculate_train_steps(args, interrupt_train_steps):
    if interrupt_train_steps == -1:
        assert args.interrupt_train_steps <= args.max_train_steps
        final_train_steps = args.max_train_steps if args.interrupt_train_steps < 0 else args.interrupt_train_steps
    else:
        assert interrupt_train_steps <= args.max_train_steps
        final_train_steps = interrupt_train_steps
    return final_train_steps
