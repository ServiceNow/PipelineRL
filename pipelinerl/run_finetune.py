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

import pipelinerl.torch_utils
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
from pipelinerl.finetune.rl.utils import get_avg_rl_stats
from pipelinerl.finetune.types import TrainingMetrics
from pipelinerl.finetune.utils import VersionedTensors, create_sentinel_batch
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
    # Initialize the result dictionary
    gathered_rl_metrics = {}

    # Process each metric separately
    for key, values in rl_metrics.items():
        if values:
            # Initialize a list to gather the results from all processes
            gathered_values = [None] * dist.get_world_size()

            # Gather the values from all processes
            dist.all_gather_object(gathered_values, values)

            # Flatten the list of lists into a single list
            combined_values = []
            for process_values in gathered_values:
                combined_values.extend(process_values)

            # Store the combined values
            gathered_rl_metrics[key] = combined_values

    return gathered_rl_metrics


def run_sample_loader(data_stream: SingleStreamSpec, sample_queue: Queue[Dict | Exception], pop_old_data: bool = False):
    with read_stream(data_stream) as stream_reader:
        pop_count = 0  # Counter for popped items
        while True:
            try:
                for data_item in stream_reader.read():
                    if pop_old_data:
                        if sample_queue.full():
                            sample_queue.get()  # Pop old data
                            pop_count += 1
                            if pop_count % 100 == 0:
                                logger.info(f"Popped {pop_count} old items from the sample queue")
                        sample_queue.put_nowait(data_item)
                    else:
                        sample_queue.put(data_item)
            except Exception as e:
                logger.error(f"Error in stream reader: {e}")
                sample_queue.put(e)
                break


def sample_generator_fn(sample_queue):
    while True:
        timeout = 0.1
        while True:
            try:
                sample_or_exc = sample_queue.get(timeout=timeout)
                break
            except Empty:
                logger.info(f"Sample queue is empty, retrying with timeout {timeout}")
                timeout = min(timeout * 1.5, 5.0)
                yield None
        if isinstance(sample_or_exc, Exception):
            raise sample_or_exc
        assert isinstance(sample_or_exc, Dict)
        yield sample_or_exc


def run_fixed_batch_data_loader(
    sample_queue: Queue[Dict | Exception],
    batch_queue: Queue[VersionedTensors | Exception],
    batch_size: int,
    tokenizer: PreTrainedTokenizerFast,
):
    """Incrementally load chunks to populate the dataset queue."""
    sample_generator = sample_generator_fn(sample_queue)
    while True:
        try:
            buffer = []
            while True:
                entry = next(sample_generator)
                if entry is None:
                    continue
                buffer.append(entry)
                if len(buffer) == batch_size:
                    batch = collate(buffer, tokenizer=tokenizer)
                    batch = {
                        k: (v.to(get_accelerator().device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()
                    }
                    batch = VersionedTensors(
                        tensors=batch, model_version=min(sample["model_version"] for sample in buffer)
                    )
                    batch_queue.put(batch)
                    logger.debug(f"Loaded {len(buffer)} samples, queue size is now {batch_queue.qsize()}")
                    break

        except Exception as e:
            logger.error(f"Error in dataset loader: {e}")
            batch_queue.put(e)
            break


def run_dynamic_batch_size_data_loader(
    sample_queue: Queue[Dict | Exception],
    batch_queue: Queue[VersionedTensors | Exception],
    max_seq_length: int,
    samples_per_worker_per_step: int,
    tokenizer: PreTrainedTokenizerFast,
):
    """Incrementally load chunks to populate the dataset queue."""

    current_batch = []
    current_length = 0
    samples_in_step = 0
    sample_generator = sample_generator_fn(sample_queue)
    skip_count = 0
    while True:
        try:
            while True:
                # TODO: handle timeout
                entry = next(sample_generator)
                sample_length = len(entry["input_ids"]) if entry else 0

                if sample_length > max_seq_length:
                    skip_count += 1
                    logger.warning(
                        f"Sample length {sample_length} > max allowed length {max_seq_length}, skipping. Total {skip_count} samples skipped so far."
                    )
                    continue

                # check if adding current sample would exceed max_seq_length or if we've reached sample limit
                boundary = samples_in_step == samples_per_worker_per_step
                if (current_length + sample_length > max_seq_length) or boundary:
                    logger.debug(
                        f"Adding batch with total sequence length {current_length} and {len(current_batch)} samples"
                    )
                    collated_batch = collate_packed(current_batch, tokenizer=tokenizer)
                    collated_batch = {
                        k: (v.to(get_accelerator().device) if isinstance(v, torch.Tensor) else v)
                        for k, v in collated_batch.items()
                    }
                    collated_batch = VersionedTensors(
                        tensors=collated_batch,
                        model_version=min(sample["model_version"] for sample in current_batch),
                    )
                    batch_queue.put(collated_batch)

                    current_batch = []
                    current_length = 0
                    if boundary:
                        samples_in_step = 0

                if entry:
                    # add sample to current batch
                    current_batch.append(entry)
                    current_length += sample_length
                    samples_in_step += 1

        except Exception as e:
            logger.error(f"Error in dataset loader: {e}")
            batch_queue.put(e)
            break


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


class WeightBeingSavedToDisk(BaseModel):
    kind: Literal["weight_being_saved_to_disk"] = "weight_being_saved_to_disk"
    version: int
    timestamp: float = time.time()


TrainerMessage = WeightUpdateRequest | WeightUpdateSuccess | WeightBeingSavedToDisk


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
            named_parameters = dict(module.named_parameters())

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
    attention_mask = batch.get("attention_mask")
    assert attention_mask is not None, "We need attention_mask for accurate token counting"
    return attention_mask.sum().item()


def get_batch_sequence_count(batch):
    """Count actual sequences in a batch, accounting for sequence packing"""
    is_packed = "position_ids" in batch
    if is_packed:
        # for packed sequences, count the number of actual sequences by looking at position_ids
        position_ids = batch["position_ids"][0]  # [1, seq_len]
        # sequence boundary computation - each position with id 0 starts a new sequence
        sequence_starts = torch.where(position_ids == 0)[0]
        return len(sequence_starts)
    else:
        # For unpacked sequences, each row is one sample
        return batch["input_ids"].size(0)


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
    gradient_accumulation_passes_per_gpu = args.gradient_accumulation_passes // num_processes
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

    # Render-vous with inference servers
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

    if args.seq_packing:
        samples_per_worker_per_step = gradient_accumulation_passes_per_gpu * args.train_batch_size
        run_data_loader = partial(
            run_dynamic_batch_size_data_loader,
            max_seq_length=args.seq_length,
            samples_per_worker_per_step=samples_per_worker_per_step,
        )
    else:
        run_data_loader = partial(run_fixed_batch_data_loader, batch_size=args.train_batch_size)

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
        weight_update_stream = None
    get_accelerator().wait_for_everyone()

    training_metrics = TrainingMetrics()
    if os.path.exists(training_state_dir):
        # WARNING: In case of deepspeed this will overwrite model weights too
        training_metrics = load_training_state(training_state_dir, model, optimizer, lr_scheduler, training_metrics)
        training_metrics.lr = optimizer.param_groups[0]["lr"]
        logger.info("LR after loading training state: %.2E" % training_metrics.lr)
        dt = log_time(dt, time_stats, "finetune/training_state_load")

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

    sample_queue = Queue(maxsize=args.queue_size)
    sample_loader_worker_fn = partial(
        run_sample_loader,
        data_stream=data_stream,
        sample_queue=sample_queue,
        pop_old_data=cfg.max_lag is None and cfg.pop_old_data and not cfg.debug.mode,
    )

    stream_to_queue_thread = threading.Thread(target=sample_loader_worker_fn, args=())
    stream_to_queue_thread.start()

    batch_queue = Queue(maxsize=1)
    data_loader_worker_fn = partial(
        run_data_loader,
        batch_queue=batch_queue,
        sample_queue=sample_queue,
        tokenizer=tokenizer,
    )
    data_loader_thread = threading.Thread(target=data_loader_worker_fn, args=())

    get_accelerator().wait_for_everyone()
    model.train()
    data_loader_thread.start()

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
            sample_queue,
            batch_queue,
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
    sample_queue: Queue[Dict | Exception],
    batch_queue: Queue[VersionedTensors | Exception],
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
            assert isinstance(batch_or_exc, VersionedTensors)
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

    samples_per_worker_per_step = gradient_accumulation_passes_per_gpu * args.train_batch_size
    samples_per_step = samples_per_worker_per_step * get_accelerator().state.num_processes
    start_samples = training_metrics.samples
    last_model_version = training_metrics.samples
    tokens_processed = []
    passes_took = []
    micro_batches_size = []
    target_samples_per_worker = samples_per_worker_per_step
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

        # check if current worker has process enough data to do a grad step
        if local_samples[0] == target_samples_per_worker:
            logger.debug("creating sentinel batch")
            versioned_batch = create_sentinel_batch(
                get_accelerator().device, tokenizer, model_version=last_model_version
            )
            is_sentinel_batch = True
        else:
            versioned_batch = next(data_generator)
            is_sentinel_batch = False

        time_waiting_for_data += time.time() - before_getting_next_batch
        # check if too old, don't drop but count
        if (
            args.max_lag is not None
            and training_metrics.last_broadcasted_version - versioned_batch.model_version > args.max_lag
        ):
            training_metrics.samples_too_old_to_train += args.train_batch_size
        batch = versioned_batch.tensors
        lag_stats["min_version"] = min(
            lag_stats.get("min_version", versioned_batch.model_version), versioned_batch.model_version
        )
        lag_stats["max_version"] = max(
            lag_stats.get("max_version", versioned_batch.model_version), versioned_batch.model_version
        )
        last_model_version = versioned_batch.model_version

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
        total_samples = sum(int(tensor.item()) for tensor in all_samples)
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

                training_metrics.lr = optimizer.param_groups[0]["lr"]

            backward(loss, is_final_micro_batch=do_optimizer_step)

        if not is_sentinel_batch:
            passes_took.append(time.time() - time_before_pass)

        logger.debug(f"Did a pass, version was {versioned_batch.model_version}")
        if not do_optimizer_step:
            continue

        target_samples_per_worker += samples_per_worker_per_step
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
        assert sum(micro_batches_size) == samples_per_worker_per_step
        training_metrics.time_waiting_for_data += time_waiting_for_data
        if time_to_log or time_to_save:
            dt = log_time(dt, time_stats, "finetune/interim_eval")
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
                    "stats/queue/samples": sample_queue.qsize(),
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
                    "dataset_stats/max_batch_len": training_metrics.max_batch_len,
                    "dataset_stats/min_batch_len": training_metrics.min_batch_len,
                }
            )

            gathered_rl_metrics = gather_rl_metrics(rl_metrics)
            time_waiting_for_data = 0.0

            average_rl_metrics = get_avg_rl_stats(gathered_rl_metrics, samples_per_step)
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
