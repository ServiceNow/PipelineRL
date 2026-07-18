import os
from collections import defaultdict, deque

os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

import logging
import queue
import threading
import time
from dataclasses import dataclass
from functools import partial
from multiprocessing import Process, Queue
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from queue import Empty
from typing import List

import datasets
import transformers
from litellm import BaseModel, Field

from pipelinerl.finetune.logging_ import flatten_dict_config
from pipelinerl.finetune_loop import calculate_train_steps, samples_per_optimizer_step
from pipelinerl.shared_memory_array import EntrySizeExceeded, SharedMemoryQueue
from pipelinerl.state import TrainerState
from pipelinerl.utils import init_wandb, setup_logging, wait_for_inference_servers
from pipelinerl.world import WorldMap

datasets.disable_caching()
import traceback

from omegaconf import DictConfig

from pipelinerl.finetune.checkpoints import (
    load_tokenizer,
)
from pipelinerl.finetune.data import collate, collate_packed, preprocess_fn
from pipelinerl.finetune.rl import RLConfig, populate_rl_data
from pipelinerl.finetune.types import PipelineBatchEncoding
from pipelinerl.finetune.utils import create_sentinel_batch, create_sentinel_example
from pipelinerl.llm import TrainableLLM
from pipelinerl.rollouts import (
    TrainingGroupEnvelope,
    cached_tokenizer_token_ids,
)
from pipelinerl.streams import (
    SingleStreamSpec,
    StreamRangeSpec,
    StreamWriter,
    read_stream,
    set_streams_backend,
    write_to_streams,
)

logger = logging.getLogger(__name__)


@dataclass
class AtomicUpdate:
    entries: list[dict]
    n_groups: int
    padding: int


class AtomicGroupBuffer:
    """Bound complete training-group envelopes and stage one protected update."""

    def __init__(self, capacity: int, samples_per_step: int):
        if capacity <= 0 or samples_per_step <= 0:
            raise ValueError("Atomic queue capacity and samples_per_step must be positive")
        self.capacity = capacity
        self.samples_per_step = samples_per_step
        self.ready_groups = deque()
        self.ready_entries = 0
        self.staged_groups = []
        self.staged_entries = 0
        self.evicted_groups = 0
        self.evicted_entries = 0
        self.rejected_groups = 0
        self.rejected_entries = 0
        self.updates = 0
        self.padding = 0

    def enqueue(self, envelope: TrainingGroupEnvelope, pop_old_data: bool) -> bool:
        """Queue one envelope; return False only when backpressure must retain it."""
        group_size = len(envelope.entries)
        if group_size > self.samples_per_step or group_size > self.capacity:
            self.rejected_groups += 1
            self.rejected_entries += group_size
            logger.warning(
                "Rejecting atomic envelope %s with %d entries: update capacity=%d, ready capacity=%d",
                envelope.group_id,
                group_size,
                self.samples_per_step,
                self.capacity,
            )
            return True
        if not pop_old_data and self.ready_entries + group_size > self.capacity:
            return False
        while self.ready_groups and self.ready_entries + group_size > self.capacity:
            evicted = self.ready_groups.popleft()
            self.ready_entries -= len(evicted.entries)
            self.evicted_groups += 1
            self.evicted_entries += len(evicted.entries)
        self.ready_groups.append(envelope)
        self.ready_entries += group_size
        return True

    def compose_update(self) -> AtomicUpdate | None:
        """Move whole envelopes into the protected accumulator until an update closes."""
        while self.ready_groups:
            envelope = self.ready_groups[0]
            group_size = len(envelope.entries)
            if self.staged_entries + group_size > self.samples_per_step:
                return self._finish_update()
            self.ready_groups.popleft()
            self.ready_entries -= group_size
            self.staged_groups.append(envelope)
            self.staged_entries += group_size
            if self.staged_entries == self.samples_per_step:
                return self._finish_update()
        return None

    def _finish_update(self) -> AtomicUpdate:
        assert self.staged_groups and 0 < self.staged_entries <= self.samples_per_step
        entries = [
            entry
            for envelope in self.staged_groups
            for entry in envelope.entries
        ]
        update = AtomicUpdate(
            entries=entries,
            n_groups=len(self.staged_groups),
            padding=self.samples_per_step - self.staged_entries,
        )
        self.staged_groups = []
        self.staged_entries = 0
        self.updates += 1
        self.padding += update.padding
        return update


def materialize_atomic_update(
    update: AtomicUpdate,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> tuple[list[dict], int]:
    newest_model_version = max(entry["model_version"] for entry in update.entries)
    entries = list(update.entries)
    template = update.entries[0]
    for _ in range(update.padding):
        sentinel = create_sentinel_example(
            8,
            tokenizer=tokenizer,
            model_version=newest_model_version,
        )
        for key, value in template.items():
            if key in sentinel:
                continue
            if isinstance(value, list):
                fill_value = (
                    0.0
                    if value and isinstance(value[0], float)
                    else 0
                )
                sentinel[key] = [fill_value] * 8
            elif isinstance(value, str):
                sentinel[key] = ""
            elif isinstance(value, dict):
                sentinel[key] = {}
            elif value is None:
                sentinel[key] = None
            else:
                sentinel[key] = type(value)()
        entries.append(sentinel)
    return entries, newest_model_version


def _needs_reference_logprobs(rl_config: RLConfig) -> bool:
    return max(rl_config.kl_coef, rl_config.final_kl_coef) > 0.0


def _get_worker_ref_llm(worker_idx: int, llms: list[TrainableLLM]) -> TrainableLLM | None:
    return llms[worker_idx % len(llms)] if llms else None


def _validate_reference_logprob_setup(rl_config: RLConfig, llm_urls: list[str]):
    if _needs_reference_logprobs(rl_config) and not llm_urls:
        raise ValueError(
            "Reference logprobs require preprocessor reference LLM URLs. "
            "Set world.preprocessor_fraction > 0 and launch the preprocessor with reference servers."
        )



def _check_group_sizes(texts: list[dict], group_size: int) -> bool:
    """Check that each group_id occures exactly group_size times."""
    group_rollouts = defaultdict(set)
    for text in texts:
        group_id = text["group_id"]
        rollout_index = text["metadata"]["rollout_index"]
        group_rollouts[group_id].add(rollout_index)

    for group_id, rollout_ids in group_rollouts.items():
        if len(rollout_ids) != group_size:
            logger.error(f"Group sizes are wrong: {group_rollouts}")
            return False

    return True


def batch_annotate_traces_with_ref_logprobs(llm: TrainableLLM, traces: List[dict]):
    logger.info(f"Annotating {len(traces)} samples with ref logprobs")
    prompt_token_ids = []
    completion_token_ids = []
    full_alignment = []
    for trace in traces:
        is_full_alignment = len(trace["logprobs"]) == len(trace["input_ids"])
        full_alignment.append(is_full_alignment)
        if is_full_alignment:
            if len(trace["input_ids"]) < 2:
                raise ValueError("Full-length logprob alignment requires at least two tokens")
            prompt_token_ids.append(trace["input_ids"][:1])
            completion_token_ids.append(trace["input_ids"][1:])
        else:
            prompt_token_ids.append(trace["input_ids"][: -len(trace["logprobs"])])
            completion_token_ids.append(trace["input_ids"][-len(trace["logprobs"]) :])
    try:
        all_ref_logprobs = llm.get_batch_logprobs_token_ids(prompt_token_ids, completion_token_ids)
    except Exception as e:
        logger.error(f"Failed to get ref logprobs: {e}")
        assert (response := getattr(e, "response", None))
        logger.error(f"Response content: {response.text}")
        raise e
    for trace, ref_logprobs, is_full_alignment in zip(traces, all_ref_logprobs, full_alignment):
        values = [c["logprob"] for c in ref_logprobs["content"]]
        if is_full_alignment:
            assert len(values) == len(trace["input_ids"]) - 1, (
                f"{len(values)} != {len(trace['input_ids']) - 1}"
            )
            values = [0.0] + values
            assert len(trace["labels"]) == len(values)
            trace["ref_logprobs"] = [
                value if label != -100 else 0.0
                for value, label in zip(values, trace["labels"])
            ]
        else:
            trace["ref_logprobs"] = values
            assert len(trace["ref_logprobs"]) == len(trace["logprobs"]), (
                f"{len(trace['ref_logprobs'])} != {len(trace['logprobs'])}"
            )


def replace_oov_tokens_with_the(data: list[dict], tokenizer: transformers.PreTrainedTokenizerBase) -> list[dict]:
    patched_entries = 0

    # TODO: yes this is slow. But should not be the bottleneck. We have to pickle the entire tokenizer
    # every time we sent a task to the process pool anyway.
    token_ids = set(tokenizer.get_vocab().values())
    the_token_id = tokenizer.get_vocab()["the"]

    new_data = []
    for entry in data:
        new_input_ids = []
        invalid_token_ids = []
        for token_id in entry["input_ids"]:
            if token_id not in token_ids:
                new_input_ids.append(the_token_id)
                invalid_token_ids.append(token_id)
            else:
                new_input_ids.append(token_id)
        if invalid_token_ids:
            patched_entries += 1
            logger.warning(f"Patching entry with invalid token ids: {invalid_token_ids}")
            # Also need to update logprobs if they exist since we're changing tokens
            if "logprobs" in entry and len(entry["logprobs"]) > 0:
                # Find positions of invalid tokens in the completion part
                completion_length = len(entry["logprobs"])
                completion_start = len(entry["input_ids"]) - completion_length
                for i, token_id in enumerate(invalid_token_ids):
                    if i + completion_start < len(entry["input_ids"]):
                        logger.warning("Invalid token in completion part, logprobs may be inconsistent")
        entry["input_ids"] = new_input_ids
        new_data.append(entry)

    if patched_entries > 0:
        logger.warning(f"Patched {patched_entries} entries with invalid token ids from {len(data)}")

    return new_data


def preprocess_dataset(
    llm: TrainableLLM | None,
    data: list[dict],
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    rl_config: RLConfig,
    rewrite_oov_tokens: bool = True,
) -> list[dict]:
    preprocess = partial(preprocess_fn, seq_length=seq_length, tokenizer=tokenizer, is_rl=True)

    if rewrite_oov_tokens:
        data = replace_oov_tokens_with_the(data, tokenizer)

    # inplace update of the traces with ref logprobs
    if llm is not None:
        batch_annotate_traces_with_ref_logprobs(llm, data)
    else:
        for entry in data:
            entry["ref_logprobs"] = entry["logprobs"]

    # now without Huggingface datasets
    dataset = []
    for i in range(len(data)):
        entry = dict(data[i])
        for k, v in preprocess(data[i]).items():
            entry[k] = v
        dataset.append(entry)        
    for entry in dataset:
        entry["model_version"] = entry["metadata"]["model_version"]
        entry["rollout_index"] = entry["metadata"]["rollout_index"]
        entry["step_index"] = entry["metadata"]["step_index"]
    if not isinstance(tokenizer.eos_token_id, int):
        raise ValueError(f"Tokenizer {tokenizer} does not have an eos_token_id")
    try:
        dataset = populate_rl_data(dataset=dataset, eos_token_id=tokenizer.eos_token_id, config=rl_config)
    except Exception as e:
        logger.error(f"Error in populate_rl_data: {e}", extra={
            "data": data,
            "dataset": dataset,
            "tokenizer": tokenizer,
            "eos_token_id": tokenizer.eos_token_id,
            "rl_config": rl_config,
            "llm": llm,
            "seq_length": seq_length,
        })
        raise
    return dataset


def atomic_envelope_oov_token_ids(
    envelope: TrainingGroupEnvelope,
    tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[int]:
    valid_token_ids = cached_tokenizer_token_ids(tokenizer)
    return sorted(
        {
            token_id
            for entry in envelope.entries
            for token_id in entry["input_ids"]
            if token_id not in valid_token_ids
        }
    )


def atomic_group_rejection(
    envelope: TrainingGroupEnvelope,
    reason: str,
    queue_hop: str,
    **details,
) -> dict:
    return {
        "kind": "atomic_group_rejection",
        "group_id": envelope.group_id,
        "reason": reason,
        "queue_hop": queue_hop,
        **details,
    }


def put_atomic_input_queue(
    input_queue: SharedMemoryQueue,
    envelope: TrainingGroupEnvelope,
    rejection_counts: dict[str, int],
) -> bool:
    try:
        input_queue.put(envelope)
    except EntrySizeExceeded as exc:
        rejection_counts["preprocess_input/queue_oversize"] += 1
        logger.warning(
            "Dropping atomic group %s at preprocess input: %d > %d bytes",
            envelope.group_id,
            exc.size,
            exc.max_size,
        )
        return False
    return True


def put_atomic_output_queue(
    output_queue: SharedMemoryQueue,
    processed: TrainingGroupEnvelope | dict,
    source_envelope: TrainingGroupEnvelope,
) -> None:
    try:
        output_queue.put(processed)
    except EntrySizeExceeded as exc:
        if not isinstance(processed, TrainingGroupEnvelope):
            raise
        output_queue.put(
            atomic_group_rejection(
                source_envelope,
                "queue_oversize",
                "preprocess_output",
                serialized_size=exc.size,
                max_size=exc.max_size,
            )
        )


def count_atomic_group_rejection(
    rejection: dict,
    rejection_counts: dict[str, int],
) -> str:
    counter_key = f"{rejection['queue_hop']}/{rejection['reason']}"
    rejection_counts[counter_key] += 1
    return counter_key


def preprocess_atomic_envelope(
    llm: TrainableLLM | None,
    envelope: TrainingGroupEnvelope,
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    rl_config: RLConfig,
) -> TrainingGroupEnvelope | dict:
    invalid_token_ids = atomic_envelope_oov_token_ids(envelope, tokenizer)
    if invalid_token_ids:
        return atomic_group_rejection(
            envelope,
            "oov_token_ids",
            "preprocess_validation",
            invalid_token_ids=invalid_token_ids,
        )
    dataset = preprocess_dataset(
        llm=llm,
        data=envelope.entries,
        tokenizer=tokenizer,
        seq_length=seq_length,
        rl_config=rl_config,
        rewrite_oov_tokens=False,
    )
    return envelope.model_copy(update={"entries": dataset})



def run_dataset_loader(
    raw_chunk_queue: Queue,
    data_stream: SingleStreamSpec,
    check_group_size: int,
    chunk_n_groups: int,
    pop_old_data: bool,
):
    old_and_dropped = 0
    last_time_notice = 0
    with read_stream(data_stream) as reader:
        while True:
            try:
                buffer = []
                n_groups = 0
                atomic_envelope = None
                for group in reader.read():
                    if (
                        isinstance(group, dict)
                        and group.get("kind") == "atomic_training_group"
                    ):
                        if n_groups > 0:
                            raise ValueError(
                                "stream mixes atomic envelopes and legacy groups"
                            )
                        atomic_envelope = TrainingGroupEnvelope.model_validate(group)
                        if (
                            atomic_envelope.expected_rollouts != check_group_size
                            or not atomic_envelope.entries
                            or any(
                                entry["group_id"] != atomic_envelope.group_id
                                for entry in atomic_envelope.entries
                            )
                            or not _check_group_sizes(
                                atomic_envelope.entries,
                                atomic_envelope.expected_rollouts,
                            )
                        ):
                            raise ValueError("Invalid atomic group envelope")
                        try:
                            raw_chunk_queue.put_nowait(atomic_envelope)
                        except queue.Full:
                            if pop_old_data:
                                try:
                                    raw_chunk_queue.get_nowait()
                                    old_and_dropped += 1
                                    if old_and_dropped // 100 != last_time_notice:
                                        logger.info(
                                            f"So far removed {old_and_dropped} old elements "
                                            "from preprocessor queue"
                                        )
                                        last_time_notice = old_and_dropped // 100
                                except Empty:
                                    pass
                            raw_chunk_queue.put(atomic_envelope)
                        break
                    buffer.extend(group)
                    n_groups += 1
                    if n_groups == chunk_n_groups:
                        break
                if atomic_envelope is not None:
                    continue
                if not _check_group_sizes(buffer, check_group_size):
                    raise ValueError("Invalid group sizes in data")
                try:
                    raw_chunk_queue.put_nowait(buffer)
                except queue.Full:
                    # Try to remove oldest element if queue is full
                    if pop_old_data:
                        try:
                            raw_chunk_queue.get_nowait()
                            old_and_dropped += 1
                            if old_and_dropped // 100 != last_time_notice:
                                logger.info(f"So far removed {old_and_dropped} old elements from preprocessor queue")
                                last_time_notice = old_and_dropped // 100
                        except Empty:
                            pass
                    # Put new element in now that we made space
                    # This is a blocking call, but in most cases there will be space
                    raw_chunk_queue.put(buffer)
            except Exception as e:
                logger.error(f"Error in dataset loader: {e}")
                raw_chunk_queue.put(e)
                break


class SlidingWindowData(BaseModel):
    tokens_window: list[list[int]] = Field(
        default_factory=list,
        description="Token counts for each chunk in the window",
    )
    timestamps: list[float] = Field(default_factory=list)


class SlidingWindowAggregator:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data = SlidingWindowData()

    def has_enough_data(self):
        return len(self.data.tokens_window) == self.window_size

    def update(self, token_counts: list[int]):
        self.data.tokens_window.append(token_counts)
        self.data.timestamps.append(time.time())
        if len(self.data.tokens_window) > self.window_size:
            self.data.tokens_window.pop(0)
            self.data.timestamps.pop(0)

    def get_stats(self):
        # 1. How many samples do we produce per second?
        # 2. How many total tokens do we produce per second?
        null_stats = {
            "samples_per_second": 0,
            "tokens_per_second": 0,
        }
        if not self.data.timestamps:
            return null_stats

        time_span = self.data.timestamps[-1] - self.data.timestamps[0]
        if time_span < 1e-6:
            return null_stats

        num_samples = sum(len(tokens) for tokens in self.data.tokens_window)
        total_tokens = sum(sum(tokens) for tokens in self.data.tokens_window)

        return {
            "samples_per_second": num_samples / time_span,
            "tokens_per_second": total_tokens / time_span,
        }


def process_chunk(
    llm: TrainableLLM | None,
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    rl_config: RLConfig,
    input_queue: SharedMemoryQueue,
    output_queue: SharedMemoryQueue,
):
    """Worker process function to preprocess chunks of data"""
    try:
        if llm is not None and llm.tokenizer is None:
            llm.tokenizer = tokenizer
        worker_ref_source = llm.base_url if llm is not None else "rollout logprobs fallback"
        logger.info(f"Preprocessor worker started with reference source: {worker_ref_source}")
        while True:
            chunk = None
            try:
                chunk = input_queue.get()
                if isinstance(chunk, TrainingGroupEnvelope):
                    processed = preprocess_atomic_envelope(
                        llm=llm,
                        envelope=chunk,
                        tokenizer=tokenizer,
                        seq_length=seq_length,
                        rl_config=rl_config,
                    )
                    put_atomic_output_queue(output_queue, processed, chunk)
                else:
                    dataset = preprocess_dataset(
                        llm=llm,
                        data=chunk,
                        tokenizer=tokenizer,
                        seq_length=seq_length,
                        rl_config=rl_config,
                    )
                    output_queue.put(dataset)
            except Exception as e:
                error_info = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                output_queue.put(error_info)
    except KeyboardInterrupt:
        return


def filter_zero_advantage_groups(dataset: list[dict], epsilon: float = 1e-6) -> tuple[list[dict], int]:
    """
    Filter out groups where all advantages are zero.
    
    Args:
        dataset: List of dataset entries with group_id and advantages
        epsilon: Threshold for considering advantage non-zero
        
    Returns:
        Tuple of (filtered_entries, num_filtered_out)
    """
    filtered_entries = []
    groups = {}
    
    # Group entries by group_id
    for entry in dataset:
        group_id = entry["group_id"]
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(entry)
    
    num_filtered_out = 0
    
    # Filter groups based on advantage values
    for group_id, entries in groups.items():
        has_non_zero_advantage = False
        for entry in entries:
            # advantages is a list, check if any absolute value is > epsilon
            if any(abs(adv) > epsilon for adv in entry["advantages"]):
                has_non_zero_advantage = True
                break
        
        if has_non_zero_advantage:
            filtered_entries.extend(entries)
        else:
            num_filtered_out += len(entries)
    
    return filtered_entries, num_filtered_out


def write_micro_batch_slices(
    lead_trainer_id: int,
    data_writer: StreamWriter,
    micro_batch: PipelineBatchEncoding,
    seq_parallel: int
):
    if seq_parallel > 1:
        # make micro batch slices and write each to the corresponding trainer
        for index, micro_slice in enumerate(micro_batch.make_slices(seq_parallel)):
            data_writer.write(micro_slice, lead_trainer_id + index)
    else:
        data_writer.write(micro_batch, lead_trainer_id)


def run_preprocessing_loop(
    
    cfg: DictConfig,
):
    set_streams_backend(**cfg.streams)

    world_map = WorldMap(cfg, verbose=True)
    exp_root_dir = Path(cfg.output_dir)
    preprocess_dir = exp_root_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(preprocess_dir, "preprocessor")

    if cfg.wandb.use_wandb:
        wandb_run = init_wandb(cfg, preprocess_dir, flatten_dict_config(cfg))
        if wandb_run is None:
            raise ValueError("Failed to initialize wandb run")
    else:
        wandb_run = None

    tokenizer = load_tokenizer(cfg.finetune.config_name)
    
    llm_urls = str(cfg.me.llm_urls).split("+") if cfg.me.llm_urls else []
    rl_config = RLConfig(**cfg.finetune.rl)
    _validate_reference_logprob_setup(rl_config, llm_urls)
    logger.info(f"Discovered {len(llm_urls)} reference LLM urls")
    if llm_urls:
        wait_for_inference_servers(llm_urls)

    input_stream = SingleStreamSpec(exp_path=exp_root_dir, topic=cfg.preprocess.input)
    output_stream = StreamRangeSpec(
        exp_path=exp_root_dir,
        topic=cfg.preprocess.output,
        partition_range=(0, max(world_map.total_finetune_gpus, 1)),
    )
    stats_streams = SingleStreamSpec(exp_path=exp_root_dir, topic="preprocessor_stats")
    logger.info("Streams initialized")

    raw_chunk_queue = Queue(cfg.preprocess.raw_queue_size)
    pop_old_data = cfg.max_lag is None and cfg.pop_old_data and not cfg.debug.mode
    dataset_loader_worker_fn = partial(
        run_dataset_loader,
        raw_chunk_queue=raw_chunk_queue,
        data_stream=input_stream,
        check_group_size=cfg.attempts,
        chunk_n_groups=cfg.preprocess.chunk_n_groups,
        pop_old_data=pop_old_data,
    )
    # Start the dataset loader thread using Thread
    dataset_loader_thread = threading.Thread(target=dataset_loader_worker_fn, daemon=True)
    dataset_loader_thread.start()
    
    # Initialize TrainerState
    trainer_state = TrainerState(exp_root_dir)
    if cfg.debug.mode == "preprocessor":
        logger.info("Debug mode: preprocessor")
        trainer_state.debug_mode_init()
    elif cfg.debug.mode == "finetune+preprocessor":
        logger.info("Debug mode: finetune+preprocessor")
        trainer_state.start_listening()
        trainer_state.wait_for_processed_samples()
    else:
        logger.info("Normal mode, waiting for finetune loop to start")
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()
    final_train_steps = calculate_train_steps(cfg.finetune, cfg.finetune.interrupt_train_steps)
    samples_target = final_train_steps * cfg.finetune.train_batch_size * cfg.finetune.gradient_accumulation_passes

    # Load published samples from state file
    llms = [
        TrainableLLM(
            base_url=url,
            model_name=cfg.finetune.config_name,
            tokenizer_name=cfg.finetune.config_name,
            parameters=cfg.llm.parameters,
        )
        for url in llm_urls
    ]

    submitted_chunks = 0
    processed_chunks = 0
    worker_pool_size = cfg.preprocess.n_workers
    processed_entries_queue_popped_data = 0
    last_time_notice = 0
    trainer_id = 0
    published_samples = None
    batch_done = False

    stats_aggregator = SlidingWindowAggregator(window_size=max(10, 1000 // cfg.preprocess.chunk_n_groups))

    buffer = deque()
    
    # Sequence packing configuration
    num_trainers = world_map.total_finetune_gpus
    num_lead_trainers = world_map.total_finetune_gpus // cfg.finetune.seq_parallel
    gradient_accumulation_passes_per_lead = cfg.finetune.gradient_accumulation_passes // num_lead_trainers
    samples_per_lead_per_step = cfg.finetune.train_batch_size * gradient_accumulation_passes_per_lead
    train_batch_size = samples_per_lead_per_step * num_lead_trainers
    samples_per_step = samples_per_optimizer_step(cfg.finetune)
    assert train_batch_size == samples_per_step
    processed_entries_queue = deque(maxlen=cfg.preprocess.ring_buffer_size)
    atomic_pending_envelopes = deque()
    atomic_group_buffer = AtomicGroupBuffer(
        int(cfg.preprocess.ring_buffer_size),
        samples_per_step,
    )
    atomic_mode = None
    published_samples = trainer_state.wait_for_processed_samples()
    last_published_samples = published_samples
    assert published_samples % num_lead_trainers == 0
    samples_per_trainer = {
        idx: published_samples // num_trainers 
        for idx in range(0, num_trainers, cfg.finetune.seq_parallel)
    }

    max_model_version = None
    time_to_write = False
    current_batch = []
    current_length = 0
    batch_boundary = published_samples + train_batch_size
    target_samples_per_lead = samples_per_trainer[0] + samples_per_lead_per_step
    
    # Per-trainer sample tracking (similar to finetune_loop.py)
    total_filtered_out = 0  # Track total filtered samples across all batches

    atomic_rejection_counts = defaultdict(int)
    last_atomic_update_groups = 0
    last_atomic_update_real_entries = 0
    last_atomic_update_padding = 0

    def prepare_atomic_update() -> None:
        nonlocal max_model_version
        nonlocal last_atomic_update_groups
        nonlocal last_atomic_update_real_entries
        nonlocal last_atomic_update_padding
        if atomic_mode is not True or processed_entries_queue or current_batch:
            return
        update = atomic_group_buffer.compose_update()
        if update is None:
            return
        update_entries, update_model_version = materialize_atomic_update(
            update,
            tokenizer,
        )
        processed_entries_queue.extend(update_entries)
        assert len(processed_entries_queue) == samples_per_step
        max_model_version = update_model_version
        last_atomic_update_groups = update.n_groups
        last_atomic_update_real_entries = len(update.entries)
        last_atomic_update_padding = update.padding
        stats_aggregator.update([len(entry["input_ids"]) for entry in update.entries])
        logger.info(
            "Composed atomic update with %d groups, %d real entries, and %d sentinel slots",
            update.n_groups,
            len(update.entries),
            update.padding,
        )

    with write_to_streams(output_stream) as data_writer, write_to_streams(stats_streams) as stats_writer:
        with SharedMemoryManager() as smm:
            # Create shared memory queues without the manager parameter
            input_queue = SharedMemoryQueue(smm, cfg.preprocess.input_queue_size, cfg.preprocess.shared_memory_entry_size)
            output_queue = SharedMemoryQueue(smm, cfg.preprocess.output_queue_size, cfg.preprocess.shared_memory_entry_size)
            logger.info(f"Input queue size: {input_queue.get_memory_size() / 2**30} Gb")
            logger.info(f"Output queue size: {output_queue.get_memory_size() / 2**30} Gb")
            logger.info(f"Start {worker_pool_size} workers for preprocessing")
            
            # List to keep track of worker processes
            workers = []
            
            # Start worker processes
            for worker_idx in range(worker_pool_size):
                worker_llm = _get_worker_ref_llm(worker_idx, llms)
                worker_ref_source = worker_llm.base_url if worker_llm is not None else "rollout logprobs fallback"
                logger.info(f"Starting preprocessing worker {worker_idx} with reference source: {worker_ref_source}")
                worker = Process(
                    target=process_chunk,
                    args=(
                        worker_llm,
                        tokenizer,
                        cfg.finetune.seq_length,
                        rl_config,
                        input_queue,
                        output_queue,
                    )
                )
                worker.start()
                workers.append(worker)
            
            try:
                start_processing = time.time()
                fetching_took = 0
                writing_took = 0
                num_filtered_out = 0
                while True:
                    if (
                        trainer_state.samples_processed is not None
                        and trainer_state.samples_processed >= samples_target
                    ):
                        logger.info("Trainer signalled completion; stopping preprocessor loop")
                        break
                    if not input_queue.full():
                        try:
                            raw_chunk = raw_chunk_queue.get(timeout=0.001)
                            if isinstance(raw_chunk, Exception):
                                raise raw_chunk
                            raw_is_atomic = isinstance(raw_chunk, TrainingGroupEnvelope)
                            if atomic_mode is None:
                                atomic_mode = raw_is_atomic
                                if atomic_mode:
                                    processed_entries_queue = deque()
                            elif atomic_mode != raw_is_atomic:
                                raise ValueError(
                                    "Cannot mix atomic envelopes and legacy chunks in one preprocessor"
                                )
                            if raw_is_atomic:
                                if put_atomic_input_queue(
                                    input_queue,
                                    raw_chunk,
                                    atomic_rejection_counts,
                                ):
                                    submitted_chunks += 1
                            else:
                                input_queue.put(raw_chunk)
                                submitted_chunks += 1
                        except Empty:
                            pass

                    dataset = None
                    try:
                        start_fetching = time.time()
                        dataset = output_queue.get(timeout=0.001)
                        if isinstance(dataset, Exception):
                            raise dataset
                        if (
                            isinstance(dataset, dict)
                            and dataset.get("kind") == "atomic_group_rejection"
                        ):
                            count_atomic_group_rejection(
                                dataset,
                                atomic_rejection_counts,
                            )
                            logger.warning(
                                "Dropping atomic group %s at %s: %s (%s)",
                                dataset["group_id"],
                                dataset["queue_hop"],
                                dataset["reason"],
                                dataset,
                            )
                            dataset = None
                        elif isinstance(dataset, TrainingGroupEnvelope):
                            if rl_config.filter_zero_advantage_groups:
                                entries, num_filtered_out = filter_zero_advantage_groups(
                                    dataset.entries
                                )
                                total_filtered_out += num_filtered_out
                                dataset = (
                                    dataset.model_copy(update={"entries": entries})
                                    if entries
                                    else None
                                )
                                if num_filtered_out > 0:
                                    logger.info(
                                        f"Filtered out {num_filtered_out} samples from "
                                        "groups with zero advantage."
                                    )
                        elif rl_config.filter_zero_advantage_groups:
                            dataset, num_filtered_out = filter_zero_advantage_groups(dataset)
                            total_filtered_out += num_filtered_out
                            if num_filtered_out > 0:
                                logger.info(f"Filtered out {num_filtered_out} samples from groups with zero advantage.")
                        fetching_took += time.time() - start_fetching
                    except Empty:
                        pass
                    
                    if dataset:
                        if isinstance(dataset, dict) and "error" in dataset:
                            logger.error(f"Got exception from the result queue: {dataset['error']}")
                            logger.error(f"Traceback: {dataset['traceback']}")
                            raise Exception(dataset['error'])
                        if isinstance(dataset, TrainingGroupEnvelope):
                            atomic_pending_envelopes.append(dataset)
                        else:
                            for entry in dataset:
                                buffer.append(entry)
                        processed_chunks += 1

                    buffered_samples = (
                        sum(
                            len(envelope.entries)
                            for envelope in atomic_pending_envelopes
                        )
                        if atomic_mode
                        else len(buffer)
                    )
                    if buffered_samples < cfg.preprocess.dataset_buffer_size:
                        continue
                    if cfg.preprocess.dataset_buffer_size:
                        logger.info(
                            f"Buffer is full with {buffered_samples} samples, start writing"
                        )

                    if atomic_mode:
                        prepare_atomic_update()
                        while atomic_pending_envelopes:
                            envelope = atomic_pending_envelopes[0]
                            if not atomic_group_buffer.enqueue(envelope, pop_old_data):
                                break
                            atomic_pending_envelopes.popleft()
                        prepare_atomic_update()
                    else:
                        while len(buffer) > 0:
                            if len(processed_entries_queue) == processed_entries_queue.maxlen:
                                if not pop_old_data:
                                    break
                                else:
                                    processed_entries_queue_popped_data += 1
                                    if processed_entries_queue_popped_data % 100 == 0 and last_time_notice != processed_entries_queue_popped_data // 100:
                                        logger.warning(f"Popped {processed_entries_queue_popped_data} old entries from processed entries queue")
                                        last_time_notice = processed_entries_queue_popped_data // 100
                            entry = buffer.popleft()
                            processed_entries_queue.append(entry) # drop from the left if full

                            stats_aggregator.update([len(entry["input_ids"]) for entry in processed_entries_queue])
                            max_model_version = max([entry["model_version"] for entry in processed_entries_queue]) if processed_entries_queue else 0
                    
                    max_unconsumed_samples = cfg.preprocess.max_ready_samples_per_lead * num_trainers

                    assert isinstance(trainer_state.samples_processed, int)
                    if published_samples - trainer_state.samples_processed > max_unconsumed_samples:
                        # wait for the finetune loop to finish processing data
                        continue

                    batch_done = False
                    start_writing = time.time()
                    while (len(processed_entries_queue) > 0 and not batch_done) or (cfg.preprocess.dataset_buffer_size and not batch_done):
                        logger.debug(f"[inner loop] trainer {trainer_id} has {samples_per_trainer[trainer_id]} samples, target is {target_samples_per_lead}")
                        if cfg.finetune.seq_packing:
                            if samples_per_trainer[trainer_id] == target_samples_per_lead:
                                logger.debug(f"[inner loop] trainer {trainer_id} has all {target_samples_per_lead} samples, creating sentinel batch")
                                sentinel_batch = create_sentinel_batch(
                                    device=None,
                                    tokenizer=tokenizer,
                                    model_version=max_model_version
                                )
                                write_micro_batch_slices(trainer_id, data_writer, sentinel_batch, cfg.finetune.seq_parallel)
                                trainer_id = (trainer_id + cfg.finetune.seq_parallel) % num_trainers
                            else:
                                
                                while len(processed_entries_queue) > 0:
                                    entry = processed_entries_queue[0]  # Peek at next entry
                                    sample_length = len(entry["input_ids"])

                                    if current_length + sample_length > cfg.finetune.seq_length:
                                        time_to_write = True
                                        break  # Current micro batch is full
                                    
                                    # Add sample to current micro batch
                                    current_batch.append(processed_entries_queue.popleft())
                                    current_length += sample_length
                                    
                                    # Check if we've reached the sample limit per step
                                    if len(current_batch) + samples_per_trainer[trainer_id] == target_samples_per_lead:
                                        time_to_write = True
                                        break
                            
                                if time_to_write:
                                    assert len(current_batch) > 0, "Current batch should not be empty when writing"
                                    batch_encoding = collate_packed(current_batch, tokenizer, cfg.finetune.seq_parallel)
                                    write_micro_batch_slices(trainer_id, data_writer, batch_encoding, cfg.finetune.seq_parallel)
                                    published_samples += len(current_batch)
                                    samples_per_trainer[trainer_id] += len(current_batch)
                                    # Reset batch state for this trainer
                                    trainer_id = (trainer_id + cfg.finetune.seq_parallel) % num_trainers
                                    time_to_write = False
                                    current_batch = []
                                    current_length = 0
                                    logger.debug(f"[inner loop] Packed microbatch with {len(current_batch)} samples for trainer {trainer_id}")
                        else:
                            batch_entries = []
                            for _ in range(cfg.finetune.train_batch_size ):
                                batch_entries.append(processed_entries_queue.popleft())
                            batch_encoding = collate(batch_entries, tokenizer=tokenizer)
                            write_micro_batch_slices(trainer_id, data_writer, batch_encoding, cfg.finetune.seq_parallel)
                            published_samples += len(batch_entries)
                            samples_per_trainer[trainer_id] += len(batch_entries)
                            logger.debug(f"[inner loop] Packed microbatch with {len(batch_entries)} samples for trainer {trainer_id}")
                            trainer_id = (trainer_id + cfg.finetune.seq_parallel) % num_trainers

                        batch_done = published_samples == batch_boundary and trainer_id == 0
                        if batch_done:
                            batch_boundary += train_batch_size
                            target_samples_per_lead += samples_per_lead_per_step
                            if cfg.preprocess.dataset_buffer_size and len(processed_entries_queue) > 0:
                                # There is enough data in the processed entries queue to write multiple batches
                                batch_done = False
                                
                        logger.debug(
                            f"[inner loop] wrote {published_samples} samples, "
                            f"trainer {trainer_id} is at {samples_per_trainer[trainer_id]} samples, "
                            f"batch done: {batch_done}"
                        )
                    writing_took += time.time() - start_writing
                            
                    if (
                        published_samples > last_published_samples 
                        and (cfg.debug.mode or batch_done or (published_samples - last_published_samples > cfg.preprocess.log_every_n_samples))
                    ):
                        queued_entry_samples = (
                            cfg.attempts
                            if atomic_mode
                            else cfg.preprocess.chunk_n_groups * cfg.attempts
                        )
                        samples_in_output_queue = (
                            output_queue.qsize() * queued_entry_samples
                        )
                        stats = {
                            "preprocessor/published_samples": published_samples,
                            "preprocessor/published_model_version": max_model_version,
                            "preprocessor/queue/raw_samples": (
                                raw_chunk_queue.qsize() * queued_entry_samples
                            ),
                            "preprocessor/queue/raw": raw_chunk_queue.qsize(),
                            "preprocessor/queue/output_samples": samples_in_output_queue,
                            "preprocessor/queue/output": output_queue.qsize(),
                            "preprocessor/filtered_out_samples": num_filtered_out,
                            "preprocessor/total_filtered_out_samples": total_filtered_out,
                        }
                        if atomic_mode:
                            stats.update(
                                {
                                    "preprocessor/atomic/ready_groups": len(
                                        atomic_group_buffer.ready_groups
                                    ),
                                    "preprocessor/atomic/ready_entries": (
                                        atomic_group_buffer.ready_entries
                                    ),
                                    "preprocessor/atomic/staged_groups": len(
                                        atomic_group_buffer.staged_groups
                                    ),
                                    "preprocessor/atomic/staged_entries": (
                                        atomic_group_buffer.staged_entries
                                    ),
                                    "preprocessor/atomic/pending_groups": len(
                                        atomic_pending_envelopes
                                    ),
                                    "preprocessor/atomic/pending_entries": sum(
                                        len(envelope.entries)
                                        for envelope in atomic_pending_envelopes
                                    ),
                                    "preprocessor/atomic/evicted_groups": (
                                        atomic_group_buffer.evicted_groups
                                    ),
                                    "preprocessor/atomic/evicted_entries": (
                                        atomic_group_buffer.evicted_entries
                                    ),
                                    "preprocessor/atomic/rejected_groups": (
                                        atomic_group_buffer.rejected_groups
                                    ),
                                    "preprocessor/atomic/rejected_entries": (
                                        atomic_group_buffer.rejected_entries
                                    ),
                                    "preprocessor/atomic/updates": (
                                        atomic_group_buffer.updates
                                    ),
                                    "preprocessor/atomic/padding": (
                                        atomic_group_buffer.padding
                                    ),
                                    "preprocessor/atomic/last_update_groups": (
                                        last_atomic_update_groups
                                    ),
                                    "preprocessor/atomic/last_update_real_entries": (
                                        last_atomic_update_real_entries
                                    ),
                                    "preprocessor/atomic/last_update_padding": (
                                        last_atomic_update_padding
                                    ),
                                }
                            )
                            for reason, count in atomic_rejection_counts.items():
                                stats[
                                    f"preprocessor/atomic_group_drop/{reason}"
                                ] = count
                        if stats_aggregator.has_enough_data():
                            stats.update({"preprocessor/" + k: v for k, v in stats_aggregator.get_stats().items()})
                        if wandb_run is not None:
                            wandb_run.log(stats)
                        stats_writer.write(stats)
                        
                        processing_took = time.time() - start_processing
                        processed_samples = published_samples - last_published_samples
                        last_published_samples = published_samples
                        logger.info(
                            f"Processed {processed_samples} samples (filtered out {num_filtered_out}) in {processing_took:.3f}s"
                            f" (fetching took {fetching_took:.3f} and writing took {writing_took:.3f})"
                            f" and wrote to {output_stream}, total {published_samples} samples so far,"
                            f" {samples_in_output_queue} samples in output queue, max output queue entry size {output_queue.max_actual_entry_size()} bytes"
                        )
                        start_processing = time.time()
                        fetching_took = 0
                        writing_took = 0
                        num_filtered_out = 0
            finally:
                # Clean up worker processes
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=1.0)
