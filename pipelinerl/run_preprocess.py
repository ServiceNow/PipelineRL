from collections import defaultdict
import os

os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

import logging
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from queue import Empty, Queue
from typing import List
import random

import datasets
import transformers
from litellm import BaseModel, Field

from pipelinerl.finetune.logging_ import flatten_dict_config
from pipelinerl.shared_memory_array import SharedMemoryArray
from pipelinerl.utils import wait_for_inference_servers, init_wandb
from pipelinerl.world import WorldMap

datasets.disable_caching()
from datasets.arrow_dataset import Dataset
from datasets.fingerprint import Hasher
from omegaconf import DictConfig
from tapeagents.llms import TrainableLLM

from pipelinerl.finetune.checkpoints import (
    load_tokenizer,
)
from pipelinerl.finetune.data import preprocess_fn
from pipelinerl.finetune.rl import RL_DATA_COLUMNS, RLConfig, populate_rl_data
import traceback
from pipelinerl.streams import (
    SingleStreamSpec,
    StreamRangeSpec,
    read_stream,
    set_streams_backend,
    write_to_streams,
)

logger = logging.getLogger(__name__)


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
    for trace in traces:
        prompt_token_ids.append(trace["input_ids"][: -len(trace["logprobs"])])
        completion_token_ids.append(trace["input_ids"][-len(trace["logprobs"]) :])
    try:
        all_ref_logprobs = llm.get_batch_logprobs_token_ids(prompt_token_ids, completion_token_ids)
    except Exception as e:
        logger.error(f"Failed to get ref logprobs: {e}")
        assert (response := getattr(e, "response", None))
        logger.error(f"Response content: {response.text}")
        raise e
    for trace, ref_logprobs in zip(traces, all_ref_logprobs):
        trace["ref_logprobs"] = [c["logprob"] for c in ref_logprobs["content"]]
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
) -> Dataset:
    preprocess = partial(preprocess_fn, seq_length=seq_length, tokenizer=tokenizer, is_rl=True)
    columns = ["input_ids", "labels", "attention_mask", "group_id"] + RL_DATA_COLUMNS
    
    logger.debug(f"Instantiated preprocess function hash {Hasher.hash(preprocess)}")

    data = replace_oov_tokens_with_the(data, tokenizer)

    # inplace update of the traces with ref logprobs
    if llm is not None:
        batch_annotate_traces_with_ref_logprobs(llm, data)
    else:
        for entry in data:
            entry["ref_logprobs"] = entry["logprobs"]

    dataset = Dataset.from_list(data)    
    dataset = dataset.map(preprocess, keep_in_memory=True, load_from_cache_file=False)
    dataset = dataset.with_format(columns=columns)
    if not isinstance(tokenizer.eos_token_id, int):
        raise ValueError(f"Tokenizer {tokenizer} does not have an eos_token_id")
    dataset = dataset.add_column("model_version", [entry["metadata"]["model_version"] for entry in data])  # type: ignore
    dataset = dataset.add_column("rollout_index", [entry["metadata"]["rollout_index"] for entry in data])  # type: ignore    
    dataset = dataset.add_column("step_index", [entry["metadata"]["step_index"] for entry in data])  # type: ignore
    dataset = populate_rl_data(dataset=dataset, eos_token_id=tokenizer.eos_token_id, config=rl_config)
    return dataset


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
                for group in reader.read():
                    buffer.extend(group)
                    n_groups += 1
                    if n_groups == chunk_n_groups:
                        break
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
    llm: TrainableLLM,
    io_buffer,
    slot: int,
    tokenizer: transformers.PreTrainedTokenizerBase,
    seq_length: int,
    rl_config: RLConfig,
    dataset_queue: Queue,
):
    try:
        chunk = io_buffer[slot]
        dataset = preprocess_dataset(
            llm=llm,
            data=chunk,
            tokenizer=tokenizer,
            seq_length=seq_length,
            rl_config=rl_config,
        )
        io_buffer[slot] = dataset
        dataset_queue.put(slot)
    except Exception as e:
        logger.error(f"Failed to preprocess chunk: {e}")
        e.__traceback_str = traceback.format_exc()
        io_buffer[slot] = e
        dataset_queue.put(slot)


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

def run_preprocessing_loop(
    cfg: DictConfig,
):
    set_streams_backend(**cfg.streams)

    world_map = WorldMap(cfg, verbose=True)
    exp_root_dir = Path(cfg.output_dir)

    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_root_dir / "preprocessor", flatten_dict_config(cfg))
        if run is None:
            raise ValueError("Failed to initialize wandb run")

    tokenizer = load_tokenizer(cfg.finetune.config_name)

    llm_urls = str(cfg.me.llm_urls).split("+") if cfg.me.llm_urls else []
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

    raw_chunk_queue = Queue(cfg.preprocess.queue_size)
    rl_config = RLConfig(**cfg.finetune.rl)
    dataset_loader_worker_fn = partial(
        run_dataset_loader,
        raw_chunk_queue=raw_chunk_queue,
        data_stream=input_stream,
        check_group_size=cfg.attempts,
        chunk_n_groups=cfg.preprocess.chunk_n_groups,
        pop_old_data=cfg.max_lag is None and cfg.pop_old_data and not cfg.debug.mode,
    )
    # Start the dataset loader thread using Thread
    dataset_loader_thread = threading.Thread(target=dataset_loader_worker_fn)
    dataset_loader_thread.start()

    published_samples = 0
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
    next_llm_index = 0

    stats_aggregator = SlidingWindowAggregator(window_size=max(10, 1000 // cfg.preprocess.chunk_n_groups))

    buffer = []
    total_filtered_out = 0  # Track total filtered samples across all batches
    with write_to_streams(output_stream) as writer, write_to_streams(stats_streams) as stats_writer:
        with mp.Manager() as manager, SharedMemoryManager() as smm:
            max_dataset_queue_size = 128
            max_pool_tasks = 2 * worker_pool_size
            buffer_size = 2 * max_pool_tasks + max_dataset_queue_size
            dataset_queue = manager.Queue(max_dataset_queue_size)
            io_buffer = SharedMemoryArray(smm, buffer_size, cfg.preprocess.shared_memory_entry_size)
            free_slots = set(range(buffer_size))
            logger.info(f"Shared memory buffer size: {io_buffer.get_memory_size() / 2**30} Gb")
            logger.info(f"Start {worker_pool_size} workers for preprocessing")
            with ProcessPoolExecutor(max_workers=worker_pool_size) as executor:
                while True:
                    llm = llms[next_llm_index] if llms else None
                    if submitted_chunks - processed_chunks < max_pool_tasks:
                        try:
                            raw_chunk = raw_chunk_queue.get(timeout=0.001)
                            if isinstance(raw_chunk, Exception):
                                raise raw_chunk
                            slot = free_slots.pop()
                            io_buffer[slot] = raw_chunk
                            future = executor.submit(
                                process_chunk,
                                llm,
                                io_buffer,
                                slot,
                                tokenizer,
                                cfg.finetune.seq_length,
                                rl_config,
                                dataset_queue,
                            )
                            future.add_done_callback(
                                lambda fut: logger.error(
                                    f"Exception while preprocessing: {fut.exception()}",
                                    exc_info=fut.exception(),
                                )
                                if fut.exception()
                                else None
                            )
                            submitted_chunks += 1
                            next_llm_index = (next_llm_index + 1) % len(llms) if llms else 0
                        except Empty:
                            pass

                    start_processing = time.time()
                    try:
                        # Try to write the next dataset to the output stream, if it is ready
                        slot = dataset_queue.get(timeout=0.001)
                        dataset = io_buffer[slot]
                        free_slots.add(slot)
                        fetching_took = time.time() - start_processing
                    except Empty:
                        continue
                    if isinstance(dataset, Exception):
                        logger.error(f"Got exception from the result queue")
                        logger.error(dataset.__traceback_str)
                        raise dataset
                    start_writing = time.time()
                    buffer += dataset
                    processed_chunks += 1

                    if len(buffer) < cfg.preprocess.buffer_size:
                        continue
                    if cfg.preprocess.buffer_size:
                        # If buffer size is not set, no point in logging
                        logger.info(f"Buffer is full with {len(buffer)} samples, start writing")
                        random.shuffle(buffer)

                    # Conditionally filter out groups where all advantages are zero
                    if rl_config.filter_zero_advantage_groups:
                        filtered_buffer, num_filtered_out = filter_zero_advantage_groups(buffer)
                        total_filtered_out += num_filtered_out
                        
                        if num_filtered_out > 0:
                            logger.info(f"Filtered out {num_filtered_out} samples from groups with zero advantage.")
                    else:
                        filtered_buffer = buffer
                        num_filtered_out = 0

                    # Write the entries (filtered or unfiltered based on config)
                    for entry in filtered_buffer:
                        writer.write(entry)
                    writing_took = time.time() - start_writing
                    stats_aggregator.update([len(entry["input_ids"]) for entry in filtered_buffer])
                    published_samples += len(filtered_buffer)  # Count only written samples
                    max_model_version = max([entry["model_version"] for entry in filtered_buffer]) if filtered_buffer else 0
                    samples_in_queue = dataset_queue.qsize() * cfg.preprocess.chunk_n_groups
                    stats = {
                        "preprocessor/published_samples": published_samples,
                        "preprocessor/published_model_version": max_model_version,
                        "preprocessor/queue/raw_samples": raw_chunk_queue.qsize() * cfg.preprocess.chunk_n_groups,
                        "preprocessor/queue/raw": raw_chunk_queue.qsize(),
                        "preprocessor/queue/dataset_samples": samples_in_queue,
                        "preprocessor/queue/dataset": dataset_queue.qsize(),
                        "preprocessor/filtered_out_samples": num_filtered_out,
                        "preprocessor/total_filtered_out_samples": total_filtered_out,
                    }
                    if stats_aggregator.has_enough_data():
                        stats.update({"preprocessor/" + k: v for k, v in stats_aggregator.get_stats().items()})
                    run.log(stats)
                    stats_writer.write(stats)
                    processing_took = time.time() - start_processing
                    logger.info(
                        f"Processed {len(filtered_buffer)} samples (filtered out {num_filtered_out}) in {processing_took:.3f}s"
                        f" (last fetching took {fetching_took:.3f}, all writing took {writing_took:.3f})"
                        f" and wrote to {output_stream}, total {published_samples} samples so far,"
                        f" {samples_in_queue} samples in queue, max buffer entry size {io_buffer._max_written_entry_size}"
                    )
                    buffer = []

