import asyncio
import logging
import math
import multiprocessing as mp
import os
import queue
from queue import Empty
import random
import time
from collections import defaultdict, deque
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import aiohttp
import hydra
import uvloop
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field
from tapeagents.llms import TrainableLLM
from typing import Dict, List, Optional

import wandb
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from pipelinerl.rollouts import RolloutResult, BaseMetrics
from pipelinerl.shared_memory_array import SharedMemoryQueue
from pipelinerl.state import TrainerState
from pipelinerl.streams import (
    SingleStreamSpec,
    StreamSpec,
    StreamWriter,
    set_streams_backend,
    write_to_streams,
)

from .utils import (
    always_or_never_success_stats,
    calculate_stats,
    setup_logging,
    wait_for_environments,
    wait_for_inference_servers,
)

logger = logging.getLogger(__name__)


class SlidingWindowData(BaseModel):
    prompt_tokens_window: list[list[int]] = Field(
        default_factory=list,
        description="Prompt token counts for each chunk in the window",
    )
    output_tokens_window: list[list[int]] = Field(
        default_factory=list,
        description="Output token counts for each chunk in the window",
    )
    timestamps: list[float] = Field(default_factory=list)


class SlidingWindowAggregator:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.data = SlidingWindowData()

    def update(self, prompt_tokens: list[int], output_tokens: list[int]):
        self.data.prompt_tokens_window.append(prompt_tokens)
        self.data.output_tokens_window.append(output_tokens)
        self.data.timestamps.append(time.time())
        if len(self.data.prompt_tokens_window) > self.window_size:
            self.data.prompt_tokens_window.pop(0)
            self.data.output_tokens_window.pop(0)
            self.data.timestamps.pop(0)

    def get_stats(self):
        if len(self.data.prompt_tokens_window) < self.window_size:
            return None

        # 1. How many samples do we produce per second?
        # 2. How many output tokens do we produce per second?
        # 3. How many prompt tokens do we produce per second?
        # 4. How many total tokens do we produce per second?
        null_stats = {
            "samples_per_second": 0,
            "output_tokens_per_second": 0,
            "prompt_tokens_per_second": 0,
            "total_tokens_per_second": 0,
        }
        if not self.data.timestamps:
            return null_stats

        time_span = self.data.timestamps[-1] - self.data.timestamps[0]
        if time_span < 1e-6:
            return null_stats

        num_samples = sum(len(tokens) for tokens in self.data.prompt_tokens_window)
        total_output_tokens = sum(sum(tokens) for tokens in self.data.output_tokens_window)
        total_prompt_tokens = sum(sum(tokens) for tokens in self.data.prompt_tokens_window)

        return {
            "samples_per_second": num_samples / time_span,
            "output_tokens_per_second": total_output_tokens / time_span,
            "prompt_tokens_per_second": total_prompt_tokens / time_span,
            "total_tokens_per_second": (total_output_tokens + total_prompt_tokens) / time_span,
        }



class CurriculumSuccessTracker:
    def __init__(self) -> None:
        self._buffers: dict[str, deque[int]] = {}
        self._max_windows: dict[str, int] = {}
        self._total_counts: defaultdict[str, int] = defaultdict(int)

    def ensure_window(self, dataset: str, window: int) -> None:
        if window <= 0:
            window = 1
        current = self._max_windows.get(dataset, 0)
        if window <= current:
            return
        existing = self._buffers.get(dataset, deque(maxlen=window))
        if existing.maxlen != window:
            new_buffer = deque(existing, maxlen=window)
        else:
            new_buffer = existing
        self._buffers[dataset] = new_buffer
        self._max_windows[dataset] = window

    def update(self, dataset: str, success_values: list[int | bool]) -> None:
        if not success_values:
            return
        buffer = self._buffers.get(dataset)
        if buffer is None:
            maxlen = self._max_windows.get(dataset, max(1, len(success_values)))
            buffer = deque(maxlen=maxlen)
            self._buffers[dataset] = buffer
            self._max_windows[dataset] = maxlen
        for value in success_values:
            buffer.append(1 if bool(value) else 0)
            self._total_counts[dataset] += 1

    def success_mean(self, dataset: str, window: Optional[int] = None) -> Optional[float]:
        buffer = self._buffers.get(dataset)
        if buffer is None or not buffer:
            return None
        if window is None or window <= 0 or window >= len(buffer):
            values = list(buffer)
        else:
            if len(buffer) < window:
                return None
            values = list(buffer)[-window:]
        if not values:
            return None
        return sum(values) / len(values)

    def total_samples(self, dataset: str) -> int:
        return self._total_counts.get(dataset, 0)



def make_stats_dict() -> dict:
    return defaultdict(lambda: defaultdict(list))


def parse_curriculum_schedule(curriculum_cfg) -> list[dict]:
    raw_schedule = curriculum_cfg.get("schedule", [])
    if not raw_schedule:
        return [{"step": 0, "hard_weight": 0.0, "thresholds": []}]
    if not isinstance(raw_schedule, list):
        raw_schedule = [raw_schedule]
    parsed_schedule: list[dict] = []
    for entry in raw_schedule:
        step = int(entry.get("step", 0))
        hard_weight = float(entry.get("hard_weight", 0.0))
        hard_weight = max(0.0, min(1.0, hard_weight))
        thresholds_cfg = entry.get("success_thresholds", []) or []
        if not isinstance(thresholds_cfg, list):
            thresholds_cfg = [thresholds_cfg]
        thresholds: list[dict] = []
        for threshold_entry in thresholds_cfg:
            dataset = threshold_entry.get("dataset")
            if not dataset:
                continue
            threshold_value = float(threshold_entry.get("threshold", 1.0))
            window = int(threshold_entry.get("window", threshold_entry.get("window_size", 0) or 1))
            if window <= 0:
                window = 1
            min_samples_value = threshold_entry.get("min_samples")
            min_samples = int(min_samples_value) if min_samples_value is not None else None
            thresholds.append(
                {
                    "dataset": dataset,
                    "threshold": threshold_value,
                    "window": window,
                    "min_samples": min_samples,
                }
            )
        parsed_schedule.append({"step": step, "hard_weight": hard_weight, "thresholds": thresholds})
    parsed_schedule.sort(key=lambda item: item["step"])
    return parsed_schedule


async def schedule_rollouts(
    cfg: DictConfig,
    attempts: int,
    problem_queue: SharedMemoryQueue,
    result_queue: SharedMemoryQueue,
    trainer_state: TrainerState,
    llms: list[TrainableLLM],
    scheduler_name: str,
):
    """This courotuine does the following.

    - It run asyncio loop for doing many rollouts in parallel using llm_async_generate
    - For each problem it does exactly `attempts` rollouts (let's call this a group)
    - It keeps track of how many rollout coroutines are running for each llms
    - it uses the LLM that has the least number of running coroutines for each new rollout
    - when all LLMs are busy it does nothing
    - It keeps track of how many rollouts are done for each group
    - When the group is done it puts the result in the result queue
    """
    loop = asyncio.get_running_loop()

    # Track active tasks per LLM
    active_rollouts = [0] * len(llms)
    started_rollouts = 0
    finished_rollouts = 0
    # Track rollouts per problem group
    group_rollouts = {}
    rollout_policy = hydra.utils.get_method(cfg.actor.rollout_policy)
    logger.info(f"Use rollout policy: {rollout_policy}")

    async def rollout_and_maybe_produce_result(
        problem: dict,
        group_id: int,
        rollout_index: int,
        llm_index: int,
        session: aiohttp.ClientSession,
    ):
        nonlocal started_rollouts, finished_rollouts
        try:
            llm = llms[llm_index]
            model_version = trainer_state.propagated_weight_version
            assert model_version is not None
            rollout_result = await rollout_policy(cfg, llm, problem, session)
            rollout_result.model_version = model_version
            # Make a group id that will be different from groups made by another rollout maker
            full_group_id = f"{scheduler_name}_{group_id}"
            rollout_result.group_id = full_group_id
            for step_index, sample in enumerate(rollout_result.training_texts):
                # Downstream in the pipeline we'll need these fields in every sample
                sample.metadata["model_version"] = model_version
                sample.metadata["rollout_index"] = rollout_index
                sample.metadata["step_index"] = step_index
                sample.group_id = full_group_id
            group_rollouts[group_id].append(rollout_result)
            if len(group_rollouts[group_id]) == attempts:
                # This is blocking call, but there's just one other thread reading from this queue.
                random.shuffle(group_rollouts[group_id])
                result_queue.put(group_rollouts[group_id])
                del group_rollouts[group_id]
            finished_rollouts += 1
        except Exception as e:
            # Cancel all tasks except the current one
            logger.error("Exception in rollout, stop all other rollout tasks", exc_info=e)
            current_task = asyncio.current_task(loop=loop)
            for task in asyncio.all_tasks(loop=loop):
                if task != current_task:
                    task.cancel()
            result_queue.put(e)
            logger.error("Stopped all tasks and put exception in the result queue")
        finally:
            active_rollouts[llm_index] -= 1

    group_id = -1
    group_rollout_index = attempts
    problem = None

    last_logged = time.time()
    logger.info("Starting rollout scheduler")
    connector = aiohttp.TCPConnector(limit=50000, limit_per_host=50000, keepalive_timeout=1.0)
    timeout = aiohttp.ClientTimeout(total=10*3600.0, connect=10*3600.0, sock_read=10*3600.0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while True:
            if time.time() - last_logged > 1.0 and sum(active_rollouts):
                logger.info(
                    f"{scheduler_name}: "
                    f"rollouts in progress: {sum(active_rollouts)}, "
                    f"groups in progress: {len(group_rollouts)}, "
                    f"rollouts started so far: {started_rollouts}, "
                    f"rollouts finished so far: {finished_rollouts}, "
                    f"max group size in bytes: {result_queue.max_actual_entry_size()}, "
                )
                last_logged = time.time()

            if group_rollout_index == attempts:
                try:
                    problem = problem_queue.get(block=False)
                except Empty:
                    # give some quality time for other couroutines to work
                    await asyncio.sleep(0.01)
                    continue
                group_id += 1
                group_rollouts[group_id] = []
                group_rollout_index = 0

            next_llm = active_rollouts.index(min(active_rollouts))
            if active_rollouts[next_llm] == cfg.actor.llm_max_rollouts:
                # all llms are busy, wait for one to finish
                await asyncio.sleep(0.01)
                continue
            active_rollouts[next_llm] += 1
            started_rollouts += 1
            assert problem is not None
            loop.create_task(
                rollout_and_maybe_produce_result(
                    problem=problem,
                    group_id=group_id,
                    rollout_index=group_rollout_index,
                    llm_index=next_llm,
                    session=session,
                )
            )
            group_rollout_index += 1
    logger.info("Rollout scheduler finished")


def rollout_maker_entrypoint(
    cfg: DictConfig,
    attempts: int,
    problem_queue: SharedMemoryQueue,
    result_queue: SharedMemoryQueue,
    llms: list[TrainableLLM],
    scheduler_name: str,
):
    trainer_state = TrainerState(Path(cfg.output_dir))
    if cfg.debug.mode:
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        schedule_rollouts(cfg, attempts, problem_queue, result_queue, trainer_state, llms, scheduler_name)
    )
    loop.close()
    logger.info("Rollout maker loop closed")


def random_iter(problems: list):
    while True:
        yield random.sample(problems, 1)[0]


def curriculum_iter(
    problems: list,
    trainer_state: TrainerState,
    curriculum_cfg: DictConfig,
    logger: logging.Logger | None = None,
    success_tracker: CurriculumSuccessTracker | None = None,
    stage_state: Optional[dict] = None,
    parsed_schedule: Optional[list[dict]] = None,
):
    curriculum_obj = (
        OmegaConf.to_container(curriculum_cfg, resolve=True)
        if isinstance(curriculum_cfg, DictConfig)
        else curriculum_cfg
    )
    base_names = set(curriculum_obj.get("base_datasets", []))
    hard_names = set(curriculum_obj.get("hard_datasets", []))
    if hard_names and not base_names:
        base_names = {problem.get("dataset") for problem in problems if problem.get("dataset") not in hard_names}

    base_pool = [
        problem
        for problem in problems
        if (problem.get("dataset") in base_names) or (not base_names and problem.get("dataset") not in hard_names)
    ]
    hard_pool = [problem for problem in problems if problem.get("dataset") in hard_names]

    if not hard_pool:
        if logger:
            logger.warning(
                "Curriculum enabled but no problems matched hard_datasets list; falling back to base sampling"
            )
        yield from random_iter(problems)
        return

    if not base_pool:
        if logger:
            logger.warning("Curriculum enabled but base pool is empty; sampling exclusively from hard dataset")
        base_pool = hard_pool

    schedule = parsed_schedule or parse_curriculum_schedule(curriculum_obj)
    if success_tracker:
        for stage in schedule:
            for threshold in stage["thresholds"]:
                success_tracker.ensure_window(threshold["dataset"], threshold["window"])

    def stage_ready(stage_cfg: dict) -> tuple[bool, list[str]]:
        if not stage_cfg["thresholds"] or success_tracker is None:
            return True, []
        blockers: list[str] = []
        for threshold in stage_cfg["thresholds"]:
            dataset = threshold["dataset"]
            threshold_value = threshold["threshold"]
            window = threshold["window"]
            min_samples = threshold.get("min_samples")
            if min_samples is not None:
                total_samples = success_tracker.total_samples(dataset)
                if total_samples < min_samples:
                    blockers.append(
                        f"{dataset}: waiting for {min_samples} samples (have {total_samples})"
                    )
                    continue
            success_mean_value = success_tracker.success_mean(dataset, window)
            if success_mean_value is None:
                blockers.append(f"{dataset}: insufficient window data (need {window})")
                continue
            if success_mean_value < threshold_value:
                blockers.append(
                    f"{dataset}: success_mean {success_mean_value:.3f} < {threshold_value:.3f} (window={window})"
                )
        return (len(blockers) == 0), blockers

    current_stage = -1
    last_block_log: tuple[int, tuple[str, ...]] | None = None
    if stage_state is None:
        stage_state = {"index": 0}

    while True:
        samples_processed = trainer_state.samples_processed or 0
        desired_stage_index = 0
        hard_weight = schedule[0]["hard_weight"]

        for idx, stage_cfg in enumerate(schedule):
            step = stage_cfg["step"]
            if samples_processed >= step:
                desired_stage_index = idx
                hard_weight = stage_cfg["hard_weight"]
            else:
                break

        stage_index = desired_stage_index
        blocker_messages: list[str] = []
        while stage_index >= 0:
            ready, blockers = stage_ready(schedule[stage_index])
            if ready:
                blocker_messages = []
                break
            blocker_messages = blockers
            stage_index -= 1

        if stage_index < 0:
            stage_index = 0
            hard_weight = schedule[0]["hard_weight"]
        else:
            hard_weight = schedule[stage_index]["hard_weight"]

        if logger and desired_stage_index != stage_index and blocker_messages:
            block_signature = (desired_stage_index, tuple(blocker_messages))
            if block_signature != last_block_log:
                logger.info(
                    "Curriculum stage %d gated by: %s",
                    desired_stage_index,
                    "; ".join(blocker_messages),
                )
                last_block_log = block_signature

        if logger and stage_index != current_stage:
            logger.info(
                "Curriculum stage %d active (samples_processed=%d, hard_weight=%.3f)",
                stage_index,
                samples_processed,
                hard_weight,
            )
            current_stage = stage_index

        stage_state["index"] = stage_index

        if hard_pool and random.random() < hard_weight:
            yield random.choice(hard_pool)
        else:
            yield random.choice(base_pool)


def sequential_iter(problems: list):
    for problem in problems:
        yield problem


class ActorLoop:
    def __init__(
        self,
        cfg: DictConfig,
        llms: list[TrainableLLM],
        data_stream: StreamSpec,
        stats_stream: StreamSpec,
        trainer_state: TrainerState,
        is_training: bool = True,
    ) -> None:
        self.data_stream = data_stream
        self.trainer_state = trainer_state
        self.stats_stream = stats_stream
        self.sliding_aggregator = SlidingWindowAggregator(window_size=cfg.actor.throughput_window_size)
        self.llms = llms
        self.loop_start_time = -1
        self.cfg = cfg
        self.is_training = is_training
        self.is_scheduling_paused = False
        self.debug_mode = bool(cfg.debug.mode)
        self.curriculum_tracker: CurriculumSuccessTracker | None = None
        self.curriculum_stage_state: dict | None = None

        # Determine the number of processes to use
        num_processes = min(self.cfg.actor.rollout_workers, len(self.llms))
        attempts = self.cfg.attempts if is_training else 1

        # Divide LLMs approximately equally across processes
        llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.llms):
            llm_groups[i % num_processes].append((i, llm))

        self.smm = SharedMemoryManager()
        self.smm.start()

        
        # Use SharedMemoryQueue instead of separate problem_queue, result_queue, and io_buffer
        self.problem_queue = SharedMemoryQueue(self.smm, self.cfg.actor.problem_queue_size, cfg.actor.shared_memory_entry_size)
        self.result_queue = SharedMemoryQueue(self.smm, self.cfg.actor.result_queue_size, cfg.actor.shared_memory_entry_size)
        
        logger.info(f"Initialized {'train' if self.is_training else 'test'} actor loop")
        logger.info(f"Problem queue size: {self.problem_queue.max_size}, result queue size: {self.result_queue.max_size}")
        logger.info(f"Result queue buffer size: {self.result_queue.get_memory_size() / 2**30} Gb")

        # Create and start multiple rollout processes
        self.rollout_processes = []
        for llm_group in llm_groups:
            assert llm_group
            llm_idxs = [llm[0] for llm in llm_group]
            llms = [llm[1] for llm in llm_group]
            scheduler_name = (
                f"{'train' if is_training else 'test'} scheduler for llms {','.join([str(i) for i in llm_idxs])}"
            )
            process = mp.Process(
                target=rollout_maker_entrypoint,
                args=(self.cfg, attempts, self.problem_queue, self.result_queue, llms, scheduler_name),
            )
            process.start()
            self.rollout_processes.append(process)

    def init_stats(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.latency_list = []
        self.model_versions_list = []
        self.sliding_stats = defaultdict(list)
    
    def compute_domain_agnostic_metrics(self, result: RolloutResult) -> Dict[str, float]:
        metrics = {}
        
        metrics['overflow'] = all([not training_text.finished for training_text in result.training_texts ])
        metrics['num_turns'] = len(result.training_texts)
        metrics['prompt_tokens'] = [training_text.prompt_tokens for training_text in result.training_texts]
        metrics['output_tokens'] = [training_text.output_tokens for training_text in result.training_texts]
        
        return metrics

    def update_stats(self, rollout_results: List[RolloutResult]):
        for result in rollout_results:
            assert result.model_version is not None
            assert isinstance(result.metrics, BaseMetrics), "Metrics should be an instance of BaseMetrics"
            dataset_name = result.dataset_name
            group_id = result.group_id
            self.latency_list.append(result.latency)
            self.model_versions_list.append(result.model_version)
            domain_agnostic_metrics = self.compute_domain_agnostic_metrics(result) 
            all_metrics = result.metrics.model_dump() | domain_agnostic_metrics
            for k, v in all_metrics.items():
                if isinstance(v, list):
                    self.stats[k][dataset_name][group_id] += v
                    if k == "success" and self.curriculum_tracker:
                        self.curriculum_tracker.update(dataset_name, v)
                elif isinstance(v, float) | isinstance(v, bool) | isinstance(v, int):
                    self.stats[k][dataset_name][group_id].append(v)
                    if k == "success" and self.curriculum_tracker:
                        self.curriculum_tracker.update(dataset_name, [v])
                else:
                    raise ValueError(f"Unsupported metric type: {type(v)} for key {k}")
        
        prompt_length_tokens = [training_text.prompt_tokens for result in rollout_results for training_text in result.training_texts]
        output_length_tokens = [training_text.output_tokens for result in rollout_results for training_text in result.training_texts]
        self.sliding_aggregator.update(prompt_length_tokens, output_length_tokens)
        sliding_window_stats = self.sliding_aggregator.get_stats()
        if sliding_window_stats is not None:
            for k, v in sliding_window_stats.items():
                self.sliding_stats[k].append(v)
        


    def run(self, dataset: list[tuple[str, dict]]):
        loop_start_time = time.time()
        self.init_stats()

        attempts = self.cfg.attempts if self.is_training else 1
        published_samples = 0
        submitted_groups = 0
        finished_groups = 0
        expected_rollouts = -1 if self.is_training else len(dataset)
        if expected_rollouts > 0:
            logger.info(f"Will stop after {expected_rollouts} rollouts")
        trainer_version_to_publish = None

        # If training, we expect to sample infinitely
        # for train sample, sample random batches infinitely
        # for test samples, loop through the dataset once
        if self.is_training:
            curriculum_cfg = getattr(self.cfg.actor, "curriculum", None)
            use_curriculum = bool(
                curriculum_cfg and getattr(curriculum_cfg, "enabled", False) and dataset
            )
            if use_curriculum:
                curriculum_obj = (
                    OmegaConf.to_container(curriculum_cfg, resolve=True)
                    if isinstance(curriculum_cfg, DictConfig)
                    else curriculum_cfg
                )
                parsed_schedule = parse_curriculum_schedule(curriculum_obj)
                self.curriculum_tracker = CurriculumSuccessTracker()
                for stage in parsed_schedule:
                    for threshold in stage["thresholds"]:
                        self.curriculum_tracker.ensure_window(threshold["dataset"], threshold["window"])
                self.curriculum_stage_state = {"index": 0}
                problem_iter = curriculum_iter(
                    dataset,
                    trainer_state=self.trainer_state,
                    curriculum_cfg=curriculum_cfg,
                    logger=logger,
                    success_tracker=self.curriculum_tracker,
                    stage_state=self.curriculum_stage_state,
                    parsed_schedule=parsed_schedule,
                )
            else:
                problem_iter = random_iter(dataset)
                self.curriculum_tracker = None
                self.curriculum_stage_state = None
        else:
            problem_iter = sequential_iter(dataset)
            self.curriculum_tracker = None
            self.curriculum_stage_state = None
        assert self.trainer_state.propagated_weight_version is not None

        last_trainer_version = self.trainer_state.propagated_weight_version
        max_lag = self.cfg.finetune.max_lag if self.is_training else None
        if max_lag is not None:
            total_batch_size = self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
            total_update_size = (
                math.ceil(self.cfg.finetune.weight_update_interval / total_batch_size) * total_batch_size
            )
            if total_batch_size % self.cfg.attempts != 0:
                logger.warning(
                    f"I'm trying to submit the exact right number of groups for this batch."
                    f" The attempt number  {self.cfg.attempts} ideally should divide"
                    f" total batch size {total_batch_size}"
                )
            groups_per_update = math.ceil(total_update_size / self.cfg.attempts)
            lag_groups = math.ceil(self.cfg.finetune.max_lag / self.cfg.attempts)
            logger.info(
                f"Sync RL mode on, can submit {groups_per_update} groups for each update,"
                f" that makes {groups_per_update * self.cfg.attempts} samples per update"
            )
            logger.info(
                f"Max lag is {self.cfg.finetune.max_lag} samples, that makes {lag_groups} additional starting chunks"
            )
            can_submit_before_update = lag_groups + groups_per_update
        else:
            groups_per_update = None
            can_submit_before_update = math.inf

        logger.info(f"Start {'train' if self.is_training else 'test'} actor loop")
        with (
            write_to_streams(self.data_stream, "a") as data_stream_writer,
            write_to_streams(self.stats_stream, "a") as stats_writer,
        ):
            while True:
                # the user function must do next(...) to run each iteration
                yield

                if self.trainer_state.propagated_weight_version > last_trainer_version:
                    if max_lag is not None:
                        assert groups_per_update is not None
                        can_submit_before_update += groups_per_update
                    # the weights have been updated, publish the stats of the previous trainer version
                    trainer_version_to_publish = last_trainer_version
                    last_trainer_version = self.trainer_state.propagated_weight_version

                # First, submit all problems you can until the problem queue is full
                if not self.is_scheduling_paused:
                    while True:
                        blocked_by_lag = submitted_groups == can_submit_before_update and self.is_training
                        if not blocked_by_lag and not self.problem_queue.full():
                            try:
                                try:
                                    problem = next(problem_iter)
                                    self.problem_queue.put(problem, block=False)
                                    submitted_groups += 1
                                except queue.Full:            
                                    assert False, "Problem queue was not full just a moment ago, but now it is full"
                            except StopIteration:
                                break
                        else:
                            break

                # Second, try return a result
                try:
                    # Directly get the result from the SharedMemoryQueue
                    rollout_results = self.result_queue.get(block=False)
                except queue.Empty:
                    continue

                if isinstance(rollout_results, Exception):
                    logger.error("Stop actor loop due to error")
                    raise rollout_results

                assert isinstance(rollout_results, list)
                assert isinstance(rollout_results[0], RolloutResult)
                group_samples = sum(len(r.training_texts) for r in rollout_results)

                published_samples += group_samples
                samples_in_queue = self.result_queue.qsize() * attempts
                all_text_dumps = []
                for r in rollout_results:
                    for text in r.training_texts:
                        all_text_dumps.append(text.model_dump())
                data_stream_writer.write(all_text_dumps)
                in_progress = submitted_groups - finished_groups
                logger.info(
                    f"Published {group_samples} {'train' if self.is_training else 'test'} samples"
                    f" to {self.data_stream}, total {published_samples} samples so far, {samples_in_queue} samples in the result queue,"
                    f" {in_progress} groups in progress"
                )

                    
                self.update_stats(rollout_results=rollout_results)

                finished_groups += 1
                time_to_publish_train_stats = (
                    self.is_training
                    and trainer_version_to_publish is not None
                ) or self.debug_mode 
                time_to_publish_test_stats = finished_groups == expected_rollouts

                # Publish stats at every new model version or if all tapes are finished
                if time_to_publish_train_stats or time_to_publish_test_stats:
                    if self.is_training:
                        loop_stats = {
                            "published_samples": published_samples,
                            "problem_queue_size": self.problem_queue.qsize(),
                            "result_queue_size": self.result_queue.qsize(),
                            "finished_groups": finished_groups,
                            "trainer_model_version": trainer_version_to_publish, 
                            "time_since_start": time.time() - loop_start_time,
                            "groups_in_progress": in_progress,
                        }
                        trainer_version_to_publish = None
                    else:
                        loop_stats = {
                            "trainer_model_version": last_trainer_version
                            }

                    self.publish_stats(
                        stats_writer=stats_writer,
                        loop_stats=loop_stats,
                    )


                if finished_groups == expected_rollouts:
                    logger.info(f"Finished {expected_rollouts} rollouts, stopping actor loop")
                    break

    def publish_stats(self, stats_writer: StreamWriter, loop_stats: Dict):
        split_name = "test_" if not self.is_training else ""

        stats = defaultdict(float)
        for metric_name, dict_of_stats_per_metric in self.stats.items():
            for agg, group_stats in calculate_stats(dict_of_stats_per_metric).items():
                stats[f"{split_name}{metric_name}_{agg}"] = group_stats

            for dataset_name, list_of_stats_per_metric_and_dataset in self.stats[metric_name].items():
                for agg, sub_stats in calculate_stats(list_of_stats_per_metric_and_dataset).items():
                    stats[f"{dataset_name}/{metric_name}_{agg}"] = sub_stats

        stats |= (
            {
                f"{split_name}{k}": v
                for k, v in always_or_never_success_stats(self.stats["success"]).items()
            }
            | {
                f"{split_name}latency_" + k: v
                for k, v in calculate_stats(self.latency_list).items()
            }
            | {
                f"{split_name}model_version_" + k: v
                for k, v in calculate_stats(self.model_versions_list).items()
            }
        )

        stats |= loop_stats
        for k, v in self.sliding_stats.items():
            stats[k] = sum(v) / len(v) if v else 0
        if self.curriculum_stage_state is not None:
            stats["curriculum_stage_active"] = self.curriculum_stage_state.get("index", 0)
        if self.cfg.wandb.use_wandb:
            wandb.log({f"actor/{k}": v for k, v in stats.items()})
        stats_writer.write(stats)
        self.init_stats()  # Reset stats for the next iteration


def run_actor_loop(cfg: DictConfig):
    set_streams_backend(**cfg.streams)

    # set seed for reproducibility (mostly intended for dataset loading)
    random.seed(cfg.seed)

    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path / "actor", "actor")
    logger.info(f"Current dir: {os.getcwd()}, experiment root dir: {cfg.output_dir}")
    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_path / "actor", flatten_dict_config(cfg))  # type: ignore
        if run is None:
            raise ValueError("Failed to initialize wandb run")
    llm_urls = str(cfg.me.llm_urls).split("+")

    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
    test_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats_test")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")
    test_data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor_test")

    dataset_loader = hydra.utils.get_method(cfg.dataset_loader)
    # Get dataset loader parameters if they exist in config, otherwise use empty dict
    dataset_loader_params = cfg.get('dataset_loader_params', {})
    # Use **dataset_loader_params to pass parameters only if they exist
    train_dataset = dataset_loader(cfg.train_dataset_names, **dataset_loader_params)
    test_dataset = dataset_loader(cfg.test_dataset_names, **dataset_loader_params)
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin : cfg.train_subset.end]
    logger.info(f"Loaded {len(train_dataset)} training problems")
    logger.info(f"Loaded {len(test_dataset)} test problems")

    finetune_model_path = exp_path / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
    else:
        actor_model_path = cfg.model_path
    
    train_llms = [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_model_path),
            parameters=cfg.llm.parameters,
            use_cache=False,
            collect_logprobs=True,
            observe_llm_calls=False,
        )
        for url in llm_urls
    ]
    test_llms = [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_model_path),
            parameters=cfg.test_llm.parameters,
            use_cache=False,
            collect_logprobs=True,
            observe_llm_calls=False,
        )
        for url in llm_urls
    ]

    wait_for_inference_servers(llm_urls)
    wait_for_environments(cfg)
    trainer_state = TrainerState(exp_path)
    if cfg.debug.mode:
        trainer_state.debug_mode_init()
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()

    train_loop = ActorLoop(
        data_stream=data_stream, cfg=cfg, trainer_state=trainer_state, stats_stream=stats_stream, llms=train_llms
    )
    train_loop_run = train_loop.run(
        dataset=train_dataset,
    )
    test_loop = ActorLoop(
        data_stream=test_data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=test_stats_stream,
        llms=test_llms,
        is_training=False,
    )
    test_loop_run = None

    last_regular_eval = -1
    current_eval = -1
    while True:
        assert trainer_state.propagated_weight_version is not None

        # 1. Start a new test loop if needed
        next_regular_eval = (
            trainer_state.propagated_weight_version
            if last_regular_eval == -1
            else last_regular_eval + cfg.eval_every_n_versions
        )
        if (
            cfg.eval_every_n_versions
            and not cfg.debug.mode
            and trainer_state.propagated_weight_version >= next_regular_eval
            and test_dataset
            and test_loop_run is None
        ):
            logger.info("Create test loop")
            test_loop_run = test_loop.run(
                dataset=test_dataset,
            )
            train_loop.is_scheduling_paused = True
            current_eval = next_regular_eval

        # 2. If there is an active test loop, keep it running
        if test_loop_run is not None:
            try:
                _ = next(test_loop_run)
            except StopIteration:
                # 2.1 If the test loop is finished, resume scheduling the training loop
                test_loop_run = None
                last_regular_eval = current_eval
                train_loop.is_scheduling_paused = False
                logger.info("Test loop finished")

        # 3. Keep running the training loop
        _ = next(train_loop_run)
