import asyncio
import itertools
import logging
import math
import multiprocessing as mp
import os
import queue
import random
import time
from collections import defaultdict
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path

import aiohttp
import hydra
import uvloop
from omegaconf import DictConfig
from pydantic import BaseModel, Field
from tapeagents.llms import TrainableLLM
from pipelinerl.async_vlm import TrainableVLM
from typing import Dict, List

import wandb
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from pipelinerl.rollouts import RolloutResult
from pipelinerl.shared_memory_array import SharedMemoryArray
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



def make_stats_dict() -> dict:
    return defaultdict(lambda: defaultdict(list))


async def schedule_rollouts(
    cfg: DictConfig,
    attempts: int,
    problem_queue: mp.Queue,
    result_queue: mp.Queue,
    io_buffer,
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
    max_group_size_bytes = 0
    # Track rollouts per problem group
    group_rollouts = {}
    rollout_policy = hydra.utils.get_method(cfg.actor.rollout_policy)
    logger.info(f"Use rollout policy: {rollout_policy}")

    async def rollout_and_maybe_produce_result(
        problem: dict,
        slot: int,
        group_id: int,
        rollout_index: int,
        llm_index: int,
        session: aiohttp.ClientSession,
    ):
        nonlocal started_rollouts, finished_rollouts, max_group_size_bytes
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
                group_bytes = group_rollouts[group_id]
                max_group_size_bytes = max(max_group_size_bytes, len(group_bytes))
                io_buffer[slot] = group_bytes
                result_queue.put(slot)
                del group_rollouts[group_id]
            finished_rollouts += 1
        except Exception as e:
            # Cancel all tasks except the current one
            logger.error("Exception in rollout, stop all other rollout tasks", exc_info=e)
            current_task = asyncio.current_task(loop=loop)
            for task in asyncio.all_tasks(loop=loop):
                if task != current_task:
                    task.cancel()
            io_buffer[slot] = e
            result_queue.put(slot)
            logger.error("Stopped all tasks and put exception in the result queue")
        finally:
            active_rollouts[llm_index] -= 1

    group_id = -1
    group_rollout_index = attempts
    problem = None
    slot = None

    last_logged = time.time()
    logger.info("Starting rollout scheduler")
    connector = aiohttp.TCPConnector(limit=50000, limit_per_host=50000, keepalive_timeout=1.0)
    timeout = aiohttp.ClientTimeout(total=3600.0, connect=3600.0, sock_read=3600.0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while True:
            if time.time() - last_logged > 10.0 and sum(active_rollouts):
                logger.info(
                    f"{scheduler_name}: "
                    f"rollouts in progress: {sum(active_rollouts)}, "
                    f"groups in progress: {len(group_rollouts)}, "
                    f"rollouts started so far: {started_rollouts}, "
                    f"rollouts finished so far: {finished_rollouts}, "
                    f"max group size in bytes: {max_group_size_bytes}"
                )
                last_logged = time.time()

            if group_rollout_index == attempts:
                try:
                    slot = problem_queue.get_nowait()
                    problem = io_buffer[slot]
                except queue.Empty:
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
            assert problem is not None and slot is not None
            loop.create_task(
                rollout_and_maybe_produce_result(
                    problem=problem,
                    slot=slot,
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
    problem_queue: mp.Queue,
    result_queue: mp.Queue,
    io_buffer,
    llms: list[TrainableLLM],
    scheduler_name: str,
):
    trainer_state = TrainerState(Path(cfg.output_dir))
    if cfg.debug.mode in ["actor", "open_loop"]:
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()
    loop = uvloop.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        schedule_rollouts(cfg, attempts, problem_queue, result_queue, io_buffer, trainer_state, llms, scheduler_name)
    )
    loop.close()
    logger.info("Rollout maker loop closed")


def random_iter(problems: list):
    while True:
        yield random.sample(problems, 1)[0]


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
        self.debug_mode = cfg.debug.mode in ["actor", "open_loop"]

        # Determine the number of processes to use
        num_processes = min(self.cfg.actor.rollout_workers, len(self.llms))
        attempts = self.cfg.attempts if is_training else 1

        # Divide LLMs approximately equally across processes
        llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.llms):
            llm_groups[i % num_processes].append((i, llm))

        self.smm = SharedMemoryManager()
        self.smm.start()

        # we can have a pending almost ready group for each last rollout in progress ...
        self.max_groups_in_progress = cfg.actor.llm_max_rollouts * len(self.llms)
        max_ready_groups_waiting = 128
        self.problem_queue = mp.Queue()
        self.result_queue = mp.Queue(max_ready_groups_waiting)
        self.buffer_size = self.max_groups_in_progress + max_ready_groups_waiting
        self.io_buffer = SharedMemoryArray(self.smm, self.buffer_size, int(1e10))
        self.free_slots = set(range(self.buffer_size))
        logger.info(f"Initialized {'train' if self.is_training else 'test'} actor loop")
        logger.info(f"Max groups in progress: {self.max_groups_in_progress}, buffer size: {self.buffer_size}")
        logger.info(f"Shared memory buffer size: {self.io_buffer.get_memory_size() / 2**30} Gb")

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
                args=(self.cfg, attempts, self.problem_queue, self.result_queue, self.io_buffer, llms, scheduler_name),
            )
            process.start()
            self.rollout_processes.append(process)

    def init_stats(self):
        self.reward_stats = make_stats_dict()
        self.step_stats = make_stats_dict()
        self.no_errors_stats = make_stats_dict()
        self.no_answer_stats = make_stats_dict()
        self.success_stats = make_stats_dict()
        self.prompt_tokens = make_stats_dict()
        self.output_tokens = make_stats_dict()
        self.num_turns = make_stats_dict()
        self.overflows = make_stats_dict()
        self.latency_list = []
        self.model_versions_list = []
        self.sliding_stats = defaultdict(list)

    def update_stats(self, rollout_results: List[RolloutResult]):
        for result in rollout_results:
            assert result.model_version is not None
            dataset_name = result.dataset_name
            group_id = result.group_id
            stats = result.metrics
            self.reward_stats[dataset_name][group_id].append(stats["reward"])
            self.success_stats[dataset_name][group_id].append(stats["success"])
            self.no_errors_stats[dataset_name][group_id].append(stats["no_error"])
            self.no_answer_stats[dataset_name][group_id].append(stats["no_answer"])
            self.prompt_tokens[dataset_name][group_id].append(sum(result.prompt_tokens))
            self.output_tokens[dataset_name][group_id].append(sum(result.output_tokens))
            self.overflows[dataset_name][group_id].append(stats["overflow"])
            self.num_turns[dataset_name][group_id].append(len(result.training_texts))
            self.latency_list.append(result.latency)
            self.model_versions_list.append(result.model_version)
        dataset_name = rollout_results[0].dataset_name
        
        prompt_length_tokens = list(itertools.chain(*(result.prompt_tokens for result in rollout_results)))
        output_length_tokens = list(itertools.chain(*(result.output_tokens for result in rollout_results)))
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
            problem_iter = random_iter(dataset)
        else:
            problem_iter = sequential_iter(dataset)
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

                # First, submit all problems you can
                if not self.is_scheduling_paused:
                    while True:
                        blocked_by_lag = submitted_groups == can_submit_before_update and self.is_training
                        in_progress = submitted_groups - finished_groups
                        assert 0 <= in_progress and in_progress <= self.max_groups_in_progress
                        if not blocked_by_lag and in_progress < self.max_groups_in_progress:
                            try:
                                problem = next(problem_iter)
                                slot = self.free_slots.pop()
                                self.io_buffer[slot] = problem
                                self.problem_queue.put_nowait(slot)
                                submitted_groups += 1
                            except StopIteration:
                                break
                        else:
                            break

                # Second, try return a result
                try:
                    slot = self.result_queue.get_nowait()
                    rollout_results = self.io_buffer[slot]
                    self.free_slots.add(slot)
                    if isinstance(rollout_results, Exception):
                        logger.error("Stop actor loop due to error")
                        raise rollout_results
                except queue.Empty:
                    continue

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
                    f" to {self.data_stream}, total {published_samples} samples so far, {samples_in_queue} samples in the queue,"
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
                            "samples_in_queue": samples_in_queue,
                            "finished_groups": finished_groups,
                            "trainer_model_version": trainer_version_to_publish, 
                            "time_since_start": time.time() - loop_start_time,
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
        stats = (
            {
                f"{split_name}reward_" + k: v
                for k, v in calculate_stats(self.reward_stats).items()
            }
            | {
                f"{split_name}success_" + k: v
                for k, v in calculate_stats(self.success_stats).items()
            }
            | {
                f"{split_name}no_error_" + k: v
                for k, v in calculate_stats(self.no_errors_stats).items()
            }
            | {
                f"{split_name}no_answer_" + k: v
                for k, v in calculate_stats(self.no_answer_stats).items()
            }
            | {
                f"{split_name}prompt_tokens_" + k: v
                for k, v in calculate_stats(self.prompt_tokens).items()
            }
            | {
                f"{split_name}output_tokens_" + k: v
                for k, v in calculate_stats(self.output_tokens).items()
            }
            | {
                f"{split_name}num_turns_" + k: v
                for k, v in calculate_stats(self.num_turns).items()
            }
            | {
                f"{split_name}overflows_" + k: v
                for k, v in calculate_stats(self.overflows).items()
            }
            | {
                f"{split_name}{k}": v
                for k, v in always_or_never_success_stats(self.success_stats).items()
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

        for dataset_name in self.reward_stats.keys():
            sub_stats = (
                {"reward_" + k: v for k, v in calculate_stats(self.reward_stats[dataset_name]).items()}
                | {"success_" + k: v for k, v in calculate_stats(self.success_stats[dataset_name]).items()}
                | {"no_error_" + k: v for k, v in calculate_stats(self.no_errors_stats[dataset_name]).items()}
                | {"no_answer_" + k: v for k, v in calculate_stats(self.no_answer_stats[dataset_name]).items()}
                | {"prompt_tokens_" + k: v for k, v in calculate_stats(self.prompt_tokens[dataset_name]).items()}
                | {"output_tokens_" + k: v for k, v in calculate_stats(self.output_tokens[dataset_name]).items()}
                | {"overflows_" + k: v for k, v in calculate_stats(self.overflows[dataset_name]).items()}
            )
            sub_stats = {dataset_name + "/" + k: v for k, v in sub_stats.items()}
            stats |= sub_stats
    
        stats |= loop_stats
        for k, v in self.sliding_stats.items():
            stats[k] = sum(v) / len(v) if v else 0
        wandb.log({f"actor/{k}": v for k, v in stats.items()})
        stats_writer.write(stats)
        self.init_stats()  # Reset stats for the next iteration


def run_actor_loop(cfg: DictConfig):
    set_streams_backend(**cfg.streams)

    random.seed(42)
    exp_path = Path(cfg.output_dir)
    setup_logging(str(exp_path / "actor"))
    logger.info(f"Current dir: {os.getcwd()}, experiment root dir: {cfg.output_dir}")
    run = init_wandb(cfg, exp_path / "actor", flatten_dict_config(cfg))  # type: ignore
    llm_urls = str(cfg.me.llm_urls).split("+")
    if run is None:
        raise ValueError("Failed to initialize wandb run")

    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
    test_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats_test")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")
    test_data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor_test")

    dataset_loader = hydra.utils.get_method(cfg.dataset_loader)
    train_dataset = dataset_loader(cfg.train_dataset_names)
    test_dataset = dataset_loader(cfg.test_dataset_names)
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin : cfg.train_subset.end]
    logger.info(f"Loaded {len(train_dataset)} training problems")
    logger.info(f"Loaded {len(test_dataset)} test problems")

    finetune_model_path = exp_path / "finetune" / "current"
    if os.path.exists(finetune_model_path):
        actor_model_path = finetune_model_path
    else:
        actor_model_path = cfg.model_path
    
    # Use TrainableVLM for vision-language models
    is_vision_model = cfg.finetune.model_class == "vision2seq-language-modeling"
    llm_class = TrainableVLM if is_vision_model else TrainableLLM
    
    train_llms = [
        llm_class(
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
        llm_class(
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
    if cfg.debug.mode in ["actor", "open_loop"]:
        trainer_state.propagated_weight_version = 0
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
