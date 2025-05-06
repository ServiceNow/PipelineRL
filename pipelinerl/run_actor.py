import logging
import math
import os
import queue
import random
import time
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import uvloop
import aiohttp

from omegaconf import DictConfig
from pydantic import BaseModel, Field

from pipelinerl.verifier_api import wait_for_verifier
from tapeagents.llms import TrainableLLM
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb

import wandb
from pipelinerl.load_datasets import load_datasets
from pipelinerl.math_rollouts import RolloutResult, generate_math_rollout
from pipelinerl.state import TrainerState
import asyncio
from collections import defaultdict
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
    calculate_per_group_stats,
    setup_logging,
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
        total_output_tokens = sum(
            sum(tokens) for tokens in self.data.output_tokens_window
        )
        total_prompt_tokens = sum(
            sum(tokens) for tokens in self.data.prompt_tokens_window
        )

        return {
            "samples_per_second": num_samples / time_span,
            "output_tokens_per_second": total_output_tokens / time_span,
            "prompt_tokens_per_second": total_prompt_tokens / time_span,
            "total_tokens_per_second": (total_output_tokens + total_prompt_tokens)
            / time_span,
        }
    

async def schedule_rollouts(
    cfg: DictConfig,
    attempts: int, 
    problem_queue: mp.Queue, 
    result_queue: mp.Queue, 
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

    async def rollout_and_maybe_produce_result(
        problem: dict, 
        group_id: int,
        llm_index: int, 
        session: aiohttp.ClientSession,
    ):
        nonlocal started_rollouts, finished_rollouts
        try:
            llm = llms[llm_index]
            model_version = trainer_state.propagated_weight_version
            assert model_version is not None
            rollout_result = await generate_math_rollout(cfg, llm, problem, session)
            rollout_result.model_version = model_version    
            # Make a group id that will be different from groups made by another rollout maker
            full_group_id = f"{scheduler_name}_{group_id}"
            rollout_result.group_id = full_group_id
            for sample in rollout_result.training_texts:
                # Downstream in the pipeline we'll need these fields in every sample
                sample.metadata["model_version"] = model_version
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
            logger.error("Exception in rollout", exc_info=e)
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
    timeout = aiohttp.ClientTimeout(total=3600.0, connect=3600.0, sock_read=3600.0)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while True:
            if time.time() - last_logged > 10. and sum(active_rollouts):
                logger.info(f"{scheduler_name}: "
                            f"rollouts in progress: {sum(active_rollouts)}, "
                            f"groups in progress: {len(group_rollouts)}, "
                            f"rollouts started so far: {started_rollouts}, "
                            f"rollouts finished so far: {finished_rollouts}")
                last_logged = time.time()

            if group_rollout_index == attempts:
                try:
                    problem = problem_queue.get_nowait()
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
            loop.create_task(
                rollout_and_maybe_produce_result(
                    problem=problem,
                    group_id=group_id,
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
        schedule_rollouts(cfg, attempts, problem_queue, result_queue, trainer_state, llms, scheduler_name)
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
        self.window_size = 500 // cfg.attempts
        self.stats_aggregator = SlidingWindowAggregator(window_size=self.window_size)
        self.llms = llms
        # Can't use typing with multiprocessing.
        # Queue of dict or None
        self.problem_queue = mp.Queue(32)
         # Queue of list[RolloutResult] or Exception 
        self.result_queue = mp.Queue()    
        self.loop_start_time = -1
        self.cfg = cfg
        self.is_training = is_training
        self.is_scheduling_paused = False

        # Determine the number of processes to use 
        num_processes = min(self.cfg.actor.rollout_workers, len(self.llms))
        attempts = self.cfg.attempts if is_training else 1
        
        # Divide LLMs approximately equally across processes
        llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.llms):
            llm_groups[i % num_processes].append((i, llm))
        
        # Create and start multiple rollout processes
        self.rollout_processes = []
        for llm_group in llm_groups:
            assert llm_group             
            llm_idxs = [llm[0] for llm in llm_group]
            llms = [llm[1] for llm in llm_group]   
            scheduler_name = f"{'train' if is_training else 'test'} scheduler for llms {','. join([str(i) for i in llm_idxs])}"
            process = mp.Process(
                target=rollout_maker_entrypoint,
                args=(self.cfg, attempts, self.problem_queue, self.result_queue, llms, scheduler_name)
            )
            process.start()
            self.rollout_processes.append(process)

    def init_stats(self):
        # reset after publishing
        self.reward_stats = defaultdict(lambda: defaultdict(list))
        self.step_stats = defaultdict(lambda: defaultdict(list))
        self.no_errors_stats = defaultdict(lambda: defaultdict(list))
        self.no_answer_stats = defaultdict(lambda: defaultdict(list))
        self.success_stats = defaultdict(lambda: defaultdict(list))
        self.prompt_tokens = defaultdict(lambda: defaultdict(list))
        self.output_tokens = defaultdict(lambda: defaultdict(list))
        self.overflows = defaultdict(lambda: defaultdict(list))
    
    def update_stats(self, result: RolloutResult):
        dataset_name = result.dataset_name
        group_id = result.group_id
        stats = result.metrics
        self.reward_stats[dataset_name][group_id].append(stats["reward"])
        self.success_stats[dataset_name][group_id].append(stats["success"])
        self.no_errors_stats[dataset_name][group_id].append(stats["no_error"])
        self.no_answer_stats[dataset_name][group_id].append(stats["no_answer"])
        self.prompt_tokens[dataset_name][group_id].append(stats["prompt_tokens"])
        self.output_tokens[dataset_name][group_id].append(stats["output_tokens"])
        self.overflows[dataset_name][group_id].append(stats["overflow"])

    def run(self, dataset: list[tuple[str, dict]]):
        loop_start_time = time.time()
        self.init_stats()

        attempts = self.cfg.attempts if self.is_training else 1
        published_samples = 0
        submitted_groups = 0
        finished_groups = 0
        expected_number_of_samples = -1 if self.is_training else len(dataset)
        if expected_number_of_samples > 0:
            logger.info(f"Will stop after {expected_number_of_samples} samples")

        # If training, we expect to sample infinitely
        # for train sample, sample random batches infinitely
        # for test samples, loop through the dataset once
        if self.is_training:
            problem_iter = random_iter(dataset)
        else:
            problem_iter = sequential_iter(dataset)
        split_name = "" if self.is_training else "test"
        assert self.trainer_state.propagated_weight_version is not None

        last_trainer_version = self.trainer_state.propagated_weight_version
        max_lag = self.cfg.finetune.max_lag if self.is_training else None
        if max_lag is not None:
            total_batch_size = self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
            total_update_size = math.ceil(self.cfg.finetune.weight_update_interval / total_batch_size) * total_batch_size
            if total_batch_size % self.cfg.attempts != 0:
                logger.warning(
                    f"I'm trying to submit the exact right number of groups for this batch."
                    f" The attempt number  {self.cfg.attempts} ideally should divide"
                    f" total batch size {total_batch_size}"
                )
            groups_per_update = math.ceil(total_update_size / self.cfg.attempts)
            lag_groups = math.ceil(self.cfg.finetune.max_lag / self.cfg.attempts)
            logger.info(f"Sync RL mode on, can submit {groups_per_update} groups for each update,"
                        f" that makes {groups_per_update * self.cfg.attempts} samples per update")
            logger.info(f"Max lag is {self.cfg.finetune.max_lag} samples, that makes {lag_groups}"
                        " additional starting chunks")
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
                    last_trainer_version = self.trainer_state.propagated_weight_version

                # First, submit all problems you can
                if not self.is_scheduling_paused:
                    while True:
                        blocked_by_lag = (
                            submitted_groups == can_submit_before_update and self.is_training
                        )
                        if not blocked_by_lag and not self.problem_queue.full():
                            try:
                                problem = next(problem_iter)
                                self.problem_queue.put_nowait(problem)  
                                submitted_groups += 1
                            except StopIteration:
                                break
                        else:
                            break

                # Second, try return a result
                try:
                    rollout_results = self.result_queue.get_nowait()
                    if isinstance(rollout_results, Exception):
                        logger.error("Stop actor loop due to error")
                        raise rollout_results
                except queue.Empty:
                    continue

                assert isinstance(rollout_results, list)
                assert isinstance(rollout_results[0], RolloutResult)
                for result in rollout_results:
                    if len(result.training_texts) > 1:
                        raise NotImplementedError("Multi-turn rollouts not tested yet")
                group_samples = sum(len(r.training_texts) for r in rollout_results)
    
                published_samples += group_samples
                samples_in_queue = self.result_queue.qsize() * attempts                   
                for r in rollout_results:
                    for text in r.training_texts:
                        data_stream_writer.write(text)
                logger.info(
                    f"Published {group_samples}{' ' + split_name if split_name else ''} samples"
                    f" to {self.data_stream}, total {published_samples} samples so far, {samples_in_queue} samples in the queue"
                )

                flattened_prompt_tokens = [
                    call.prompt_length_tokens
                    for result in rollout_results
                    for call in result.llm_calls
                ]
                flattened_output_tokens = [
                    call.output_length_tokens
                    for result in rollout_results
                    for call in result.llm_calls
                ]
                max_model_version = 0
                max_latency = 0
                for result in rollout_results:
                    assert result.model_version is not None
                    max_model_version = max(max_model_version, result.model_version)
                    max_latency = max(max_latency, result.latency)
                    self.update_stats(result)

                self.stats_aggregator.update(flattened_prompt_tokens, flattened_output_tokens)

                finished_groups += 1

                # if we are training publish stats at every step else if all tapes are finished, publish stats
                if self.is_training or published_samples == expected_number_of_samples:
                    if self.is_training:
                        loop_stats = {
                            "published_samples": published_samples,
                            "samples_in_queue": samples_in_queue,
                            "finished_groups": finished_groups,
                            "published_model_version": max_model_version,
                            "latency": max_latency,
                            "time_since_start": time.time() - loop_start_time,
                        }
                    else:
                        loop_stats = {"published_model_version": max_model_version}

                    self.publish_stats(
                        stats_writer=stats_writer,
                        loop_stats=loop_stats,
                        split_name=split_name,
                    )

                if published_samples == expected_number_of_samples:
                    logger.info(f"Finished {expected_number_of_samples} samples, stopping actor loop")
                    break


    def publish_stats(self, stats_writer: StreamWriter, loop_stats, split_name: str = ""):
        sliding_stats = self.stats_aggregator.get_stats()
        stats = (
            {
                (split_name + "_" if split_name else "") + "reward_" + k: v
                for k, v in calculate_per_group_stats(self.reward_stats).items()
            }
            | {
                (split_name + "_" if split_name else "") + "success_" + k: v
                for k, v in calculate_per_group_stats(self.success_stats).items()
            }
            | {
                (split_name + "_" if split_name else "") + "no_error_" + k: v
                for k, v in calculate_per_group_stats(self.no_errors_stats).items()
            }
            | {
                (split_name + "_" if split_name else "") + "no_answer_" + k: v
                for k, v in calculate_per_group_stats(self.no_answer_stats).items()
            }
            | {
                (split_name + "_" if split_name else "") + "prompt_tokens_" + k: v
                for k, v in calculate_per_group_stats(self.prompt_tokens).items()
            }
            | {
                (split_name + "_" if split_name else "") + "output_tokens_" + k: v
                for k, v in calculate_per_group_stats(self.output_tokens).items()
            }
            | {
                (split_name + "_" if split_name else "") + "overflows_" + k: v
                for k, v in calculate_per_group_stats(self.overflows).items()
            }
            | {
                (split_name + "_" if split_name else "") + k: v
                for k, v in always_or_never_success_stats(self.success_stats).items()
              }
        )

        for dataset_name in self.reward_stats.keys():
            sub_stats = (
                {
                    "reward_" + k: v
                    for k, v in calculate_stats(self.reward_stats[dataset_name]).items()
                }
                | {
                    "success_" + k: v
                    for k, v in calculate_stats(
                        self.success_stats[dataset_name]
                    ).items()
                }
                | {
                    "no_error_" + k: v
                    for k, v in calculate_stats(
                        self.no_errors_stats[dataset_name]
                    ).items()
                }
                | {
                    "no_answer_" + k: v
                    for k, v in calculate_stats(
                        self.no_answer_stats[dataset_name]
                    ).items()
                }
                | {
                    "prompt_tokens_" + k: v
                    for k, v in calculate_stats(
                        self.prompt_tokens[dataset_name]
                    ).items()
                }
                | {
                    "output_tokens_" + k: v
                    for k, v in calculate_stats(
                        self.output_tokens[dataset_name]
                    ).items()
                }
                | {
                    "overflows_" + k: v
                    for k, v in calculate_stats(self.overflows[dataset_name]).items()
                }
            )
            sub_stats = {dataset_name + "_" + k: v for k, v in sub_stats.items()}
            stats |= sub_stats

        stats |= loop_stats
        if loop_stats.get("finished_groups", 0) >= 2 * self.window_size:
            stats |= sliding_stats
        wandb.log({"actor/" + k: v for k, v in stats.items()})
        stats_writer.write(stats)
        self.init_stats()


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

    train_dataset = load_datasets(cfg.train_dataset_names)
    test_dataset = load_datasets(cfg.test_dataset_names)
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin:cfg.train_subset.end]
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
    wait_for_verifier(cfg.verifier)
    trainer_state = TrainerState(exp_path)
    if cfg.debug.mode in ["actor", "open_loop"]:
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        trainer_state.wait_for_model_version()

    train_loop = ActorLoop(
        data_stream=data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=stats_stream,
        llms=train_llms
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
