import contextlib
import copy
import logging
import math
import os
import queue
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import List

from pipelinerl.cot_math_agent import (
    CoTMathAgent,
    RLMathTape,
)
from pipelinerl.actor_processing import (
    convert_problems_to_tapes,
    extract_tape_training_samples,
)
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from tapeagents.core import Tape, TrainingText
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from tapeagents.llms import TrainableLLM

import wandb
from pipelinerl.load_datasets import load_datasets
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


def chunks(problems: list, n: int, random_batch=False):
    """Yield chunks from problems, either sequentially or randomly.

    Args:
        problems: List to chunk
        n: Size of each chunk
        random_batch: If True, yield random batches. If False, loop through once.
    """
    if n > len(problems):
        raise ValueError("n should be less than or equal to the length of problems")
    if random_batch:
        while True:
            yield random.sample(problems, min(n, len(problems)))
    else:
        for i in range(0, len(problems), n):
            yield problems[i : i + n]


def without_replacement(problems: list, n: int):
    while True:
        shuffled_problems = random.sample(problems, len(problems))
        for i in range(0, len(problems), n):
            if i + n > len(problems):
                break
            yield shuffled_problems[i : i + n]


def focused_chunks(problems: list, n: int, focus_n: int, focus_duration: int):
    """Choose focus_n samples to focus on. Yield focus_duration chunks of those samples. Repeat"""
    if focus_n > len(problems):
        raise ValueError("focus_n should be less than or equal to the length of problems")
    while True:
        focus_samples = random.sample(problems, focus_n)
        for _ in range(focus_duration):
            # sample n examples from the focus samples
            yield random.sample(focus_samples, n)


def batch_run_agent_replica(
    agent: CoTMathAgent, tapes: list[RLMathTape], model_version: int, final_tape_queue: Queue
):
    time_before_llm_call = time.time()
    final_tapes = agent.run_batch(tapes)
    for tape in final_tapes:
        tape.metadata.result["model_version"] = model_version
    latency = time.time() - time_before_llm_call
    try:
        final_tape_queue.put_nowait((final_tapes, latency))
    except queue.Full:
        logger.warning(
            "We are making rollouts faster than we can write them"
        )
        final_tape_queue.put((final_tapes, latency))


class ActorLoop:
    def __init__(
        self,
        agent_replicas: list[CoTMathAgent],
        tapes_stream: StreamSpec,
        data_stream: StreamSpec | None,
        trainer_state: TrainerState,
        stats_stream: StreamSpec,
        cfg: DictConfig,
    ) -> None:
        self.agent_replicas = agent_replicas
        self.tapes_stream = tapes_stream
        self.data_stream = data_stream
        self.trainer_state = trainer_state
        self.stats_stream = stats_stream
        self.window_size = 500 // cfg.actor.chunk_size
        self.stats_aggregator = SlidingWindowAggregator(window_size=self.window_size)
        self.thread_pool_size = cfg.actor.threads_per_llm * len(agent_replicas)
        self.submit_delay = cfg.actor.submit_delay / len(agent_replicas)
        self.sampling_cfg = cfg.actor.sampling
        logger.info(
            f"Using a thread pool of size {self.thread_pool_size}, submitting training chunks with delay {self.submit_delay}"
        )
        self.cfg = cfg
        # should be removed in the future when extract_tape_training_samples moved in the repo
        OmegaConf.set_struct(self.cfg, False)  # Disable strict mode

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
    
    def update_stats(self, new_tape, tape_stats):
        dataset_name = new_tape.steps[0].metadata.other["dataset"]
        parent_id = new_tape.metadata.parent_id
        self.reward_stats[dataset_name][parent_id].append(tape_stats["reward"])
        self.step_stats[dataset_name][parent_id].append(tape_stats["steps"])
        self.success_stats[dataset_name][parent_id].append(tape_stats["success"])
        self.no_errors_stats[dataset_name][parent_id].append(tape_stats["no_error"])
        self.no_answer_stats[dataset_name][parent_id].append(tape_stats["no_answer"])
        self.prompt_tokens[dataset_name][parent_id].append(tape_stats["prompt_tokens"])
        self.output_tokens[dataset_name][parent_id].append(tape_stats["output_tokens"])
        self.overflows[dataset_name][parent_id].append(tape_stats["overflow"])

    def run_agent_make_data(
        self,
        dataset: List[dict],
        is_training: bool = True,
    ):
        self.init_stats()
        final_tape_queue = Queue[tuple[list[RLMathTape], float]]()
        next_agent_replica = 0
        last_submit = 0

        attempts = self.cfg.attempts if is_training else 1
        submitted_tapes = 0
        published_samples = 0
        expected_number_of_tapes = -1 if is_training else len(dataset) * attempts
        finished_tapes = 0
        submitted_chunks = 0
        finished_chunks = 0
        # If training, we expect to sample infinitely
        # for train sample, sample random batches infinitely
        # for test samples, loop through the dataset once
        if is_training:
            if self.sampling_cfg.method == "random":
                logger.info(f"Using random problem sampling")
                dataset_iter = chunks(
                    dataset, self.cfg.actor.chunk_size // attempts, random_batch=True
                )
            elif self.sampling_cfg.method == "without_replacement":
                logger.info(f"Using without replacement problem sampling")
                dataset_iter = without_replacement(
                    dataset, self.cfg.actor.chunk_size // attempts
                )
            elif self.sampling_cfg.method == "focused":
                logger.info(f"Using focused problem sampling: {self.sampling_cfg}")
                dataset_iter = focused_chunks(
                    dataset,
                    self.cfg.actor.chunk_size // attempts,
                    self.sampling_cfg.focus_n,
                    self.sampling_cfg.focus_duration,
                )
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_cfg.method}")
        else:
            dataset_iter = chunks(dataset, self.cfg.actor.chunk_size // attempts)
        split_name = "" if is_training else "test"
        assert self.trainer_state.propagated_weight_version is not None

        last_trainer_version = self.trainer_state.propagated_weight_version
        max_lag = self.cfg.max_lag if is_training else None
        if max_lag is not None:
            total_batch_size = self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
            total_update_size = math.ceil(self.cfg.finetune.weight_update_interval / total_batch_size) * total_batch_size
            if total_batch_size % self.cfg.actor.chunk_size != 0:
                logger.warning(
                    f"I'm trying to submit the exact right number of chunks for this batch."
                    f" Actor chunk size {self.cfg.actor.chunk_size} ideally should divide total batch size {total_batch_size}"
                )
            chunks_per_update = math.ceil(total_update_size / self.cfg.actor.chunk_size)
            lag_chunks = math.ceil(max_lag / self.cfg.actor.chunk_size)
            logger.info(f"Sync RL mode on, can submit {chunks_per_update} chunks for each update,"
                        f" that makes {chunks_per_update * self.cfg.actor.chunk_size} samples per update")
            logger.info(f"Max lag is {max_lag} samples, that makes {lag_chunks} additional starting chunks")
            can_submit_before_update = lag_chunks + chunks_per_update
        else:
            chunks_per_update = None
            can_submit_before_update = math.inf

        final_tape_queue = queue.Queue()
        @contextlib.contextmanager
        def maybe_write_stream(stream):
            if stream:
                with write_to_streams(stream, "a") as writer:
                    yield writer
            else:
                yield None
        with (
            ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor,
            maybe_write_stream(self.data_stream) as data_stream_writer,
            write_to_streams(self.tapes_stream, "a") as tapes_stream_writer,
            write_to_streams(self.stats_stream, "a") as stats_writer,
        ):
            logger.info(f"Using {executor} for parallel processing")
            # take sub samples of size cfg.actor.chunk_size // cfg.attempts
            while True:
                if self.trainer_state.propagated_weight_version > last_trainer_version:
                    if max_lag is not None:
                        assert chunks_per_update is not None
                        can_submit_before_update += chunks_per_update
                    last_trainer_version = self.trainer_state.propagated_weight_version

                # First, submit a new task
                cur_time = time.time()
                if (
                    cur_time - last_submit > (self.submit_delay if is_training else 0)
                    and (submitted_chunks < can_submit_before_update or not is_training)
                    and submitted_chunks - finished_chunks < self.thread_pool_size
                ):
                    try:
                        sub_samples = next(dataset_iter)
                        start_tapes = convert_problems_to_tapes(sub_samples, self.cfg)
                        start_tapes = [
                            copy.deepcopy(tape)
                            for tape in start_tapes
                            for _ in range(attempts)
                        ]

                        future = executor.submit(
                            batch_run_agent_replica,
                            self.agent_replicas[next_agent_replica],
                            start_tapes,
                            self.trainer_state.propagated_weight_version,
                            final_tape_queue
                        )
                        future.add_done_callback(
                            lambda fut: logger.error(
                                f"Exception while running the agent: {fut.exception()}",
                                exc_info=fut.exception(),
                            )
                            if fut.exception()
                            else None
                        )
                        logger.info(
                            f"Submitted {len(start_tapes)}{' ' + split_name if split_name else ''} tapes to agent replica {next_agent_replica},"
                            f" model version is {self.trainer_state.propagated_weight_version}"
                        )
                        next_agent_replica = (next_agent_replica + 1) % len(
                            self.agent_replicas
                        )
                        submitted_chunks += 1
                        submitted_tapes += len(start_tapes)
                        last_submit = cur_time
                    except StopIteration:
                        pass

                # Second, try return a result
                try:
                    final_tapes, latency = final_tape_queue.get_nowait()
                except Empty:
                    continue
                assert isinstance(final_tapes, list) and all(
                    isinstance(tape, Tape) for tape in final_tapes
                )
                training_samples: List[TrainingText] = []

                max_model_version = -1
                for new_tape in final_tapes:
                    tape_training_samples, tape_stats = (
                        extract_tape_training_samples(
                            new_tape, self.agent_replicas[0], self.cfg
                        )
                    )

                    max_model_version = max(
                        max_model_version, new_tape.metadata.result["model_version"]
                    )
                    overflow = False
                    for sample in tape_training_samples:
                        sample.metadata["model_version"] = new_tape.metadata.result[
                            "model_version"
                        ]
                        eos_token_id = self.agent_replicas[
                            next_agent_replica
                        ].llm.tokenizer.eos_token_id
                        overflow = (
                            False if sample.input_ids[-1] == eos_token_id else True
                        ) or overflow
                    
                    tape_stats |= {"overflow": overflow}
                    self.update_stats(new_tape, tape_stats)

                    if is_training:
                        training_samples.extend(tape_training_samples)

                    # Important: write the tape after training samples are extracted,
                    # because rewards are added at that time
                    tapes_stream_writer.write(new_tape)                            


                published_samples += len(training_samples)
                samples_in_queue = final_tape_queue.qsize() * self.cfg.actor.chunk_size                    
                if data_stream_writer and training_samples:
                    for text in training_samples:
                        data_stream_writer.write(text)
                    logger.info(
                        f"Published {len(training_samples)}{' ' + split_name if split_name else ''} samples/tapes"
                        f" to {self.data_stream} and {self.tapes_stream}, total {published_samples} samples so far, {samples_in_queue} samples in the queue"
                    )
                else:
                    logger.info(
                        f"Published {len(final_tapes)}{' ' + split_name if split_name else ''} tapes to {self.tapes_stream}"
                    )

                flattened_prompt_tokens = [
                    prompt_tokens
                    for dataset_dict in self.prompt_tokens.values()
                    for attempt_list in dataset_dict.values()
                    for prompt_tokens in attempt_list
                ]
                flattened_output_tokens = [
                    output_tokens
                    for dataset_dict in self.output_tokens.values()
                    for attempt_list in dataset_dict.values()
                    for output_tokens in attempt_list
                ]
                self.stats_aggregator.update(flattened_prompt_tokens, flattened_output_tokens)

                finished_chunks += 1

                finished_tapes += len(final_tapes)
                yield final_tapes

                # if we are training publish stats at every step else if all tapes are finished, publish stats
                if is_training or finished_tapes == expected_number_of_tapes:
                    if is_training:
                        loop_stats = {
                            "published_samples": published_samples,
                            "samples_in_queue": samples_in_queue,
                            "finished_chunks": finished_chunks,
                            "published_model_version": max_model_version,
                            "latency": latency,
                        }
                    else:
                        loop_stats = {"published_model_version": max_model_version}

                    self.publish_stats(
                        stats_writer=stats_writer,
                        loop_stats=loop_stats,
                        split_name=split_name,
                    )

                if finished_tapes == expected_number_of_tapes:
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
        if "finished_chunks" in loop_stats and loop_stats["finished_chunks"] >= 2 * self.window_size:
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

    tape_stream = SingleStreamSpec(exp_path=exp_path, topic="tapes")
    test_tape_stream = SingleStreamSpec(exp_path=exp_path, topic="test_tapes")
    stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
    data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")

    train_dataset = load_datasets(cfg.train_dataset_names)
    test_dataset = load_datasets(cfg.test_dataset_names)
    if cfg.train_subset:
        train_dataset = train_dataset[cfg.train_subset.begin:cfg.train_subset.end]
    logger.info(f"Loaded {len(train_dataset)} training problems")
    logger.info(f"Loaded {len(test_dataset)} test problems")

    # For now we assume that LLM and Actor Processes are co-located
    #llm_urls = [f"http://localhost:{port}" for port in str(cfg.me.llm_ports).split("-")]
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
    train_agent_replicas = [
        CoTMathAgent.create(
            system_prompt=cfg.system_prompt,
            llm=llm,
            max_prompt_length=cfg.agent.max_prompt_length,
        )
        for llm in train_llms
    ]

    test_agent_replicas = [
        CoTMathAgent.create(
            system_prompt=cfg.system_prompt,
            llm=llm,
            max_prompt_length=cfg.agent.max_prompt_length,
        )
        for llm in test_llms
    ]

    wait_for_inference_servers(llm_urls)
    trainer_state = TrainerState(exp_path)
    if cfg.debug.mode in ["actor", "open_loop"]:
        trainer_state.propagated_weight_version = 0
    else:
        trainer_state.start_listening()
        while trainer_state.propagated_weight_version is None:
            logger.info("Waiting for the trainer to declare the initial weight version")
            time.sleep(1)

    train_set_generator = ActorLoop(
        agent_replicas=train_agent_replicas,
        tapes_stream=tape_stream,
        data_stream=data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=stats_stream,
    ).run_agent_make_data(
        dataset=train_dataset,
        is_training=True,
    )
    test_actor_loop = ActorLoop(
        agent_replicas=test_agent_replicas,
        tapes_stream=test_tape_stream,
        data_stream=None,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=stats_stream,
    )


    last_regular_eval = -1
    while True:
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
        ):
            logger.info("Evaluating model on test set")
            _ = list(
                test_actor_loop.run_agent_make_data(
                    dataset=test_dataset,
                    is_training=False,
                )
            )
            last_regular_eval = next_regular_eval
        else:
            _ = next(train_set_generator)
