import asyncio
import logging
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import time
from collections import defaultdict
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from queue import Empty
from typing import Dict, List

import aiohttp
import hydra
import uvloop
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

import wandb
from pipelinerl.async_llm import RetryableAbortedCompletionError
from pipelinerl.domain_sampling import DomainWeightedSampler
from pipelinerl.domains.math.rollouts import length_penalty
from pipelinerl.domains.tau2.prerun import (
    CALIBRATION_GROUP_SIZE,
    CalibrationGroupEvidence,
    select_calibration_problems,
)
from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
from pipelinerl.finetune_loop import calculate_train_steps, samples_per_optimizer_step
from pipelinerl.prerun_evidence import (
    append_evidence_record,
    evidence_directory,
    prerun_enabled,
)
from pipelinerl.llm import TrainableLLM
from pipelinerl.rollouts import (
    BaseMetrics,
    RolloutResult,
    TrainingGroupEnvelope,
    rollout_has_overflow,
)
from pipelinerl.shared_memory_array import EntrySizeExceeded, SharedMemoryQueue
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


def make_rollout_audit_record(result: RolloutResult) -> dict:
    record = dict(result.audit)
    record.setdefault("task_id", None)
    record.setdefault("domain", result.domain)
    record["group_id"] = result.group_id
    record.setdefault("rollout_index", None)
    record.setdefault("model_version", result.model_version)
    record["dataset_name"] = result.dataset_name
    record.setdefault("policy_endpoint", None)
    record.setdefault("model_calls", [])
    record.setdefault("reward", result.metrics.reward)
    record.setdefault("submitted", None)
    record.setdefault("terminated", None)
    record.setdefault("stop_reason", None)
    record.setdefault("published", bool(result.training_texts))
    record.setdefault("entered_training", False)
    record.setdefault("drop_reason", None)
    record.setdefault(
        "prompt_token_lengths",
        [training_text.prompt_tokens for training_text in result.training_texts],
    )
    record.setdefault(
        "output_token_lengths",
        [training_text.output_tokens for training_text in result.training_texts],
    )
    record["n_training_texts"] = len(result.training_texts)
    return record


def write_rollout_audit_records(
    audit_writer: StreamWriter,
    rollout_results: List[RolloutResult],
) -> None:
    for result in rollout_results:
        audit_writer.write(make_rollout_audit_record(result))


def stamp_rollout_metadata(
    rollout_result: RolloutResult,
    scheduler_name: str,
    group_id: int,
    rollout_index: int,
    admission_model_version: int,
) -> None:
    boundary_failure = isinstance(rollout_result.audit.get("boundary_failure"), dict)
    if rollout_result.model_version is None and not boundary_failure:
        rollout_result.model_version = admission_model_version

    full_group_id = f"{scheduler_name}_{group_id}"
    rollout_result.group_id = full_group_id
    rollout_result.audit.setdefault("model_version", rollout_result.model_version)
    rollout_result.audit["admission_model_version"] = admission_model_version
    rollout_result.audit["group_id"] = full_group_id
    rollout_result.audit["rollout_index"] = rollout_index
    for step_index, sample in enumerate(rollout_result.training_texts):
        if sample.metadata.get("model_version") is None:
            sample.metadata["model_version"] = (
                rollout_result.model_version
                if rollout_result.model_version is not None
                else admission_model_version
            )
        sample.metadata["rollout_index"] = rollout_index
        sample.metadata["step_index"] = step_index
        sample.group_id = full_group_id


def apply_group_boundary_policy(
    rollout_results: List[RolloutResult],
    group_drop_counts: Dict[str, int],
    *,
    enters_training: bool = True,
) -> str | None:
    failures = []
    for result in rollout_results:
        failure = result.audit.get("boundary_failure")
        if not isinstance(failure, dict):
            continue
        reason = failure.get("reason")
        rollout_index = result.audit.get("rollout_index")
        if not isinstance(reason, str) or not isinstance(rollout_index, int):
            raise ValueError("Boundary failure audit must contain a reason and rollout_index")
        failures.append((rollout_index, reason, result))

    if not failures:
        for result in rollout_results:
            result.audit["published"] = bool(result.training_texts)
            result.audit["entered_training"] = enters_training and bool(result.training_texts)
        return None

    trigger_index, reason, trigger = min(failures, key=lambda failure: failure[0])
    group_drop_counts[reason] = group_drop_counts.get(reason, 0) + 1
    for result in rollout_results:
        result.audit["published"] = False
        result.audit["entered_training"] = False
        result.audit["drop_reason"] = reason
        result.audit["group_drop_reason"] = reason
        result.audit["group_drop_trigger_rollout_index"] = trigger_index
        result.audit["group_drop_triggered_here"] = result is trigger
    return reason


def _set_group_boundary_failure(
    rollout_results: List[RolloutResult],
    reason: str,
    detail: str,
    **failure_fields,
) -> None:
    indexed_results = [
        (result.audit.get("rollout_index"), result)
        for result in rollout_results
    ]
    if not indexed_results or not all(
        isinstance(rollout_index, int) for rollout_index, _ in indexed_results
    ):
        raise ValueError("Atomic group members must have integer rollout_index values")
    _, trigger = min(indexed_results, key=lambda item: item[0])
    trigger.audit["boundary_failure"] = {
        "reason": reason,
        "detail": detail,
        **failure_fields,
    }
    trigger.audit["drop_reason"] = reason


def put_rollout_group(
    result_queue: SharedMemoryQueue,
    rollout_results: List[RolloutResult],
) -> None:
    try:
        result_queue.put(rollout_results)
    except EntrySizeExceeded as exc:
        if not rollout_results or not all(
            result.atomic_group for result in rollout_results
        ):
            raise
        _set_group_boundary_failure(
            rollout_results,
            "actor_queue_oversize",
            "Atomic rollout group exceeds the actor result-queue entry cap",
            queue_hop="actor_result",
            serialized_size=exc.size,
            max_size=exc.max_size,
        )
        for result in rollout_results:
            result.audit.setdefault(
                "prompt_token_lengths",
                [text.prompt_tokens for text in result.training_texts],
            )
            result.audit.setdefault(
                "output_token_lengths",
                [text.output_tokens for text in result.training_texts],
            )
            result.audit.setdefault(
                "n_training_texts_before_drop",
                len(result.training_texts),
            )
            result.training_texts = []
        result_queue.put(rollout_results)


def validate_atomic_group_admission(
    rollout_results: List[RolloutResult],
    attempts: int,
    samples_per_update: int,
    ready_capacity: int,
) -> bool:
    atomic_values = {result.atomic_group for result in rollout_results}
    if len(atomic_values) != 1:
        raise ValueError("Rollout group mixes atomic and legacy members")
    is_atomic = atomic_values.pop()
    if not is_atomic:
        return False
    if any(
        isinstance(result.audit.get("boundary_failure"), dict)
        for result in rollout_results
    ):
        return True

    group_ids = {result.group_id for result in rollout_results}
    domains = {result.domain for result in rollout_results}
    rollout_indices = [result.audit.get("rollout_index") for result in rollout_results]
    entry_count = sum(len(result.training_texts) for result in rollout_results)
    failures = []
    if len(group_ids) != 1 or None in group_ids:
        failures.append(
            f"expected one non-null group_id, got "
            f"{sorted(str(value) for value in group_ids)}"
        )
    if len(domains) != 1:
        failures.append(
            f"expected one domain, got {sorted(str(value) for value in domains)}"
        )
    if (
        len(rollout_indices) != attempts
        or any(not isinstance(index, int) for index in rollout_indices)
        or len(set(rollout_indices)) != attempts
    ):
        failures.append(
            f"expected {attempts} distinct rollout indices, got {rollout_indices}"
        )
    if any(len(result.training_texts) != 1 for result in rollout_results):
        failures.append("each atomic rollout must contribute exactly one training sample")
    if entry_count > samples_per_update:
        failures.append(
            f"{entry_count} entries exceed samples_per_optimizer_step={samples_per_update}"
        )
    if entry_count > ready_capacity:
        failures.append(
            f"{entry_count} entries exceed ring_buffer_size={ready_capacity}"
        )

    if failures:
        _set_group_boundary_failure(
            rollout_results,
            "atomic_envelope_rejected",
            "; ".join(failures),
            samples_per_update=samples_per_update,
            ready_capacity=ready_capacity,
            entry_count=entry_count,
        )
    return True


def make_training_group_envelope(
    rollout_results: List[RolloutResult],
    attempts: int,
) -> TrainingGroupEnvelope:
    entries = [
        training_text.model_dump()
        for result in rollout_results
        for training_text in result.training_texts
    ]
    entries.sort(
        key=lambda entry: (
            entry["metadata"]["rollout_index"],
            entry["metadata"]["step_index"],
        )
    )
    group_ids = {entry["group_id"] for entry in entries}
    domains = {result.domain for result in rollout_results}
    assert len(group_ids) == 1
    assert len(domains) == 1
    return TrainingGroupEnvelope(
        group_id=group_ids.pop(),
        domain=domains.pop(),
        expected_rollouts=attempts,
        entries=entries,
    )


def actor_result_payload_size(
    rollout_results: List[RolloutResult],
) -> int:
    attempted_sizes = [
        failure["serialized_size"]
        for result in rollout_results
        if isinstance(
            (failure := result.audit.get("boundary_failure")),
            dict,
        )
        and failure.get("queue_hop") == "actor_result"
        and isinstance(failure.get("serialized_size"), int)
    ]
    if attempted_sizes:
        return max(attempted_sizes)
    return len(pickle.dumps(rollout_results))


def make_calibration_group_evidence(
    rollout_results: List[RolloutResult],
    *,
    attempts: int,
    actor_result_bytes: int,
    training_envelope_bytes: int | None,
    drop_reason: str | None,
) -> CalibrationGroupEvidence:
    group_ids = {result.group_id for result in rollout_results}
    datasets = {result.dataset_name for result in rollout_results}
    task_ids = {result.audit.get("task_id") for result in rollout_results}
    if (
        len(group_ids) != 1
        or None in group_ids
        or len(datasets) != 1
        or None in datasets
        or len(task_ids) != 1
        or None in task_ids
    ):
        raise ValueError(
            "Calibration group identifiers are inconsistent"
        )

    model_calls = [
        call
        for result in rollout_results
        for call in result.audit.get("model_calls", [])
        if isinstance(call, dict)
    ]
    reasons = {
        failure.get("reason")
        for result in rollout_results
        if isinstance(
            (failure := result.audit.get("boundary_failure")),
            dict,
        )
    }
    return CalibrationGroupEvidence(
        group_id=str(group_ids.pop()),
        dataset=str(datasets.pop()),
        task_id=str(task_ids.pop()),
        attempted_rollouts=attempts,
        published_rollouts=(
            0
            if drop_reason is not None
            else sum(
                len(result.training_texts)
                for result in rollout_results
            )
        ),
        actor_result_bytes=actor_result_bytes,
        training_envelope_bytes=training_envelope_bytes,
        call_prefix_tokens=[
            int(call["token_start"])
            for call in model_calls
        ],
        generation_tokens=[
            int(call["token_end"])
            - int(call["token_start"])
            for call in model_calls
        ],
        merged_sequence_tokens=[
            int(result.audit["sequence_tokens"])
            for result in rollout_results
            if isinstance(
                result.audit.get("sequence_tokens"),
                int,
            )
        ],
        drop_reason=drop_reason,
        endpoint_affinity_failures=sum(
            reason
            in {
                "policy_endpoint_mismatch",
                "endpoint_affinity",
            }
            for reason in reasons
        ),
        version_transport_failures=sum(
            reason
            in {
                "incomplete_policy_capture",
                "invalid_model_version_start",
                "invalid_model_version_end",
                "model_version_decreased",
            }
            for reason in reasons
        ),
        silent_truncations=sum(
            bool(result.audit.get("silent_truncation"))
            for result in rollout_results
        ),
    )


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

    final_steps = calculate_train_steps(cfg.finetune, cfg.finetune.interrupt_train_steps)
    samples_target = final_steps * cfg.finetune.train_batch_size * cfg.finetune.gradient_accumulation_passes
    retryable_rollout_exceptions = (
        aiohttp.ServerTimeoutError,
        asyncio.TimeoutError,
        TimeoutError,
        RetryableAbortedCompletionError,
    )
    max_rollout_retries = int(getattr(cfg.actor, "max_rollout_retries", -1))  # -1 means infinite retries
    retry_initial_delay_s = float(getattr(cfg.actor, "rollout_retry_initial_delay_s", 1.0))
    retry_max_delay_s = float(getattr(cfg.actor, "rollout_retry_max_delay_s", 30.0))

    def is_trainer_finished() -> bool:
        return (
            trainer_state.samples_processed is not None
            and trainer_state.samples_processed >= samples_target
        )

    def handle_rollout_exception(exc: Exception):
        if isinstance(exc, retryable_rollout_exceptions) and is_trainer_finished():
            logger.info(
                f"{scheduler_name}: rollout task encountered {exc.__class__.__name__} after trainer completion; ignoring"
            )
            return
        logger.error("Exception in rollout, stop all other rollout tasks", exc_info=exc)
        current_task = asyncio.current_task(loop=loop)
        for task in asyncio.all_tasks(loop=loop):
            if task != current_task:
                task.cancel()
        result_queue.put(exc)
        logger.error("Stopped all tasks and put exception in the result queue")

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
            retry_count = 0
            while True:
                try:
                    rollout_result = await rollout_policy(cfg, llm, problem, session)
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    is_retryable = isinstance(exc, retryable_rollout_exceptions)
                    can_retry = max_rollout_retries < 0 or retry_count < max_rollout_retries
                    if is_retryable and can_retry and not is_trainer_finished():
                        retry_count += 1
                        backoff_s = min(retry_max_delay_s, retry_initial_delay_s * (2 ** (retry_count - 1)))
                        if retry_count == 1 or retry_count % 10 == 0:
                            logger.warning(
                                f"{scheduler_name}: rollout {group_id}/{rollout_index} failed with "
                                f"{exc.__class__.__name__}, retry {retry_count}"
                            )
                        await asyncio.sleep(backoff_s)
                        continue
                    handle_rollout_exception(exc)
                    return
            stamp_rollout_metadata(
                rollout_result,
                scheduler_name,
                group_id,
                rollout_index,
                model_version,
            )
            group_rollouts[group_id].append(rollout_result)
            if len(group_rollouts[group_id]) == attempts:
                # This is blocking call, but there's just one other thread reading from this queue.
                random.shuffle(group_rollouts[group_id])
                put_rollout_group(result_queue, group_rollouts[group_id])
                del group_rollouts[group_id]
            finished_rollouts += 1
        except Exception as e:
            handle_rollout_exception(e)
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
            if is_trainer_finished():
                logger.info(f"{scheduler_name}: trainer signalled completion; stopping rollout scheduler")
                break
            if time.time() - last_logged > 10.0 and sum(active_rollouts):
                logger.info(
                    f"{scheduler_name}: "
                    f"rollouts in progress: {sum(active_rollouts)}, "
                    f"groups in progress: {len(group_rollouts)}, "
                    f"rollouts started so far: {started_rollouts}, "
                    f"rollouts finished so far: {finished_rollouts}, "
                    f"groups started so far: {group_id}, "
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
        audit_stream: StreamSpec,
        trainer_state: TrainerState,
        is_training: bool = True,
    ) -> None:
        self.data_stream = data_stream
        self.trainer_state = trainer_state
        self.stats_stream = stats_stream
        self.audit_stream = audit_stream
        self.sliding_aggregator = SlidingWindowAggregator(window_size=cfg.actor.throughput_window_size)
        self.llms = llms
        self.loop_start_time = -1
        self.cfg = cfg
        self.is_training = is_training
        self.is_scheduling_paused = False
        self.debug_mode = bool(cfg.debug.mode)

        # Determine the number of processes to use
        num_processes = min(self.cfg.actor.rollout_workers, len(self.llms))
        self._attempts = self.cfg.attempts if is_training else 1

        # Divide LLMs approximately equally across processes
        self._llm_groups = [[] for _ in range(num_processes)]
        for i, llm in enumerate(self.llms):
            self._llm_groups[i % num_processes].append((i, llm))

        self.smm = SharedMemoryManager()
        self.smm.start()

        # Use SharedMemoryQueue instead of separate problem_queue, result_queue, and io_buffer
        self.problem_queue = SharedMemoryQueue(self.smm, self.cfg.actor.problem_queue_size, cfg.actor.shared_memory_entry_size)
        self.result_queue = SharedMemoryQueue(self.smm, self.cfg.actor.result_queue_size, cfg.actor.shared_memory_entry_size)

        logger.info(f"Initialized {'train' if self.is_training else 'test'} actor loop")
        logger.info(f"Problem queue size: {self.problem_queue.max_size}, result queue size: {self.result_queue.max_size}")
        logger.info(f"Result queue buffer size: {self.result_queue.get_memory_size() / 2**30} Gb")

        self.rollout_processes = []
        if is_training:
            self._start_rollout_processes()

    def _start_rollout_processes(self):
        for llm_group in self._llm_groups:
            assert llm_group
            llm_idxs = [llm[0] for llm in llm_group]
            llms = [llm[1] for llm in llm_group]
            scheduler_name = (
                f"{'train' if self.is_training else 'test'} scheduler for llms {','.join([str(i) for i in llm_idxs])}"
            )
            process = mp.Process(
                target=rollout_maker_entrypoint,
                args=(self.cfg, self._attempts, self.problem_queue, self.result_queue, llms, scheduler_name),
            )
            process.start()
            self.rollout_processes.append(process)
        logger.info(f"Started {len(self.rollout_processes)} rollout processes")

    def _stop_rollout_processes(self):
        for p in self.rollout_processes:
            p.terminate()
        for p in self.rollout_processes:
            p.join(timeout=10)
        logger.info(f"Stopped {len(self.rollout_processes)} rollout processes")
        self.rollout_processes = []

    def init_stats(self):
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.latency_list = []
        self.model_versions_list = []
        self.sliding_stats = defaultdict(list)
        self.domain_counts = defaultdict(int)
        self.group_drop_counts = defaultdict(int)
        self.dataset_to_domain: Dict[str, str] = {}
    
    def compute_domain_agnostic_metrics(self, result: RolloutResult) -> Dict[str, float]:
        metrics = {}
        
        metrics['overflow'] = rollout_has_overflow(result.training_texts)
        metrics['num_turns'] = len(result.training_texts)
        metrics['prompt_tokens'] = [training_text.prompt_tokens for training_text in result.training_texts]
        metrics['output_tokens'] = [training_text.output_tokens for training_text in result.training_texts]
        metrics['penalty_delta'] = getattr(result, '_penalty_delta', 0.0)
        if self.is_training:
            max_tokens = self.cfg.llm.parameters.get("max_tokens", None)
        else:
            test_llm_cfg = getattr(self.cfg, "test_llm", None)
            if test_llm_cfg and getattr(test_llm_cfg, "parameters", None) is not None:
                max_tokens = test_llm_cfg.parameters.get("max_tokens", None)
            else:
                max_tokens = self.cfg.llm.parameters.get("max_tokens", None)
        if max_tokens is not None:
            is_overlong = any(training_text.output_tokens >= max_tokens for training_text in result.training_texts)
            metrics['overlong'] = is_overlong
            metrics['overlong_success'] = is_overlong and result.metrics.success
        
        return metrics

    def update_stats(self, rollout_results: List[RolloutResult]):
        for result in rollout_results:
            assert isinstance(result.metrics, BaseMetrics), "Metrics should be an instance of BaseMetrics"
            dataset_name = result.dataset_name
            group_id = result.group_id
            self.latency_list.append(result.latency)
            if result.model_version is not None:
                self.model_versions_list.append(result.model_version)
            domain_key: str | None = None
            if getattr(result, "domain", None):
                domain_key = str(result.domain)
            elif isinstance(dataset_name, str):
                domain_key = dataset_name.split("@", 1)[0]
            elif dataset_name is not None:
                domain_key = str(dataset_name)

            if domain_key:
                if result.audit.get("published", bool(result.training_texts)):
                    self.domain_counts[domain_key] += len(result.training_texts)
                if dataset_name is not None:
                    self.dataset_to_domain[str(dataset_name)] = domain_key
            domain_agnostic_metrics = self.compute_domain_agnostic_metrics(result) 
            all_metrics = result.metrics.model_dump() | domain_agnostic_metrics
            for k, v in all_metrics.items():
                if isinstance(v, list):
                    self.stats[k][dataset_name][group_id] += v
                elif isinstance(v, float) | isinstance(v, bool) | isinstance(v, int):
                    self.stats[k][dataset_name][group_id].append(v)
                else:
                    raise ValueError(f"Unsupported metric type: {type(v)} for key {k}")
        
        prompt_length_tokens = [
            training_text.prompt_tokens
            for result in rollout_results
            if result.audit.get("published", bool(result.training_texts))
            for training_text in result.training_texts
        ]
        output_length_tokens = [
            training_text.output_tokens
            for result in rollout_results
            if result.audit.get("published", bool(result.training_texts))
            for training_text in result.training_texts
        ]
        self.sliding_aggregator.update(prompt_length_tokens, output_length_tokens)
        sliding_window_stats = self.sliding_aggregator.get_stats()
        if sliding_window_stats is not None:
            for k, v in sliding_window_stats.items():
                self.sliding_stats[k].append(v)
        


    def run(self, dataset: list[tuple[str, dict]]):
        if not self.rollout_processes:
            self._start_rollout_processes()
        try:
            yield from self._run(dataset)
        finally:
            if not self.is_training:
                self._stop_rollout_processes()

    def _run(self, dataset: list[tuple[str, dict]]):
        loop_start_time = time.time()
        self.init_stats()

        attempts = self.cfg.attempts if self.is_training else 1
        calibration = self.is_training and prerun_enabled(self.cfg)
        published_samples = 0
        submitted_groups = 0
        finished_groups = 0
        trainer_version_to_publish = None

        domain_sampler = None
        if calibration:
            if attempts != CALIBRATION_GROUP_SIZE:
                raise ValueError("Tau2 pre-run calibration requires G16")
            samples_per_update = samples_per_optimizer_step(self.cfg.finetune)
            if samples_per_update != 192:
                raise ValueError(
                    "Tau2 pre-run calibration requires exactly "
                    "192 samples per optimizer step"
                )
            selected_problems = select_calibration_problems(dataset)
            expected_rollouts = len(selected_problems)
            problem_iter = sequential_iter(selected_problems)
        elif self.is_training:
            expected_rollouts = -1
            problem_iter = random_iter(dataset)
            domain_mix_cfg = getattr(self.cfg.actor, "domain_mix", None)
            if domain_mix_cfg:
                mix_weights = OmegaConf.to_container(domain_mix_cfg, resolve=True)
                if not isinstance(mix_weights, dict):
                    raise ValueError("actor.domain_mix must be a mapping from domain to weight")
                domain_sampler = DomainWeightedSampler(dataset, mix_weights)
        else:
            expected_rollouts = len(dataset)
            problem_iter = sequential_iter(dataset)
        if expected_rollouts > 0:
            logger.info(f"Will stop after {expected_rollouts} rollouts")
        assert self.trainer_state.propagated_weight_version is not None
        dap_cfg = getattr(self.cfg.actor, "difficulty_aware_penalty", None)
        if dap_cfg and dap_cfg.enabled:
            assert dap_cfg.gamma >= 0, (
                f"difficulty_aware_penalty.gamma must be >= 0, got {dap_cfg.gamma}"
            )
            failure_scale = getattr(dap_cfg, "failure_scale", 1.0)
            assert 0 <= failure_scale <= 1, (
                f"difficulty_aware_penalty.failure_scale must be in [0, 1], got {failure_scale}"
            )

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
            write_to_streams(self.audit_stream, "a") as audit_writer,
        ):
            while True:
                # the user function must do next(...) to run each iteration
                yield

                final_steps = calculate_train_steps(self.cfg.finetune, self.cfg.finetune.interrupt_train_steps)
                samples_target = final_steps * self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
                if self.trainer_state.samples_processed is not None and self.trainer_state.samples_processed >= samples_target:
                    logger.info("Trainer signalled completion; stopping actor loop")
                    break

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
                                    if domain_sampler is not None:
                                        problem = domain_sampler.sample()
                                    else:
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
                assert len(rollout_results) == attempts, (
                    f"Expected {attempts} rollouts, got {len(rollout_results)}"
                )
                actor_result_bytes = (
                    actor_result_payload_size(rollout_results)
                    if calibration
                    else None
                )
                atomic_group = validate_atomic_group_admission(
                    rollout_results,
                    attempts,
                    samples_per_optimizer_step(self.cfg.finetune),
                    int(self.cfg.preprocess.ring_buffer_size),
                )
                group_drop_reason = apply_group_boundary_policy(
                    rollout_results,
                    self.group_drop_counts,
                    enters_training=self.is_training,
                )
                group_samples = (
                    0
                    if group_drop_reason is not None
                    else sum(len(r.training_texts) for r in rollout_results)
                )
                write_rollout_audit_records(audit_writer, rollout_results)

                # Track completions per domain for adaptive sampling
                if domain_sampler is not None:
                    for r in rollout_results:
                        if r.domain:
                            domain_sampler.record_completion(r.domain)

                # --- Difficulty-aware length penalty adjustment ---
                # Reduces the overlong penalty for SUCCESSFUL rollouts on hard problems,
                # so the model can reason longer when it actually leads to solving the problem.
                # Failed overlong rollouts keep the full penalty (failure_scale=1.0)
                # to discourage degenerate long generation that doesn't lead to solutions.
                # Hard cap guard: sequences that hit max_tokens without finishing always
                # get full penalty, even if the rollout is marked successful.
                dap_cfg = getattr(self.cfg.actor, "difficulty_aware_penalty", None)
                max_tokens = self.cfg.llm.parameters.get("max_tokens", None)
                if (
                    group_drop_reason is None
                    and self.is_training
                    and dap_cfg
                    and dap_cfg.enabled
                    and self.cfg.rewards.buffer_tokens > 0
                    and max_tokens is not None
                ):
                    group_solve_rate = sum(r.metrics.success for r in rollout_results) / len(rollout_results)
                    gamma = dap_cfg.gamma
                    failure_scale = getattr(dap_cfg, "failure_scale", 1.0)
                    success_scale = group_solve_rate ** gamma
                    buffer_tokens = self.cfg.rewards.buffer_tokens

                    for r in rollout_results:
                        rollout_scale = success_scale if r.metrics.success else failure_scale
                        metrics_delta = 0.0
                        for text in r.training_texts:
                            # Hard cap guard: if the sequence hit max_tokens without
                            # finishing, always apply full penalty regardless of success
                            if text.output_tokens >= max_tokens and not text.finished:
                                scale = 1.0
                            else:
                                scale = rollout_scale
                            original_penalty = length_penalty(max_tokens, text.output_tokens, buffer_tokens)
                            adjusted_penalty = original_penalty * scale
                            penalty_delta = adjusted_penalty - original_penalty
                            text.reward += penalty_delta
                            metrics_delta += penalty_delta
                        r.metrics.reward += metrics_delta
                        r._penalty_delta = metrics_delta
                else:
                    for r in rollout_results:
                        r._penalty_delta = 0.0

                published_samples += group_samples
                samples_in_queue = self.result_queue.qsize() * attempts
                in_progress = submitted_groups - finished_groups
                training_envelope_bytes = None
                if group_drop_reason is None:
                    if atomic_group:
                        group_envelope = make_training_group_envelope(
                            rollout_results,
                            attempts,
                        )
                        if calibration:
                            training_envelope_bytes = len(pickle.dumps(group_envelope))
                        data_stream_writer.write(group_envelope)
                    else:
                        all_text_dumps = []
                        for r in rollout_results:
                            for text in r.training_texts:
                                all_text_dumps.append(text.model_dump())
                        data_stream_writer.write(all_text_dumps)
                    logger.info(
                        f"Published {group_samples} {'train' if self.is_training else 'test'} samples"
                        f" to {self.data_stream}, total {published_samples} samples so far, "
                        f"{samples_in_queue} samples in the result queue, {in_progress} groups in progress"
                    )
                else:
                    logger.warning(
                        f"Dropped group {rollout_results[0].group_id} at the actor boundary: "
                        f"{group_drop_reason}"
                    )

                if calibration:
                    assert actor_result_bytes is not None
                    append_evidence_record(
                        evidence_directory(self.cfg),
                        "calibration_groups",
                        make_calibration_group_evidence(
                            rollout_results,
                            attempts=attempts,
                            actor_result_bytes=actor_result_bytes,
                            training_envelope_bytes=training_envelope_bytes,
                            drop_reason=group_drop_reason,
                        ),
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
                            "result_queue_max_entry_size_bytes": (
                                self.result_queue.max_actual_entry_size()
                            ),
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

        # skip intermediate metrics that are not meaningful on their own and keep the dashboard clean  
        _hidden_metrics = {"overlong_success"}

        stats = defaultdict(float)
        for metric_name, dict_of_stats_per_metric in self.stats.items():
            if metric_name in _hidden_metrics:
                continue
            for agg, group_stats in calculate_stats(dict_of_stats_per_metric).items():
                stats[f"{split_name}{metric_name}_{agg}"] = group_stats

            domain_groups: Dict[str, Dict] = defaultdict(dict)
            for dataset_name, list_of_stats_per_metric_and_dataset in self.stats[metric_name].items():
                for agg, sub_stats in calculate_stats(list_of_stats_per_metric_and_dataset).items():
                    stats[f"{dataset_name}/{metric_name}_{agg}"] = sub_stats
                # Group datasets by domain (using mapping built in update_stats)
                domain = self.dataset_to_domain.get(str(dataset_name)) if dataset_name is not None else None
                if domain:
                    domain_groups[domain].update(
                        {(dataset_name, gid): vals for gid, vals in list_of_stats_per_metric_and_dataset.items()}
                    )
            for domain, grouped in domain_groups.items():
                for agg, agg_val in calculate_stats(grouped).items():
                    stats[f"{domain}/{metric_name}_{agg}"] = agg_val

        # compute success_given_overlong from raw stats
        overlong_stats = self.stats.get("overlong", {})
        overlong_success_stats = self.stats.get("overlong_success", {})
        # global
        overlong_list = calculate_stats(overlong_stats).get("mean")
        overlong_success_list = calculate_stats(overlong_success_stats).get("mean")
        if overlong_list and overlong_success_list and overlong_list > 0:
            stats[f"{split_name}success_given_overlong"] = overlong_success_list / overlong_list
        # per-dataset
        for ds in overlong_stats:
            if ds is None:
                continue
            ds_overlong = calculate_stats(overlong_stats[ds]).get("mean")
            ds_overlong_success = calculate_stats(overlong_success_stats.get(ds, {})).get("mean")
            if ds_overlong and ds_overlong_success and ds_overlong > 0:
                stats[f"{ds}/success_given_overlong"] = ds_overlong_success / ds_overlong

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
        for reason, count in self.group_drop_counts.items():
            stats[f"{split_name}group_drop/{reason}"] = count

        total_domain_samples = sum(self.domain_counts.values())
        if total_domain_samples:
            for domain, count in sorted(self.domain_counts.items()):
                stats[f"{split_name}domain_mix_count/{domain}"] = count
                stats[f"{split_name}domain_mix_actual/{domain}"] = count / total_domain_samples

        domain_mix_cfg = getattr(self.cfg.actor, "domain_mix", None)
        if domain_mix_cfg:
            mix_weights = OmegaConf.to_container(domain_mix_cfg, resolve=True)
            if isinstance(mix_weights, dict):
                target_total = sum(float(v) for v in mix_weights.values() if float(v) > 0)
                if target_total > 0:
                    for domain, weight in mix_weights.items():
                        stats[f"{split_name}domain_mix_target/{domain}"] = float(weight) / target_total
                else:
                    for domain in mix_weights:
                        stats[f"{split_name}domain_mix_target/{domain}"] = 0.0

        for k, v in self.sliding_stats.items():
            stats[k] = sum(v) / len(v) if v else 0
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
    audit_stream = SingleStreamSpec(exp_path=exp_path, topic="rollout_audit")
    test_audit_stream = SingleStreamSpec(exp_path=exp_path, topic="rollout_audit_test")

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
            collect_logprobs=True,
            chat_template_kwargs=cfg.llm.get("chat_template_kwargs", {}),
        )
        for url in llm_urls
    ]
    test_llms = [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_model_path),
            parameters=cfg.test_llm.parameters,
            collect_logprobs=True,
            chat_template_kwargs=cfg.test_llm.get("chat_template_kwargs", {}),
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
        data_stream=data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=stats_stream,
        audit_stream=audit_stream,
        llms=train_llms,
    )
    train_loop_run = train_loop.run(
        dataset=train_dataset,
    )
    test_loop = ActorLoop(
        data_stream=test_data_stream,
        cfg=cfg,
        trainer_state=trainer_state,
        stats_stream=test_stats_stream,
        audit_stream=test_audit_stream,
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
        try:
            _ = next(train_loop_run)
        except StopIteration:
            logger.info("Train loop finished")
            break
