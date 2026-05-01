from __future__ import annotations

import logging
import math
import random
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ray

from cube_pipelinerl.domain import CubeBenchmarkActor
from cube_pipelinerl.ray_worker_logging import CubeRayWorkerLogCollector
from cube_pipelinerl.utils import check_local_cube_actor_resources
from pipelinerl.metrics import SlidingWindowAggregator

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from pipelinerl.llm import TrainableLLM
    from pipelinerl.rollouts import RolloutResult
    from pipelinerl.streams import StreamWriter
    from pipelinerl.state import TrainerState

logger = logging.getLogger(__name__)

def BREAKPOINT():
    import pdb;pdb.set_trace()

@dataclass
class PendingRollout:
    group_id: int
    task_id: str
    rollout_index: int
    llm_index: int
    benchmark_actor_index: int
    model_version: int


def _calculate_train_steps(finetune_cfg: DictConfig, interrupt_train_steps: int) -> int:
    if interrupt_train_steps == -1:
        assert finetune_cfg.interrupt_train_steps <= finetune_cfg.max_train_steps
        return (
            finetune_cfg.max_train_steps
            if finetune_cfg.interrupt_train_steps < 0
            else finetune_cfg.interrupt_train_steps
        )
    assert interrupt_train_steps <= finetune_cfg.max_train_steps
    return interrupt_train_steps


def _samples_target(cfg: DictConfig) -> int:
    final_steps = _calculate_train_steps(cfg.finetune, cfg.finetune.interrupt_train_steps)
    return final_steps * cfg.finetune.train_batch_size * cfg.finetune.gradient_accumulation_passes


def _is_trainer_finished(cfg: DictConfig, trainer_state: TrainerState) -> bool:
    return trainer_state.samples_processed is not None and trainer_state.samples_processed >= _samples_target(cfg)


def _log_ray_cpu_capacity(owner_name: str, required_num_cpus: int) -> None:
    if required_num_cpus < 1 or not ray.is_initialized():
        return
    cluster_cpu_capacity = float(ray.cluster_resources().get("CPU", 0.0))
    if cluster_cpu_capacity + 1e-6 < float(required_num_cpus):
        logger.warning(
            "%s: Ray cluster CPU capacity %.2f is lower than required %.2f for cube benchmark actors",
            owner_name,
            cluster_cpu_capacity,
            float(required_num_cpus),
        )
    else:
        logger.info(
            "%s: Ray cluster CPU capacity %.2f satisfies required %.2f for cube benchmark actors",
            owner_name,
            cluster_cpu_capacity,
            float(required_num_cpus),
        )


def _init_ray_runtime(cfg: DictConfig, owner_name: str, min_num_cpus: int = 1) -> bool:
    if ray.is_initialized():
        _log_ray_cpu_capacity(owner_name, required_num_cpus=min_num_cpus)
        return False

    log_to_driver = bool(getattr(cfg.actor, "ray_log_to_driver", False))
    ray_address = getattr(cfg.actor, "ray_address", None)
    if ray_address:
        ray.init(
            address=str(ray_address),
            ignore_reinit_error=True,
            log_to_driver=False,
            local_mode=cfg.ray_local_mode,
        )
        logger.info("%s: connected to ray at configured address %s", owner_name, ray_address)
        _log_ray_cpu_capacity(owner_name, required_num_cpus=min_num_cpus)
        return True

    try:
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            log_to_driver=False,
            local_mode=cfg.ray_local_mode,
        )
        logger.info("%s: connected to existing auto-discovered ray cluster", owner_name)
        _log_ray_cpu_capacity(owner_name, required_num_cpus=min_num_cpus)
    except Exception:
        configured_ray_num_cpus = getattr(cfg.actor, "ray_num_cpus", None)
        ray_num_cpus = min_num_cpus if configured_ray_num_cpus is None else int(configured_ray_num_cpus)
        if ray_num_cpus < min_num_cpus:
            logger.warning(
                "%s: actor.ray_num_cpus=%d is lower than required %d; using %d for local ray runtime",
                owner_name,
                ray_num_cpus,
                min_num_cpus,
                min_num_cpus,
            )
            ray_num_cpus = min_num_cpus
        ray.init(
            num_cpus=ray_num_cpus,
            include_dashboard=False,
            ignore_reinit_error=True,
            log_to_driver=False,
            local_mode=cfg.ray_local_mode,
        )
        logger.info("%s: started local ray runtime with %d CPUs", owner_name, ray_num_cpus)

    return True


def _launch_cube_benchmark_actors(
    cfg: DictConfig,
    instances: int,
    ray_worker_log_collector: Any | None = None,
) -> list[Any]:
    from omegaconf import OmegaConf

    if instances < 1:
        raise ValueError("cube benchmark actor instance count must be >= 1")

    params = cfg.get("cube_params", None)
    if params is None:
        raise ValueError("cube actor launcher requires cube_params")

    benchmark_cfg = OmegaConf.to_container(params.benchmark, resolve=True)
    agent_cfg = OmegaConf.to_container(params.agent, resolve=True)
    if not isinstance(benchmark_cfg, dict) or not isinstance(agent_cfg, dict):
        raise ValueError("cube_params.benchmark and .agent must resolve to dictionaries")

    cube_name = getattr(params, "name", "default")
    seed = int(getattr(params, "seed", cfg.seed))
    actor_num_cpus = float(getattr(cfg.actor, "cube_actor_num_cpus", 1.0))
    worker_log_level = str(getattr(cfg.actor, "ray_worker_log_level", "ERROR"))
    worker_litellm_log_level = str(getattr(cfg.actor, "ray_worker_litellm_log_level", "WARNING"))

    actors = []
    for idx in range(instances):
        actor_name = f"CUBE_{cube_name}_{idx}"
        actor = CubeBenchmarkActor.options(num_cpus=actor_num_cpus).remote(
            benchmark_cfg=benchmark_cfg,
            agent_cfg=agent_cfg,
            cube_name=cube_name,
            train_dataset_names=cfg.train_dataset_names,
            test_dataset_names=cfg.test_dataset_names,
            seed=seed,
            actor_name=actor_name,
            ray_worker_log_collector=ray_worker_log_collector,
            ray_worker_log_level=worker_log_level,
            litellm_log_level=worker_litellm_log_level,
        )
        actors.append(actor)

    ray.get([actor.setup.remote() for actor in actors])
    return actors


def _wait_for_cube_benchmark_actors(actors: list[Any], timeout_s: float) -> None:
    deadline = time.time() + max(1.0, timeout_s)
    while True:
        states = ray.get([actor.health.remote() for actor in actors])
        if all(state.get("ready", False) for state in states):
            logger.info("All cube benchmark actors are ready")
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for cube benchmark actors: {states}")
        time.sleep(1.0)


def _launch_ray_worker_log_collector(cfg: DictConfig) -> Any | None:
    if not bool(getattr(cfg.actor, "ray_worker_log_enabled", True)):
        return None

    worker_log_path = getattr(cfg.actor, "ray_worker_log_path", None)
    if not worker_log_path:
        worker_log_path = str(Path(cfg.output_dir) / "actor" / "ray_workers.log")

    collector = CubeRayWorkerLogCollector.remote(str(worker_log_path))
    logger.info("Ray worker error logs will be collected in %s", worker_log_path)
    return collector


def _build_llm_kwargs(llm: TrainableLLM) -> dict:
    from omegaconf import DictConfig, OmegaConf

    parameters = llm.parameters
    if isinstance(parameters, DictConfig):
        parameters = OmegaConf.to_container(parameters, resolve=True)
    elif isinstance(parameters, dict):
        parameters = {
            key: OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
            for key, value in parameters.items()
        }

    kwargs = {
        "base_url": llm.base_url,
        "model_name": llm.model_name,
        "tokenizer_name": llm.tokenizer_name,
        "parameters": parameters,
        "collect_logprobs": llm.collect_logprobs,
    }
    served_model_name = getattr(llm, "served_model_name", None)
    if served_model_name:
        kwargs["served_model_name"] = served_model_name
    return kwargs


def _build_train_llms(cfg: DictConfig, llm_urls: list[str], actor_model_path: Path) -> list[TrainableLLM]:
    from pipelinerl.llm import TrainableLLM

    served_model_name = cfg.vllm_config.vllm_kwargs.get("served_model_name") if cfg.vllm_config.vllm_kwargs else None
    return [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_model_path),
            parameters=cfg.llm.parameters,
            collect_logprobs=True,
            served_model_name=served_model_name,
        )
        for url in llm_urls
    ]


def _build_test_llms(cfg: DictConfig, llm_urls: list[str], actor_model_path: Path) -> list[TrainableLLM]:
    from pipelinerl.llm import TrainableLLM

    served_model_name = cfg.vllm_config.vllm_kwargs.get("served_model_name") if cfg.vllm_config.vllm_kwargs else None
    test_parameters = cfg.test_llm.parameters if getattr(cfg, "test_llm", None) else cfg.llm.parameters
    return [
        TrainableLLM(
            base_url=url,
            model_name=str(actor_model_path),
            tokenizer_name=str(actor_model_path),
            parameters=test_parameters,
            collect_logprobs=True,
            served_model_name=served_model_name,
        )
        for url in llm_urls
    ]


def _write_group_result(
    rollout_results: list[RolloutResult],
    attempts: int,
    data_writer: StreamWriter,
) -> int:
    assert len(rollout_results) == attempts, f"Expected {attempts} rollouts, got {len(rollout_results)}"
    payload = [text.model_dump() for result in rollout_results for text in result.training_texts]
    data_writer.write(payload)
    return len(payload)


class CubeActorLoop:
    def __init__(
        self,
        *,
        cfg: DictConfig,
        llms: list[TrainableLLM],
        benchmark_actors: list[Any],
        trainer_state: TrainerState,
        data_writer: StreamWriter,
        stats_writer: StreamWriter,
        scheduler_name: str,
        is_training: bool,
    ) -> None:
        self.cfg = cfg
        self.llms = llms
        self.benchmark_actors = benchmark_actors
        self.trainer_state = trainer_state
        self.data_writer = data_writer
        self.stats_writer = stats_writer
        self.scheduler_name = scheduler_name
        self.is_training = is_training
        self.debug_mode = bool(cfg.debug.mode)
        self.is_scheduling_paused = False

        self.attempts = int(cfg.attempts) if is_training else 1
        self.llm_max_rollouts = int(cfg.actor.llm_max_rollouts)
        if self.llm_max_rollouts < 1:
            raise ValueError("actor.llm_max_rollouts must be >= 1")
        # CubeBenchmarkActor rollout is currently blocking/synchronous, so keep per-actor
        # concurrency at one and scale by actor count.
        self.cube_actor_max_rollouts = 1
        self.cube_actors_per_llm = self.llm_max_rollouts
        expected_benchmark_actors = len(llms) * self.cube_actors_per_llm
        if len(benchmark_actors) != expected_benchmark_actors:
            raise ValueError(
                f"Expected {expected_benchmark_actors} benchmark actors "
                f"({len(llms)} llms * {self.cube_actors_per_llm} actors/llm), got {len(benchmark_actors)}"
            )

        self._llm_to_actor_indices = [
            [llm_idx * self.cube_actors_per_llm + offset for offset in range(self.cube_actors_per_llm)]
            for llm_idx in range(len(self.llms))
        ]
        self.max_pending = max(1, len(llms) * self.llm_max_rollouts)
        self.rollout_timeout_s = getattr(cfg.actor, "rollout_timeout", None)
        if self.rollout_timeout_s is not None:
            self.rollout_timeout_s = float(self.rollout_timeout_s)
        self.llm_kwargs = [_build_llm_kwargs(llm) for llm in llms]
        self.sliding_aggregator = SlidingWindowAggregator(window_size=int(cfg.actor.throughput_window_size))

        self.total_published_samples = 0
        self.total_submitted_groups = 0
        self.total_finished_groups = 0
        self.init_stats()

        self._is_running = False
        self._task_ids: list[str] = []
        self._expected_groups = -1
        self._pending: dict[Any, PendingRollout] = {}
        self._retry_rollouts: deque[tuple[int, str, int]] = deque()
        self._active_rollouts: list[int] = [0] * len(self.llms)
        self._active_rollouts_by_actor: list[int] = [0] * len(self.benchmark_actors)
        self._group_rollouts: dict[int, list[RolloutResult]] = {}
        self._started_rollouts = 0
        self._finished_rollouts = 0
        self._run_submitted_groups = 0
        self._run_finished_groups = 0
        self._group_id = -1
        self._group_rollout_index = self.attempts
        self._current_task_id = ""
        self._next_task_index = 0
        self._loop_start_time = 0.0
        self._last_logged = 0.0
        self._stop_reason = "completed"
        self._last_trainer_version = 0
        self._trainer_version_to_publish: int | None = None

        logger.info(
            "%s: initialized actor pools with %d llms, %d actors/llm, actor max rollouts=%d, llm max rollouts=%d",
            self.scheduler_name,
            len(self.llms),
            self.cube_actors_per_llm,
            self.cube_actor_max_rollouts,
            self.llm_max_rollouts,
        )

    @property
    def is_running(self) -> bool:
        return self._is_running

    def _select_benchmark_actor_index_for_llm(self, llm_index: int) -> int | None:
        actor_indices = self._llm_to_actor_indices[llm_index]
        available_indices = [
            idx for idx in actor_indices if self._active_rollouts_by_actor[idx] < self.cube_actor_max_rollouts
        ]
        if not available_indices:
            return None
        return min(available_indices, key=lambda idx: (self._active_rollouts_by_actor[idx], idx))

    def init_stats(self) -> None:
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.latency_list: list[float] = []
        self.model_versions_list: list[int] = []
        self.sliding_stats = defaultdict(list)
        self.domain_counts = defaultdict(int)
        self.dataset_to_domain: dict[str, str] = {}

    def compute_domain_agnostic_metrics(self, result: RolloutResult) -> dict[str, float | bool | list[int]]:
        from pipelinerl.rollouts import rollout_has_overflow

        metrics: dict[str, float | bool | list[int]] = {
            "overflow": rollout_has_overflow(result.training_texts),
            "num_turns": len(result.training_texts),
            "prompt_tokens": [t.prompt_tokens for t in result.training_texts],
            "output_tokens": [t.output_tokens for t in result.training_texts],
            "penalty_delta": getattr(result, "_penalty_delta", 0.0),
        }
        if self.is_training:
            max_tokens = self.cfg.llm.parameters.get("max_tokens", None)
        else:
            test_llm_cfg = getattr(self.cfg, "test_llm", None)
            if test_llm_cfg and getattr(test_llm_cfg, "parameters", None) is not None:
                max_tokens = test_llm_cfg.parameters.get("max_tokens", None)
            else:
                max_tokens = self.cfg.llm.parameters.get("max_tokens", None)

        if max_tokens is not None:
            is_overlong = any(t.output_tokens >= max_tokens for t in result.training_texts)
            metrics["overlong"] = is_overlong
            metrics["overlong_success"] = is_overlong and result.metrics.success
        return metrics

    def update_stats(self, rollout_results: list[RolloutResult]) -> None:
        from pipelinerl.rollouts import BaseMetrics

        for result in rollout_results:
            assert result.model_version is not None
            assert isinstance(result.metrics, BaseMetrics), "Metrics should be BaseMetrics"
            dataset_name = result.dataset_name
            group_id = result.group_id
            self.latency_list.append(result.latency)
            self.model_versions_list.append(int(result.model_version))

            domain_key: str | None = None
            if getattr(result, "domain", None):
                domain_key = str(result.domain)
            elif isinstance(dataset_name, str):
                domain_key = dataset_name.split("@", 1)[0]
            elif dataset_name is not None:
                domain_key = str(dataset_name)

            if domain_key:
                self.domain_counts[domain_key] += len(result.training_texts)
                if dataset_name is not None:
                    self.dataset_to_domain[str(dataset_name)] = domain_key

            all_metrics = result.metrics.model_dump() | self.compute_domain_agnostic_metrics(result)
            for key, value in all_metrics.items():
                if isinstance(value, list):
                    self.stats[key][dataset_name][group_id] += value
                elif isinstance(value, (float, bool, int)):
                    self.stats[key][dataset_name][group_id].append(value)
                else:
                    raise ValueError(f"Unsupported metric type: {type(value)} for key {key}")

        prompt_tokens = [t.prompt_tokens for result in rollout_results for t in result.training_texts]
        output_tokens = [t.output_tokens for result in rollout_results for t in result.training_texts]
        self.sliding_aggregator.update(prompt_tokens, output_tokens)
        sliding_window_stats = self.sliding_aggregator.get_stats()
        if sliding_window_stats is not None:
            for key, value in sliding_window_stats.items():
                self.sliding_stats[key].append(value)

    def publish_stats(self, loop_stats: dict[str, Any]) -> None:
        from omegaconf import OmegaConf
        from pipelinerl.utils import always_or_never_success_stats, calculate_stats

        split_name = "test_" if not self.is_training else ""
        hidden_metrics = {"overlong_success"}

        stats: dict[str, Any] = defaultdict(float)
        for metric_name, dict_of_stats_per_metric in self.stats.items():
            if metric_name in hidden_metrics:
                continue
            for agg, metric_val in calculate_stats(dict_of_stats_per_metric).items():
                stats[f"{split_name}{metric_name}_{agg}"] = metric_val

            domain_groups: dict[str, dict] = defaultdict(dict)
            for dataset_name, list_of_stats_per_dataset in self.stats[metric_name].items():
                for agg, sub_stats in calculate_stats(list_of_stats_per_dataset).items():
                    stats[f"{dataset_name}/{metric_name}_{agg}"] = sub_stats
                domain = self.dataset_to_domain.get(str(dataset_name)) if dataset_name is not None else None
                if domain:
                    domain_groups[domain].update(
                        {(dataset_name, gid): vals for gid, vals in list_of_stats_per_dataset.items()}
                    )
            for domain, grouped in domain_groups.items():
                for agg, agg_val in calculate_stats(grouped).items():
                    stats[f"{domain}/{metric_name}_{agg}"] = agg_val

        overlong_stats = self.stats.get("overlong", {})
        overlong_success_stats = self.stats.get("overlong_success", {})
        overlong_mean = calculate_stats(overlong_stats).get("mean")
        overlong_success_mean = calculate_stats(overlong_success_stats).get("mean")
        if overlong_mean and overlong_success_mean and overlong_mean > 0:
            stats[f"{split_name}success_given_overlong"] = overlong_success_mean / overlong_mean
        for dataset_name in overlong_stats:
            if dataset_name is None:
                continue
            ds_overlong = calculate_stats(overlong_stats[dataset_name]).get("mean")
            ds_overlong_success = calculate_stats(overlong_success_stats.get(dataset_name, {})).get("mean")
            if ds_overlong and ds_overlong_success and ds_overlong > 0:
                stats[f"{dataset_name}/success_given_overlong"] = ds_overlong_success / ds_overlong

        stats |= (
            {f"{split_name}{k}": v for k, v in always_or_never_success_stats(self.stats["success"]).items()}
            | {f"{split_name}latency_{k}": v for k, v in calculate_stats(self.latency_list).items()}
            | {f"{split_name}model_version_{k}": v for k, v in calculate_stats(self.model_versions_list).items()}
        )
        stats |= loop_stats

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

        for key, value in self.sliding_stats.items():
            stats[key] = sum(value) / len(value) if value else 0.0

        if self.cfg.wandb.use_wandb:
            import wandb

            wandb.log({f"actor/{k}": v for k, v in stats.items()})
        self.stats_writer.write(stats)
        self.init_stats()

    def _submit_one_rollout(self, *, group_id: int, task_id: str, rollout_index: int) -> bool:
        next_llm = self._active_rollouts.index(min(self._active_rollouts))
        if self._active_rollouts[next_llm] >= self.llm_max_rollouts:
            return False

        benchmark_actor_index = self._select_benchmark_actor_index_for_llm(next_llm)
        if benchmark_actor_index is None:
            return False
        benchmark_actor = self.benchmark_actors[benchmark_actor_index]
        model_version = int(self.trainer_state.propagated_weight_version or 0)
        ref = benchmark_actor.rollout.remote(task_id=task_id, llm=self.llm_kwargs[next_llm])
        self._pending[ref] = PendingRollout(
            group_id=group_id,
            task_id=task_id,
            rollout_index=rollout_index,
            llm_index=next_llm,
            benchmark_actor_index=benchmark_actor_index,
            model_version=model_version,
        )
        self._active_rollouts[next_llm] += 1
        self._active_rollouts_by_actor[benchmark_actor_index] += 1
        self._started_rollouts += 1
        return True

    def start(self, *, task_ids: list[str], scheduler_name: str | None = None) -> str:
        if scheduler_name:
            self.scheduler_name = scheduler_name
        if not task_ids:
            logger.info("%s: no tasks available; skipping", self.scheduler_name)
            return "no_tasks"
        if self._is_running:
            return "running"

        assert self.trainer_state.propagated_weight_version is not None
        self._task_ids = task_ids
        self._expected_groups = -1 if self.is_training else len(task_ids)
        self._pending = {}
        self._retry_rollouts = deque()
        self._active_rollouts = [0] * len(self.llms)
        self._active_rollouts_by_actor = [0] * len(self.benchmark_actors)
        self._group_rollouts = {}
        self._started_rollouts = 0
        self._finished_rollouts = 0
        self._run_submitted_groups = 0
        self._run_finished_groups = 0
        self._group_id = -1
        self._group_rollout_index = self.attempts
        self._current_task_id = ""
        self._next_task_index = 0
        self._loop_start_time = time.time()
        self._last_logged = time.time()
        self._stop_reason = "running"
        self._last_trainer_version = int(self.trainer_state.propagated_weight_version)
        self._trainer_version_to_publish = None
        self._is_running = True

        logger.info("Starting %s loop (%s)", "train" if self.is_training else "test", self.scheduler_name)
        return "running"

    def _finish(self, reason: str) -> str:
        self._stop_reason = reason
        if self.is_training and self._trainer_version_to_publish is not None and self.latency_list:
            self.publish_stats(
                {
                    "published_samples": self.total_published_samples,
                    "submitted_groups": self.total_submitted_groups,
                    "finished_groups": self.total_finished_groups,
                    "pending_rollouts": len(self._pending),
                    "active_rollouts": sum(self._active_rollouts),
                    "time_since_start": time.time() - self._loop_start_time,
                    "trainer_model_version": self._trainer_version_to_publish,
                }
            )
            self._trainer_version_to_publish = None

        self._is_running = False
        logger.info(
            "Cube %s loop finished (%s): reason=%s started=%s finished=%s submitted_groups=%s",
            "train" if self.is_training else "test",
            self.scheduler_name,
            reason,
            self._started_rollouts,
            self._finished_rollouts,
            self._run_submitted_groups,
        )
        return reason

    def step(self) -> str:
        from pipelinerl.rollouts import RolloutResult

        if not self._is_running:
            return self._stop_reason

        trainer_finished = _is_trainer_finished(self.cfg, self.trainer_state)
        if self.is_training and int(self.trainer_state.propagated_weight_version or 0) > self._last_trainer_version:
            self._trainer_version_to_publish = self._last_trainer_version
            self._last_trainer_version = int(self.trainer_state.propagated_weight_version or 0)
        
        while len(self._pending) < self.max_pending:
            if self._retry_rollouts:
                group_id, task_id, rollout_index = self._retry_rollouts[0]
                if not self._submit_one_rollout(
                    group_id=group_id,
                    task_id=task_id,
                    rollout_index=rollout_index,
                ):
                    break
                self._retry_rollouts.popleft()
                continue

            if self.is_training:
                if trainer_finished or self.is_scheduling_paused:
                    break
                if self._group_rollout_index == self.attempts:
                    self._group_id += 1
                    self._current_task_id = random.choice(self._task_ids)
                    self._group_rollouts[self._group_id] = []
                    self._group_rollout_index = 0
                    self._run_submitted_groups += 1
                    self.total_submitted_groups += 1
                if not self._submit_one_rollout(
                    group_id=self._group_id,
                    task_id=self._current_task_id,
                    rollout_index=self._group_rollout_index,
                ):
                    break
                self._group_rollout_index += 1
            else:
                if self._next_task_index >= len(self._task_ids):
                    break
                self._group_id += 1
                task_id = self._task_ids[self._next_task_index]
                self._next_task_index += 1
                self._group_rollouts[self._group_id] = []
                self._run_submitted_groups += 1
                self.total_submitted_groups += 1
                if not self._submit_one_rollout(
                    group_id=self._group_id,
                    task_id=task_id,
                    rollout_index=0,
                ):
                    break

        if not self._pending:
            if self._retry_rollouts:
                time.sleep(0.01)
                return "running"
            if self.is_training and trainer_finished:
                return self._finish("trainer_finished")
            if (not self.is_training) and self._next_task_index >= len(self._task_ids):
                return self._finish("completed")
            time.sleep(0.01)
            return "running"

        timeout = self.rollout_timeout_s if self.rollout_timeout_s and self.rollout_timeout_s > 0 else 0.01
        done_refs, _ = ray.wait(list(self._pending.keys()), num_returns=1, timeout=timeout)
        if not done_refs:
            if time.time() - self._last_logged > 10.0 and sum(self._active_rollouts):
                logger.info(
                    "%s: active=%s pending=%s groups_in_progress=%s started=%s finished=%s published_samples=%s finished_groups=%s",
                    self.scheduler_name,
                    sum(self._active_rollouts),
                    len(self._pending),
                    len(self._group_rollouts),
                    self._started_rollouts,
                    self._finished_rollouts,
                    self.total_published_samples,
                    self.total_finished_groups,
                )
                self._last_logged = time.time()
            return "running"

        for ref in done_refs:
            info = self._pending.pop(ref)
            self._active_rollouts[info.llm_index] -= 1
            self._active_rollouts_by_actor[info.benchmark_actor_index] -= 1

            rollout_result = RolloutResult.model_validate(ray.get(ref))
            rollout_result.model_version = info.model_version
            full_group_id = f"{self.scheduler_name}_{info.group_id}"
            rollout_result.group_id = full_group_id
            if not rollout_result.training_texts:
                logger.warning(
                    "Dropping empty rollout result and retrying: scheduler=%s group_id=%s "
                    "task_id=%s rollout_index=%s model_version=%s",
                    self.scheduler_name,
                    full_group_id,
                    info.task_id,
                    info.rollout_index,
                    info.model_version,
                )
                self._retry_rollouts.append((info.group_id, info.task_id, info.rollout_index))
                continue

            for step_index, sample in enumerate(rollout_result.training_texts):
                sample.metadata["model_version"] = info.model_version
                sample.metadata["rollout_index"] = info.rollout_index
                sample.metadata["step_index"] = step_index
                sample.group_id = full_group_id

            self._group_rollouts[info.group_id].append(rollout_result)
            self._finished_rollouts += 1

            if len(self._group_rollouts[info.group_id]) == self.attempts:
                group_results = self._group_rollouts.pop(info.group_id)
                if self.attempts > 1:
                    random.shuffle(group_results)
                group_samples = _write_group_result(
                    rollout_results=group_results,
                    attempts=self.attempts,
                    data_writer=self.data_writer,
                )

                self._run_finished_groups += 1
                self.total_finished_groups += 1
                self.total_published_samples += group_samples
                logger.info(
                    "Published %d %s samples to actor stream, total %d samples so far, "
                    "%d rollouts finished so far, %d groups finished so far, %d pending rollouts",
                    group_samples,
                    "train" if self.is_training else "test",
                    self.total_published_samples,
                    self._finished_rollouts,
                    self.total_finished_groups,
                    len(self._pending),
                )
                self.update_stats(group_results)

                should_publish_train_stats = self.is_training and (
                    self._trainer_version_to_publish is not None or self.debug_mode
                )
                should_publish_test_stats = (not self.is_training) and self._run_finished_groups == self._expected_groups
                if should_publish_train_stats or should_publish_test_stats:
                    loop_stats = {
                        "published_samples": self.total_published_samples,
                        "submitted_groups": self.total_submitted_groups,
                        "finished_groups": self.total_finished_groups,
                        "pending_rollouts": len(self._pending),
                        "active_rollouts": sum(self._active_rollouts),
                        "time_since_start": time.time() - self._loop_start_time,
                    }
                    if self.is_training:
                        loop_stats["trainer_model_version"] = self._trainer_version_to_publish
                        self._trainer_version_to_publish = None
                    else:
                        loop_stats["trainer_model_version"] = self._last_trainer_version
                    self.publish_stats(loop_stats)

        if (
            (not self.is_training)
            and self._run_finished_groups == self._expected_groups
            and not self._pending
            and not self._retry_rollouts
        ):
            return self._finish("completed")

        return "running"

    def run(
        self,
        *,
        task_ids: list[str],
        scheduler_name: str | None = None,
        stop_when_model_version_at_least: int | None = None,
    ) -> str:
        status = self.start(task_ids=task_ids, scheduler_name=scheduler_name)
        if status in {"no_tasks", "completed", "trainer_finished"}:
            return status

        while True:
            if (
                self.is_training
                and stop_when_model_version_at_least is not None
                and int(self.trainer_state.propagated_weight_version or 0) >= stop_when_model_version_at_least
            ):
                self.is_scheduling_paused = True

            status = self.step()
            if status in {"trainer_finished", "completed"}:
                return status

            if self.is_training and self.is_scheduling_paused and not self._pending and not self._retry_rollouts:
                return self._finish("eval_boundary")


def run_actor_loop_ray(cfg: DictConfig) -> None:
    from pipelinerl.finetune.logging_ import flatten_dict_config, init_wandb
    from pipelinerl.state import TrainerState
    from pipelinerl.streams import SingleStreamSpec, set_streams_backend, write_to_streams
    from pipelinerl.utils import setup_logging, wait_for_inference_servers

    set_streams_backend(**cfg.streams)
    random.seed(cfg.seed)

    exp_path = Path(cfg.output_dir)
    setup_logging(exp_path / "actor", "actor")
    logger.info("Current dir: %s, experiment root dir: %s", Path.cwd(), cfg.output_dir)

    def handle_sigterm(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    previous_sigterm_handler = signal.signal(signal.SIGTERM, handle_sigterm)
    run = None
    should_shutdown_ray = False
    ray_worker_log_collector = None

    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_path / "actor", flatten_dict_config(cfg))  # type: ignore[arg-type]
        if run is None:
            raise ValueError("Failed to initialize wandb run")

    llm_urls = [url for url in str(cfg.me.llm_urls).split("+") if url]
    if not llm_urls:
        raise ValueError("No actor llm URLs were provided")

    llm_max_rollouts = int(cfg.actor.llm_max_rollouts)
    if llm_max_rollouts < 1:
        raise ValueError("actor.llm_max_rollouts must be >= 1")
    cube_actor_max_rollouts = 1
    cube_actors_per_llm = llm_max_rollouts

    benchmark_instances = len(llm_urls) * cube_actors_per_llm
    actor_num_cpus = float(getattr(cfg.actor, "cube_actor_num_cpus", 1.0))
    required_ray_cpus = max(1, int(math.ceil(benchmark_instances * actor_num_cpus)))
    check_local_cube_actor_resources(
        cfg,
        instances=benchmark_instances,
        actor_num_cpus=actor_num_cpus,
        required_ray_cpus=required_ray_cpus,
    )
    should_shutdown_ray = _init_ray_runtime(cfg, owner_name="cube_actor", min_num_cpus=required_ray_cpus)
    ray_worker_log_collector = _launch_ray_worker_log_collector(cfg)

    logger.info(
        "Cube actor scheduler uses llm_max_rollouts=%d; derived actors_per_llm=%d with fixed actor_max_rollouts=%d",
        llm_max_rollouts,
        cube_actors_per_llm,
        cube_actor_max_rollouts,
    )

    logger.info(
        "Launching %d cube benchmark actors for %d llm urls (actors_per_llm=%d, actor_num_cpus=%.2f)",
        benchmark_instances,
        len(llm_urls),
        cube_actors_per_llm,
        actor_num_cpus,
    )
    benchmark_actors: list[Any] = []
    try:
        benchmark_actors = _launch_cube_benchmark_actors(
            cfg,
            instances=benchmark_instances,
            ray_worker_log_collector=ray_worker_log_collector,
        )
        health_timeout = float(getattr(cfg.actor, "cube_health_timeout", 600.0))
        _wait_for_cube_benchmark_actors(benchmark_actors, timeout_s=health_timeout)

        wait_for_inference_servers(llm_urls)

        train_task_ids = ray.get(benchmark_actors[0].get_train_task_ids.remote())
        test_task_ids = ray.get(benchmark_actors[0].get_test_task_ids.remote())

        if cfg.train_subset:
            train_task_ids = train_task_ids[cfg.train_subset.begin : cfg.train_subset.end]
        if cfg.test_subset:
            test_task_ids = test_task_ids[cfg.test_subset.begin : cfg.test_subset.end]

        if not train_task_ids:
            raise ValueError("Cube benchmark returned an empty train task list")

        logger.info("Loaded %d train task IDs", len(train_task_ids))
        logger.info("Loaded %d test task IDs", len(test_task_ids))

        trainer_state = TrainerState(exp_path)
        if cfg.debug.mode:
            trainer_state.propagated_weight_version = 0
        else:
            trainer_state.start_listening()
            trainer_state.wait_for_model_version()

        finetune_model_path = exp_path / "finetune" / "current"
        actor_model_path = finetune_model_path if finetune_model_path.exists() else Path(cfg.model_path)
        train_llms = _build_train_llms(cfg, llm_urls, actor_model_path)
        test_llms = _build_test_llms(cfg, llm_urls, actor_model_path)

        data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")
        stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
        test_data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor_test")
        test_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats_test")

        with (
            write_to_streams(data_stream, "a") as data_writer,
            write_to_streams(stats_stream, "a") as stats_writer,
            write_to_streams(test_data_stream, "a") as test_data_writer,
            write_to_streams(test_stats_stream, "a") as test_stats_writer,
        ):  
            train_loop = CubeActorLoop(
                cfg=cfg,
                llms=train_llms,
                benchmark_actors=benchmark_actors,
                trainer_state=trainer_state,
                data_writer=data_writer,
                stats_writer=stats_writer,
                scheduler_name="cube_train_scheduler",
                is_training=True,
            )
            test_loop = CubeActorLoop(
                cfg=cfg,
                llms=test_llms,
                benchmark_actors=benchmark_actors,
                trainer_state=trainer_state,
                data_writer=test_data_writer,
                stats_writer=test_stats_writer,
                scheduler_name="cube_test_scheduler",
                is_training=False,
            )

            last_regular_eval = -1
            eval_every_n_versions = int(getattr(cfg, "eval_every_n_versions", 0) or 0)

            while True:
                if _is_trainer_finished(cfg, trainer_state):
                    break
                next_regular_eval = (
                    int(trainer_state.propagated_weight_version or 0)
                    if last_regular_eval == -1
                    else last_regular_eval + eval_every_n_versions
                )
                should_eval = eval_every_n_versions > 0 and not cfg.debug.mode and bool(test_task_ids)
                train_stop_version = next_regular_eval if should_eval else None

                train_loop.is_scheduling_paused = False
                stop_reason = train_loop.run(
                    task_ids=train_task_ids,
                    stop_when_model_version_at_least=train_stop_version,
                )
                if stop_reason == "trainer_finished":
                    break

                if stop_reason == "eval_boundary" and should_eval:
                    eval_version = next_regular_eval
                    logger.info("Starting cube test loop for model version %s", eval_version)
                    _ = test_loop.run(
                        task_ids=test_task_ids,
                        scheduler_name=f"cube_test_scheduler_v{eval_version}",
                    )
                    last_regular_eval = eval_version
    finally:
        if run is not None:
            try:
                run.finish()
                logger.info("Finished W&B run")
            except Exception:
                logger.exception("Failed to finish W&B run")
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

        if benchmark_actors:
            try:
                ray.get([w.close.remote() for w in benchmark_actors])
                logger.info("Closed cube benchmark actors")
            except Exception:
                logger.exception("Failed to close cube benchmark actors")
        for actor in benchmark_actors:
            try:
                ray.kill(actor, no_restart=True)
                logger.info("Killed cube benchmark actor: %s", actor)
            except Exception:
                logger.exception("Failed to kill cube benchmark actor: %s", actor)
        if ray_worker_log_collector is not None:
            try:
                ray.get(ray_worker_log_collector.close.remote())
                logger.info("Closed Ray worker log collector")
            except Exception:
                logger.exception("Failed to close Ray worker log collector")
            try:
                ray.kill(ray_worker_log_collector, no_restart=True)
                logger.info("Killed Ray worker log collector")
            except Exception:
                logger.exception("Failed to kill Ray worker log collector")
        if should_shutdown_ray and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:
                logger.exception("Failed to shut down Ray")
