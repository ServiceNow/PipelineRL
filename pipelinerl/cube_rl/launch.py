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

from pipelinerl.cube_rl.domain import CubeBenchmarkWorker, length_penalty
from pipelinerl.cube_rl.ray_worker_logging import CubeRayWorkerLogCollector
from pipelinerl.cube_rl.registry import CubeTaskRef, build_cube_registry
from pipelinerl.cube_rl.routing import RayVLLMRouter, VLLMRouterActor
from pipelinerl.cube_rl.utils import (
    check_local_cube_worker_resources,
    close_ray_actor_best_effort,
    is_expected_ray_shutdown,
    kill_ray_actor_best_effort,
)
from pipelinerl.cube_rl.worker_pool import select_worker_for_cube
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
    task: CubeTaskRef
    rollout_index: int
    llm_index: int
    worker_index: int
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
            "%s: Ray cluster CPU capacity %.2f is lower than required %.2f for cube workers",
            owner_name,
            cluster_cpu_capacity,
            float(required_num_cpus),
        )
    else:
        logger.info(
            "%s: Ray cluster CPU capacity %.2f satisfies required %.2f for cube workers",
            owner_name,
            cluster_cpu_capacity,
            float(required_num_cpus),
        )


def _init_ray_runtime(cfg: DictConfig, owner_name: str, min_num_cpus: int = 1) -> bool:
    if ray.is_initialized():
        _log_ray_cpu_capacity(owner_name, required_num_cpus=min_num_cpus)
        return False

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


def _launch_cube_workers(
    cfg: DictConfig,
    instances: int,
    cube_specs: list[dict[str, Any]],
    llm: dict[str, Any],
    test_llm: dict[str, Any] | None = None,
    llm_router: Any | None = None,
    ray_worker_log_collector: Any | None = None,
) -> list[Any]:
    if instances < 1:
        raise ValueError("cube worker instance count must be >= 1")
    seed = int(getattr(cfg.get("cube_params", {}), "seed", cfg.seed))
    worker_num_cpus = float(getattr(cfg.actor, "cube_workers_num_cpus", 1.0))

    workers = []
    for idx in range(instances):
        worker_name = f"CUBE_WORKER_{idx}"
        worker = CubeBenchmarkWorker.options(num_cpus=worker_num_cpus).remote(
            cfg=cfg,
            cube_specs=cube_specs,
            seed=seed,
            worker_name=worker_name,
            llm=llm,
            test_llm=test_llm,
            llm_router=llm_router,
            ray_worker_log_collector=ray_worker_log_collector,
        )
        workers.append(worker)

    ray.get([worker.setup.remote() for worker in workers])
    return workers


def _wait_for_cube_workers(workers: list[Any], timeout_s: float) -> None:
    deadline = time.time() + max(1.0, timeout_s)
    while True:
        states = ray.get([worker.health.remote() for worker in workers])
        if all(state.get("ready", False) for state in states):
            logger.info("All cube workers are ready")
            return
        if time.time() >= deadline:
            raise TimeoutError(f"Timed out waiting for cube workers: {states}")
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
        cube_workers: list[Any],
        trainer_state: TrainerState,
        data_writer: StreamWriter,
        stats_writer: StreamWriter,
        scheduler_name: str,
        is_training: bool,
        vllm_router: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.llms = llms
        self.cube_workers = cube_workers
        self.trainer_state = trainer_state
        self.data_writer = data_writer
        self.stats_writer = stats_writer
        self.scheduler_name = scheduler_name
        self.is_training = is_training
        self.vllm_router = vllm_router
        self.debug_mode = bool(cfg.debug.mode)
        self.is_scheduling_paused = False

        self.attempts = int(cfg.attempts) if is_training else 1
        self.llm_max_rollouts = int(cfg.actor.llm_max_rollouts)
        if self.llm_max_rollouts < 1:
            raise ValueError("actor.llm_max_rollouts must be >= 1")
        # CubeBenchmarkWorker rollout is currently blocking/synchronous, so keep per-worker
        # concurrency at one and scale by worker count. LLM load is controlled by
        # per-generation leases in the shared router, not by worker-to-LLM pinning.
        self.cube_worker_max_rollouts = 1
        self.max_pending = max(1, len(self.llms) * self.llm_max_rollouts)
        self.rollout_timeout_s = getattr(cfg.actor, "rollout_timeout", None)
        if self.rollout_timeout_s is not None:
            self.rollout_timeout_s = float(self.rollout_timeout_s)
        self.llm_kwargs = [_build_llm_kwargs(llm) for llm in llms]
        self.sliding_aggregator = SlidingWindowAggregator(window_size=int(cfg.actor.throughput_window_size))

        ## Sanity check for difficulty-aware penalty config
        self.dap_cfg = self.cfg.actor.difficulty_aware_penalty
        if self.dap_cfg and self.dap_cfg.enabled:
            assert self.dap_cfg.gamma >= 0, (
                f"difficulty_aware_penalty.gamma must be >= 0, got {self.dap_cfg.gamma}"
            )
            failure_scale = getattr(self.dap_cfg, "failure_scale", 1.0)
            assert 0 <= failure_scale <= 1, (
                f"difficulty_aware_penalty.failure_scale must be in [0, 1], got {failure_scale}"
            )
        
        self.buffer_tokens = 0
        self.max_completion_tokens = self.cfg.llm.parameters.get("max_completion_tokens", None)

        self.total_published_samples = 0
        self.total_submitted_groups = 0
        self.total_finished_groups = 0
        self.init_stats()

        self._is_running = False
        self._tasks: list[CubeTaskRef] = []
        self._expected_groups = -1
        self._pending: dict[Any, PendingRollout] = {}
        self._retry_rollouts: deque[tuple[int, CubeTaskRef, int]] = deque()
        self._active_rollouts: list[int] = [0] * len(self.llms)
        self._active_rollouts_by_worker: list[int] = [0] * len(self.cube_workers)
        self._current_cube_by_worker: list[str | None] = [None] * len(self.cube_workers)
        self._group_rollouts: dict[int, list[RolloutResult]] = {}
        self._started_rollouts = 0
        self._finished_rollouts = 0
        self._run_submitted_groups = 0
        self._run_finished_groups = 0
        self._group_id = -1
        self._group_rollout_index = self.attempts
        self._current_task: CubeTaskRef | None = None
        self._next_task_index = 0
        self._loop_start_time = 0.0
        self._last_logged = 0.0
        self._stop_reason = "completed"
        self._last_trainer_version = 0
        self._trainer_version_to_publish: int | None = None
        self._allowed_worker_indices: list[int] | None = None
        self.max_lag = self.cfg.finetune.max_lag if self.is_training else None
        self.groups_per_update: int | None = None
        self.can_submit_before_update = math.inf
        if self.max_lag is not None:
            total_batch_size = self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
            total_update_size = (
                math.ceil(self.cfg.finetune.weight_update_interval / total_batch_size) * total_batch_size
            )
            if total_batch_size % self.attempts != 0:
                logger.warning(
                    "Trying to submit the exact right number of groups for this batch. "
                    "The attempt number %s ideally should divide total batch size %s",
                    self.attempts,
                    total_batch_size,
                )
            self.groups_per_update = math.ceil(total_update_size / self.attempts)
            lag_groups = math.ceil(self.max_lag / self.attempts)
            self.can_submit_before_update = lag_groups + self.groups_per_update
            logger.info(
                "Sync RL mode on, can submit %d groups for each update, that makes %d samples per update",
                self.groups_per_update,
                self.groups_per_update * self.attempts,
            )
            logger.info(
                "Max lag is %s samples, that makes %d additional starting chunks",
                self.max_lag,
                lag_groups,
            )

        logger.info(
            "%s: initialized global worker pool with %d llms, %d workers, worker max rollouts=%d, vllm max inflight/server=%d, max pending rollouts=%d",
            self.scheduler_name,
            len(self.llms),
            len(self.cube_workers),
            self.cube_worker_max_rollouts,
            self.llm_max_rollouts,
            self.max_pending,
        )

    @property
    def is_running(self) -> bool:
        return self._is_running

    def set_allowed_worker_indices(self, indices: list[int] | None) -> None:
        if indices is None:
            self._allowed_worker_indices = None
            return
        cleaned = sorted(set(int(idx) for idx in indices))
        if not cleaned:
            raise ValueError("allowed worker indices cannot be empty")
        max_index = len(self.cube_workers) - 1
        invalid = [idx for idx in cleaned if idx < 0 or idx > max_index]
        if invalid:
            raise ValueError(f"invalid worker indices: {invalid}")
        self._allowed_worker_indices = cleaned

    def _allowed_worker_indices_or_all(self) -> list[int]:
        if self._allowed_worker_indices is not None:
            return self._allowed_worker_indices
        return list(range(len(self.cube_workers)))

    def _max_pending_for_allowed_workers(self) -> int:
        allowed_worker_capacity = len(self._allowed_worker_indices_or_all()) * self.cube_worker_max_rollouts
        return max(1, min(self.max_pending, allowed_worker_capacity))

    def _select_worker_index(self, cube_id: str) -> int | None:
        worker_indices = self._allowed_worker_indices_or_all()
        return select_worker_for_cube(
            cube_id=cube_id,
            candidate_indices=worker_indices,
            active_rollouts_by_worker=self._active_rollouts_by_worker,
            worker_max_rollouts=self.cube_worker_max_rollouts,
            current_cube_by_worker=self._current_cube_by_worker,
        )

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
                    # Non-aggregatable value (None, or descriptive task info like the question
                    # text / submitted answer that a cube may surface). Skip rather than crash —
                    # the aggregator only tracks numeric stats.
                    logger.debug("Skipping non-numeric metric %r=%r (%s)", key, value, type(value).__name__)

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

        # Average attempts among SUCCESSFUL rollouts only. episodes_to_success logs 0 for failures, so
        # its mean / the success rate = sum(attempts)/num_successes — the clean "speed when it solves"
        # signal, undiluted by failures (companion to the blended episodes_to_success_mean).
        eps_mean = calculate_stats(self.stats.get("episodes_to_success", {})).get("mean")
        success_mean = calculate_stats(self.stats.get("success", {})).get("mean")
        if eps_mean is not None and success_mean and success_mean > 0:
            stats[f"{split_name}episodes_to_success_given_success"] = eps_mean / success_mean

        # pass@m curve (m=1..K) PER cube, from episodes_to_success (= first solved attempt, 0 if never):
        # fraction of that cube's rollouts solved within m attempts. Logged per dataset/cube so the
        # independent (TirAgent) and reflection (LaMer) test cubes stay separate. K = this loop's episode
        # budget (eval_k_episodes on the test loop, lamer_k_episodes on the train loop).
        if self.is_training:
            pass_k = int(getattr(self.cfg.actor, "lamer_k_episodes", 1))
        else:
            pass_k = int(getattr(self.cfg.actor, "eval_k_episodes", 0) or 0) or int(
                getattr(self.cfg.actor, "lamer_k_episodes", 1)
            )
        for ds, by_group in self.stats.get("episodes_to_success", {}).items():
            # self.stats is metric -> dataset -> group_id -> [values]; flatten the per-group lists.
            vals = [v for group_vals in by_group.values() for v in group_vals]
            if not vals:
                continue
            label = f"{split_name}{ds}/" if ds else split_name
            for m in range(1, pass_k + 1):
                stats[f"{label}pass@{m}"] = round(sum(1 for v in vals if 1 <= v <= m) / len(vals), 4)

        stats |= (
            {f"{split_name}{k}": v for k, v in always_or_never_success_stats(self.stats["success"]).items()}
            | {f"{split_name}latency_{k}": v for k, v in calculate_stats(self.latency_list).items()}
            | {f"{split_name}model_version_{k}": v for k, v in calculate_stats(self.model_versions_list).items()}
        )
        stats |= loop_stats
        if self.vllm_router is not None:
            try:
                router_snapshot = ray.get(self.vllm_router.snapshot.remote(), timeout=2.0)
                for server in router_snapshot.get("servers", []):
                    server_id = server.get("server_id")
                    prefix = f"{split_name}vllm_router/server_{server_id}"
                    for key in (
                        "inflight",
                        "active_tokens",
                        "latency_ema",
                        "errors",
                        "requests",
                        "suppressed",
                        "suppressed_remaining_s",
                        "affinities",
                    ):
                        value = server.get(key)
                        if isinstance(value, (bool, int, float)):
                            stats[f"{prefix}/{key}"] = value
                stats[f"{split_name}vllm_router/max_inflight_per_server"] = router_snapshot.get(
                    "max_inflight_per_server", 0
                )
                stats[f"{split_name}vllm_router/active_affinities"] = router_snapshot.get("active_affinities", 0)
            except Exception:
                logger.exception("%s: failed to collect vLLM router stats", self.scheduler_name)

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

    def _submit_one_rollout(self, *, group_id: int, task: CubeTaskRef, rollout_index: int) -> bool:
        worker_indices = self._allowed_worker_indices_or_all()
        if sum(self._active_rollouts_by_worker[idx] for idx in worker_indices) >= self._max_pending_for_allowed_workers():
            return False

        worker_index = self._select_worker_index(task.cube_id)
        if worker_index is None:
            return False
        worker = self.cube_workers[worker_index]
        next_llm = 0
        model_version = int(self.trainer_state.propagated_weight_version or 0)
        rollout_key = (
            f"{self.scheduler_name}:v{model_version}:g{group_id}:r{rollout_index}:"
            f"{task.cube_id}:{task.task_id}"
        )
        ref = worker.rollout.remote(
            cube_id=task.cube_id, task_id=task.task_id, rollout_key=rollout_key, is_training=self.is_training
        )
        self._pending[ref] = PendingRollout(
            group_id=group_id,
            task=task,
            rollout_index=rollout_index,
            llm_index=next_llm,
            worker_index=worker_index,
            model_version=model_version,
        )
        self._active_rollouts[next_llm] += 1
        self._active_rollouts_by_worker[worker_index] += 1
        self._current_cube_by_worker[worker_index] = task.cube_id
        self._started_rollouts += 1
        return True

    def start(self, *, tasks: list[CubeTaskRef], scheduler_name: str | None = None) -> str:
        if scheduler_name:
            self.scheduler_name = scheduler_name
        if not tasks:
            logger.info("%s: no tasks available; skipping", self.scheduler_name)
            return "no_tasks"
        if self._is_running:
            return "running"

        assert self.trainer_state.propagated_weight_version is not None
        self._tasks = tasks
        self._expected_groups = -1 if self.is_training else len(tasks)
        self._pending = {}
        self._retry_rollouts = deque()
        self._active_rollouts = [0] * len(self.llms)
        self._active_rollouts_by_worker = [0] * len(self.cube_workers)
        self._current_cube_by_worker = [None] * len(self.cube_workers)
        self._group_rollouts = {}
        self._started_rollouts = 0
        self._finished_rollouts = 0
        self._run_submitted_groups = 0
        self._run_finished_groups = 0
        self._group_id = -1
        self._group_rollout_index = self.attempts
        self._current_task = None
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

    def length_penalty_adjustment(self, rollout_results):
        # --- Difficulty-aware length penalty adjustment ---
        # Reduces the overlong penalty for SUCCESSFUL rollouts on hard problems,
        # so the model can reason longer when it actually leads to solving the problem.
        # Failed overlong rollouts keep the full penalty (failure_scale=1.0)
        # to discourage degenerate long generation that doesn't lead to solutions.
        # Hard cap guard: sequences that hit max_tokens without finishing always
        # get full penalty, even if the rollout is marked successful.
        if (
            self.is_training
            and self.dap_cfg
            and self.dap_cfg.enabled
            and self.buffer_tokens > 0
            and self.max_completion_tokens is not None
        ):
            group_solve_rate = sum(r.metrics.success for r in rollout_results) / len(rollout_results)
            gamma = self.dap_cfg.gamma
            failure_scale = getattr(self.dap_cfg, "failure_scale", 1.0)
            success_scale = group_solve_rate ** gamma
            buffer_tokens = self.buffer_tokens

            for r in rollout_results:
                rollout_scale = success_scale if r.metrics.success else failure_scale
                metrics_delta = 0.0
                for text in r.training_texts:
                    # Hard cap guard: if the sequence hit max_tokens without
                    # finishing, always apply full penalty regardless of success
                    if text.output_tokens >= self.max_completion_tokens and not text.finished:
                        scale = 1.0
                    else:
                        scale = rollout_scale
                    original_penalty = length_penalty(self.max_completion_tokens, text.output_tokens, buffer_tokens)
                    adjusted_penalty = original_penalty * scale
                    penalty_delta = adjusted_penalty - original_penalty
                    text.reward += penalty_delta
                    metrics_delta += penalty_delta
                r.metrics.reward += metrics_delta
                r._penalty_delta = metrics_delta
        else:
            for r in rollout_results:
                r._penalty_delta = 0.0

    def step(self) -> str:
        from pipelinerl.rollouts import RolloutResult

        if not self._is_running:
            return self._stop_reason

        trainer_finished = _is_trainer_finished(self.cfg, self.trainer_state)
        if self.is_training and int(self.trainer_state.propagated_weight_version or 0) > self._last_trainer_version:
            if self.max_lag is not None:
                assert self.groups_per_update is not None
                self.can_submit_before_update += self.groups_per_update
            self._trainer_version_to_publish = self._last_trainer_version
            self._last_trainer_version = int(self.trainer_state.propagated_weight_version or 0)
        
        while len(self._pending) < self._max_pending_for_allowed_workers():
            if self._retry_rollouts:
                group_id, task, rollout_index = self._retry_rollouts[0]
                if not self._submit_one_rollout(
                    group_id=group_id,
                    task=task,
                    rollout_index=rollout_index,
                ):
                    break
                self._retry_rollouts.popleft()
                continue

            if self.is_training:
                if trainer_finished or self.is_scheduling_paused:
                    break
                if self._group_rollout_index == self.attempts:
                    blocked_by_lag = self.total_submitted_groups >= self.can_submit_before_update
                    if blocked_by_lag:
                        break
                    self._group_id += 1
                    self._current_task = random.choice(self._tasks)
                    self._group_rollouts[self._group_id] = []
                    self._group_rollout_index = 0
                    self._run_submitted_groups += 1
                    self.total_submitted_groups += 1
                assert self._current_task is not None
                if not self._submit_one_rollout(
                    group_id=self._group_id,
                    task=self._current_task,
                    rollout_index=self._group_rollout_index,
                ):
                    break
                self._group_rollout_index += 1
            else:
                if self._next_task_index >= len(self._tasks):
                    break
                self._group_id += 1
                task = self._tasks[self._next_task_index]
                self._next_task_index += 1
                self._group_rollouts[self._group_id] = []
                self._run_submitted_groups += 1
                self.total_submitted_groups += 1
                if not self._submit_one_rollout(
                    group_id=self._group_id,
                    task=task,
                    rollout_index=0,
                ):
                    break

        if not self._pending:
            if self._retry_rollouts:
                time.sleep(0.01)
                return "running"
            if self.is_training and trainer_finished:
                return self._finish("trainer_finished")
            if (not self.is_training) and self._next_task_index >= len(self._tasks):
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
            self._active_rollouts_by_worker[info.worker_index] -= 1

            rollout_result = RolloutResult.model_validate(ray.get(ref))
            rollout_result.model_version = info.model_version
            full_group_id = f"{self.scheduler_name}_{info.group_id}"
            rollout_result.group_id = full_group_id
            if not rollout_result.training_texts:
                logger.warning(
                    "Dropping empty rollout result and retrying: scheduler=%s group_id=%s "
                    "cube_id=%s task_id=%s rollout_index=%s model_version=%s",
                    self.scheduler_name,
                    full_group_id,
                    info.task.cube_id,
                    info.task.task_id,
                    info.rollout_index,
                    info.model_version,
                )
                self._retry_rollouts.append((info.group_id, info.task, info.rollout_index))
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
                self.length_penalty_adjustment(group_results)
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
                    if self.is_training:
                        loop_stats = {
                            "published_samples": self.total_published_samples,
                            "submitted_groups": self.total_submitted_groups,
                            "finished_groups": self.total_finished_groups,
                            "pending_rollouts": len(self._pending),
                            "active_rollouts": sum(self._active_rollouts),
                            "time_since_start": time.time() - self._loop_start_time,
                        }
                        loop_stats["trainer_model_version"] = self._trainer_version_to_publish
                        self._trainer_version_to_publish = None
                    else:
                        loop_stats = {"trainer_model_version": self._last_trainer_version}
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
        tasks: list[CubeTaskRef],
        scheduler_name: str | None = None,
        stop_when_model_version_at_least: int | None = None,
    ) -> str:
        status = self.start(tasks=tasks, scheduler_name=scheduler_name)
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
    cube_worker_max_rollouts = 1
    cube_workers = int(cfg.actor.cube_workers)
    if cube_workers < 1:
        raise ValueError("actor.cube_workers must be >= 1")
    worker_instances = cube_workers
    worker_num_cpus = float(getattr(cfg.actor, "cube_workers_num_cpus", 1.0))
    required_ray_cpus = max(1, int(math.ceil(worker_instances * worker_num_cpus)))
    check_local_cube_worker_resources(
        cfg,
        instances=worker_instances,
        worker_num_cpus=worker_num_cpus,
        required_ray_cpus=required_ray_cpus,
    )
    should_shutdown_ray = _init_ray_runtime(cfg, owner_name="cube_worker", min_num_cpus=required_ray_cpus)
    ray_worker_log_collector = _launch_ray_worker_log_collector(cfg)

    logger.info(
        "Cube scheduler uses llm_max_rollouts=%d as per-vLLM generation capacity, cube_workers=%d, fixed worker_max_rollouts=%d",
        llm_max_rollouts,
        cube_workers,
        cube_worker_max_rollouts,
    )

    logger.info(
        "Launching %d cube workers for %d llm urls (worker_num_cpus=%.2f)",
        worker_instances,
        len(llm_urls),
        worker_num_cpus,
    )
    cube_workers: list[Any] = []
    vllm_router = None
    try:
        trainer_state = TrainerState(exp_path)
        if cfg.debug.mode:
            trainer_state.propagated_weight_version = 0
        else:
            trainer_state.start_listening()

        finetune_model_path = exp_path / "finetune" / "current"
        actor_model_path = finetune_model_path if finetune_model_path.exists() else Path(cfg.model_path)
        train_llms = _build_train_llms(cfg, llm_urls, actor_model_path)
        test_llms = _build_test_llms(cfg, llm_urls, actor_model_path)
        cube_registry = build_cube_registry(cfg)
        cube_runtime_specs = cube_registry.runtime_payloads()
        vllm_router = VLLMRouterActor.remote(
            [_build_llm_kwargs(llm) for llm in train_llms],
            max_inflight_per_server=llm_max_rollouts,
        )
        ray_vllm_router = RayVLLMRouter(vllm_router)
        logger.info("Started cube vLLM router: %s", ray.get(vllm_router.snapshot.remote()))

        cube_workers = _launch_cube_workers(
            cfg,
            instances=worker_instances,
            cube_specs=cube_runtime_specs,
            llm=_build_llm_kwargs(train_llms[0]),
            test_llm=_build_llm_kwargs(test_llms[0]),
            llm_router=ray_vllm_router,
            ray_worker_log_collector=ray_worker_log_collector,
        )
        health_timeout = float(getattr(cfg.actor, "cube_health_timeout", 600.0))
        _wait_for_cube_workers(cube_workers, timeout_s=health_timeout)

        logger.info("Cube workers are ready; waiting for actor inference servers before scheduling rollouts")
        wait_for_inference_servers(llm_urls)

        if not cfg.debug.mode:
            logger.info("Cube workers are ready; waiting for initial trainer model version before scheduling rollouts")
            trainer_state.wait_for_model_version()

        # Apply subsets PER cube spec (not the concatenated list) so multiple test cubes — e.g. an
        # independent + a reflection eval cube on the same held-out range — EACH get [begin:end].
        # (For a single cube this is identical to slicing the concatenation.)
        if cfg.train_subset:
            sl = slice(cfg.train_subset.begin, cfg.train_subset.end)
            train_tasks = [t for spec in cube_registry.specs if spec.split == "train" for t in spec.task_refs()[sl]]
        else:
            train_tasks = list(cube_registry.train_tasks)
        if cfg.test_subset:
            sl = slice(cfg.test_subset.begin, cfg.test_subset.end)
            test_tasks = [t for spec in cube_registry.specs if spec.split == "test" for t in spec.task_refs()[sl]]
        else:
            test_tasks = list(cube_registry.test_tasks)

        if not train_tasks:
            raise ValueError("Cube benchmark returned an empty train task list")

        logger.info("Loaded %d train cube task refs", len(train_tasks))
        logger.info("Loaded %d test cube task refs", len(test_tasks))

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
                cube_workers=cube_workers,
                trainer_state=trainer_state,
                data_writer=data_writer,
                stats_writer=stats_writer,
                scheduler_name="cube_train_scheduler",
                is_training=True,
                vllm_router=vllm_router,
            )
            test_loop = CubeActorLoop(
                cfg=cfg,
                llms=test_llms,
                cube_workers=cube_workers,
                trainer_state=trainer_state,
                data_writer=test_data_writer,
                stats_writer=test_stats_writer,
                scheduler_name="cube_test_scheduler",
                is_training=False,
                vllm_router=vllm_router,
            )

            last_regular_eval = -1
            current_eval = -1
            test_loop_active = False
            eval_every_n_versions = int(getattr(cfg, "eval_every_n_versions", 0) or 0)
            eval_test_worker_fraction = float(getattr(cfg.actor, "cube_eval_workers_fraction", 1.0))
            eval_test_worker_fraction = min(1.0, max(0.0, eval_test_worker_fraction))
            all_worker_indices = list(range(len(cube_workers)))
            if len(all_worker_indices) <= 1:
                eval_test_worker_indices = all_worker_indices
                eval_train_worker_indices = all_worker_indices
            else:
                test_worker_count = math.ceil(len(all_worker_indices) * eval_test_worker_fraction)
                test_worker_count = min(len(all_worker_indices) - 1, max(1, test_worker_count))
                eval_test_worker_indices = all_worker_indices[-test_worker_count:]
                eval_train_worker_indices = all_worker_indices[:-test_worker_count]
            logger.info(
                "Cube eval worker split: %d train workers, %d test workers, fraction=%.3f",
                len(eval_train_worker_indices),
                len(eval_test_worker_indices),
                eval_test_worker_fraction,
            )

            train_loop.start(tasks=train_tasks)
            while True:
                next_regular_eval = (
                    int(trainer_state.propagated_weight_version or 0)
                    if last_regular_eval == -1
                    else last_regular_eval + eval_every_n_versions
                )
                should_eval = eval_every_n_versions > 0 and not cfg.debug.mode and bool(test_tasks)

                if (
                    should_eval
                    and int(trainer_state.propagated_weight_version or 0) >= next_regular_eval
                    and not test_loop_active
                ):
                    current_eval = next_regular_eval
                    logger.info("Starting cube test loop for model version %s", current_eval)
                    train_loop.set_allowed_worker_indices(eval_train_worker_indices)
                    test_loop.set_allowed_worker_indices(eval_test_worker_indices)
                    test_status = test_loop.start(
                        tasks=test_tasks,
                        scheduler_name=f"cube_test_scheduler_v{current_eval}",
                    )
                    test_loop_active = test_status == "running"
                    if not test_loop_active:
                        last_regular_eval = current_eval
                        train_loop.set_allowed_worker_indices(None)
                        test_loop.set_allowed_worker_indices(None)

                try:
                    if test_loop_active:
                        test_status = test_loop.step()
                        if test_status == "completed":
                            test_loop_active = False
                            last_regular_eval = current_eval
                            train_loop.set_allowed_worker_indices(None)
                            test_loop.set_allowed_worker_indices(None)

                    train_status = train_loop.step()
                except Exception as exc:
                    if is_expected_ray_shutdown(exc):
                        logger.info("Stopping Cube actor loop because a Ray worker node is shutting down")
                        break
                    raise
                if train_status == "trainer_finished":
                    break
    finally:
        if run is not None:
            try:
                run.finish()
                logger.info("Finished W&B run")
            except Exception:
                logger.exception("Failed to finish W&B run")
        signal.signal(signal.SIGTERM, previous_sigterm_handler)

        for worker in cube_workers:
            close_ray_actor_best_effort(worker, logger, "cube worker")
        for worker in cube_workers:
            kill_ray_actor_best_effort(worker, logger, "cube worker")
        if ray_worker_log_collector is not None:
            close_ray_actor_best_effort(ray_worker_log_collector, logger, "Ray worker log collector")
            kill_ray_actor_best_effort(ray_worker_log_collector, logger, "Ray worker log collector")
        if vllm_router is not None:
            kill_ray_actor_best_effort(vllm_router, logger, "cube vLLM router")
        if should_shutdown_ray and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:
                logger.exception("Failed to shut down Ray")
