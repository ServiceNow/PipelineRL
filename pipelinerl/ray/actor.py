from __future__ import annotations

import logging
import math
import random
import signal
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from pipelinerl.utils import always_or_never_success_stats, calculate_stats, SlidingWindowAggregator
from pipelinerl.domain_sampling import DomainWeightedSampler
from pipelinerl.ray.logging import RayWorkerLogCollector
from pipelinerl.ray.manager import RayRolloutManager, RolloutExecutionError
from pipelinerl.ray.utils import (
    check_local_ray_worker_resources,
    close_ray_actor_best_effort,
    is_expected_ray_shutdown,
    kill_ray_actor_best_effort,
)
from pipelinerl.ray.worker import RolloutDataset
from pipelinerl.rollouts import BaseMetrics, RolloutRequest, RolloutResult, rollout_has_overflow

logger = logging.getLogger(__name__)


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


def _is_trainer_finished(cfg: DictConfig, trainer_state: Any) -> bool:
    return trainer_state.samples_processed is not None and trainer_state.samples_processed >= _samples_target(cfg)


def _log_ray_cpu_capacity(owner_name: str, required_num_cpus: int) -> None:
    if required_num_cpus < 1 or not ray.is_initialized():
        return
    cluster_cpu_capacity = float(ray.cluster_resources().get("CPU", 0.0))
    if cluster_cpu_capacity + 1e-6 < float(required_num_cpus):
        logger.warning(
            "%s: Ray cluster CPU capacity %.2f is lower than required %.2f for rollout workers",
            owner_name,
            cluster_cpu_capacity,
            float(required_num_cpus),
        )
    else:
        logger.info(
            "%s: Ray cluster CPU capacity %.2f satisfies required %.2f for rollout workers",
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
            local_mode=False,
        )
        logger.info("%s: connected to ray at configured address %s", owner_name, ray_address)
        _log_ray_cpu_capacity(owner_name, required_num_cpus=min_num_cpus)
        return True

    try:
        ray.init(
            address="auto",
            ignore_reinit_error=True,
            log_to_driver=False,
            local_mode=False,
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
            local_mode=False,
        )
        logger.info("%s: started local ray runtime with %d CPUs", owner_name, ray_num_cpus)

    return True


def _build_llm_kwargs(llm: Any) -> dict[str, Any]:
    kwargs = {
        "base_url": llm.base_url,
        "model_name": llm.model_name,
        "tokenizer_name": llm.tokenizer_name,
        "parameters": llm.parameters,
        "collect_logprobs": llm.collect_logprobs,
    }
    served_model_name = getattr(llm, "served_model_name", None)
    if served_model_name:
        kwargs["served_model_name"] = served_model_name
    return kwargs


def _build_train_llms(cfg: DictConfig, llm_urls: list[str], actor_model_path: Path) -> list[Any]:
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


def _build_eval_llms(cfg: DictConfig, llm_urls: list[str], actor_model_path: Path) -> list[Any]:
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


class LLMLoadBalancer:
    def __init__(self, *, max_rollouts_per_url: int):
        if max_rollouts_per_url < 1:
            raise ValueError("actor.llm_max_rollouts must be >= 1")
        self.max_rollouts_per_url = max_rollouts_per_url
        self._active: dict[str, int] = defaultdict(int)

    def try_acquire(self, llms: list[Any]) -> dict[str, Any] | None:
        if not llms:
            return None
        candidates = sorted((_build_llm_kwargs(llm) for llm in llms), key=lambda item: (self._active[item["base_url"]], item["base_url"]))
        selected = candidates[0]
        base_url = selected["base_url"]
        if self._active[base_url] >= self.max_rollouts_per_url:
            return None
        self._active[base_url] += 1
        return selected

    def release(self, llm: dict[str, Any] | None) -> None:
        if not llm:
            return
        base_url = llm.get("base_url")
        if base_url is None:
            return
        current = self._active.get(base_url, 0)
        if current <= 1:
            self._active.pop(base_url, None)
        else:
            self._active[base_url] = current - 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_rollouts_per_url": self.max_rollouts_per_url,
            "active": dict(sorted(self._active.items())),
        }


def _write_group_result(rollout_results: list[RolloutResult], attempts: int, data_writer: Any) -> int:
    assert len(rollout_results) == attempts, f"Expected {attempts} rollouts, got {len(rollout_results)}"
    payload = [text.model_dump() for result in rollout_results for text in result.training_texts]
    data_writer.write(payload)
    return len(payload)


class RayActorLoop:
    def __init__(
        self,
        *,
        cfg: DictConfig,
        dataset: RolloutDataset,
        manager: RayRolloutManager,
        trainer_state: Any,
        data_writer: Any,
        stats_writer: Any,
        scheduler_name: str,
        is_training: bool,
        llm_load_balancer: LLMLoadBalancer,
        llms: list[Any],
    ) -> None:
        self.cfg = cfg
        self.dataset = dataset
        self.manager = manager
        self.trainer_state = trainer_state
        self.data_writer = data_writer
        self.stats_writer = stats_writer
        self.scheduler_name = scheduler_name
        self.is_training = is_training
        self.llm_load_balancer = llm_load_balancer
        self.llms = llms
        self.debug_mode = bool(cfg.debug.mode)
        
        self.attempts = int(cfg.attempts) if is_training else 1
        self.rollout_timeout_s = getattr(cfg.actor, "rollout_timeout", None)
        self.rollout_timeout_s = float(self.rollout_timeout_s) if self.rollout_timeout_s is not None else None
        self.sliding_aggregator = SlidingWindowAggregator(window_size=int(cfg.actor.throughput_window_size))
        self.dap_cfg = self.cfg.actor.difficulty_aware_penalty
        self.buffer_tokens = int(getattr(self.cfg.actor, "buffer_tokens", 0))
        self.max_completion_tokens = self.cfg.llm.parameters.get("max_completion_tokens", None)

        if self.dap_cfg and self.dap_cfg.enabled:
            assert self.dap_cfg.gamma >= 0, f"difficulty_aware_penalty.gamma must be >= 0, got {self.dap_cfg.gamma}"
            failure_scale = getattr(self.dap_cfg, "failure_scale", 1.0)
            assert 0 <= failure_scale <= 1, f"difficulty_aware_penalty.failure_scale must be in [0, 1], got {failure_scale}"

        self.total_published_samples = 0
        self.total_submitted_groups = 0
        self.total_finished_groups = 0
        self.init_stats()

        self._is_running = False
        self._expected_groups = -1
        self._retry_requests: deque[RolloutRequest] = deque()
        self._group_rollouts: dict[int, list[RolloutResult]] = {}
        self._started_rollouts = 0
        self._finished_rollouts = 0
        self._run_submitted_groups = 0
        self._run_finished_groups = 0
        self._group_id = -1
        self._group_rollout_index = self.attempts
        self._current_item: dict[str, Any] | None = None
        self._next_item_index = 0
        self._loop_start_time = 0.0
        self._last_logged = 0.0
        self._stop_reason = "completed"
        self._last_trainer_version = 0
        self._trainer_version_to_publish: int | None = None
        self.max_lag = self.cfg.finetune.max_lag if self.is_training else None
        self.groups_per_update: int | None = None
        self.can_submit_before_update = math.inf
        if self.max_lag is not None:
            total_batch_size = self.cfg.finetune.train_batch_size * self.cfg.finetune.gradient_accumulation_passes
            total_update_size = math.ceil(self.cfg.finetune.weight_update_interval / total_batch_size) * total_batch_size
            if total_batch_size % self.attempts != 0:
                logger.warning(
                    "The attempt number %s ideally should divide total batch size %s",
                    self.attempts,
                    total_batch_size,
                )
            self.groups_per_update = math.ceil(total_update_size / self.attempts)
            lag_groups = math.ceil(self.max_lag / self.attempts)
            self.can_submit_before_update = lag_groups + self.groups_per_update

        self.domain_sampler = None
        if self.is_training and len(cfg.train_dataset_names) > 1:
            domain_mix_cfg = getattr(self.cfg.actor, "domain_mix", None)
            if domain_mix_cfg:
                mix_weights = OmegaConf.to_container(domain_mix_cfg, resolve=True)
                if not isinstance(mix_weights, dict):
                    raise ValueError("actor.domain_mix must be a mapping from domain to weight")
                self.domain_sampler = DomainWeightedSampler(self.dataset, mix_weights)

    @property
    def is_running(self) -> bool:
        return self._is_running

    def init_stats(self) -> None:
        self.stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.latency_list: list[float] = []
        self.model_versions_list: list[int] = []
        self.sliding_stats = defaultdict(list)
        self.domain_counts = defaultdict(int)
        self.dataset_to_domain: dict[str, str] = {}

    def compute_domain_agnostic_metrics(self, result: RolloutResult) -> dict[str, float | bool | list[int]]:
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
            max_tokens = test_llm_cfg.parameters.get("max_tokens", None) if test_llm_cfg else self.cfg.llm.parameters.get("max_tokens", None)
        if max_tokens is not None:
            is_overlong = any(t.output_tokens >= max_tokens for t in result.training_texts)
            metrics["overlong"] = is_overlong
            metrics["overlong_success"] = is_overlong and result.metrics.success
        return metrics

    def update_stats(self, rollout_results: list[RolloutResult]) -> None:
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
                    domain_groups[domain].update({(dataset_name, gid): vals for gid, vals in list_of_stats_per_dataset.items()})
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
        balancer_snapshot = self.llm_load_balancer.snapshot()
        stats[f"{split_name}llm_load_balancer/max_rollouts_per_url"] = balancer_snapshot["max_rollouts_per_url"]
        for base_url, active in balancer_snapshot["active"].items():
            stats[f"{split_name}llm_load_balancer/active/{base_url}"] = active

        total_domain_samples = sum(self.domain_counts.values())
        if total_domain_samples:
            for domain, count in sorted(self.domain_counts.items()):
                stats[f"{split_name}domain_mix_count/{domain}"] = count
                stats[f"{split_name}domain_mix_actual/{domain}"] = count / total_domain_samples
        for key, value in self.sliding_stats.items():
            stats[key] = sum(value) / len(value) if value else 0.0
        if self.cfg.wandb.use_wandb:
            import wandb
            wandb.log({f"actor/{k}": v for k, v in stats.items()})
        self.stats_writer.write(stats)
        self.init_stats()

    def _make_request(self, *, group_id: int, item: dict[str, Any], rollout_index: int) -> RolloutRequest:
        model_version = int(self.trainer_state.propagated_weight_version or 0)
        full_group_id = f"{self.scheduler_name}_{group_id}"
        request_id = f"{full_group_id}:r{rollout_index}:v{model_version}:{time.time_ns()}"
        return RolloutRequest(
            request_id=request_id,
            dataset_item=item,
            model_version=model_version,
            group_id=full_group_id,
            rollout_index=rollout_index,
            seed=int(self.cfg.seed) + group_id * max(1, self.attempts) + rollout_index,
            max_steps=getattr(self.cfg.actor, "max_steps", None),
            extras={"local_group_id": group_id, "scheduler_name": self.scheduler_name},
        )

    def _release_request_llm(self, request: RolloutRequest) -> None:
        llm = request.extras.pop("llm", None)
        self.llm_load_balancer.release(llm)

    def _submit_request(self, request: RolloutRequest) -> bool:
        if self.manager.idle_count <= 0:
            return False
        selected_llm = self.llm_load_balancer.try_acquire(self.llms)
        if selected_llm is None:
            return False
        request.extras["llm"] = selected_llm
        if self.manager.try_submit(request):
            self._started_rollouts += 1
            return True
        self._release_request_llm(request)
        return False

    def start(self, *, scheduler_name: str | None = None) -> str:
        if scheduler_name:
            self.scheduler_name = scheduler_name
        if len(self.dataset) == 0:
            logger.info("%s: no dataset items available; skipping", self.scheduler_name)
            return "no_tasks"
        if self._is_running:
            return "running"
        assert self.trainer_state.propagated_weight_version is not None
        self._expected_groups = -1 if self.is_training else len(self.dataset)
        self._retry_requests = deque()
        self._group_rollouts = {}
        self._started_rollouts = 0
        self._finished_rollouts = 0
        self._run_submitted_groups = 0
        self._run_finished_groups = 0
        self._group_id = -1
        self._group_rollout_index = self.attempts
        self._current_item = None
        self._next_item_index = 0
        self._loop_start_time = time.time()
        self._last_logged = time.time()
        self._stop_reason = "running"
        self._last_trainer_version = int(self.trainer_state.propagated_weight_version)
        self._trainer_version_to_publish = None
        self._is_running = True
        logger.info("Starting %s Ray actor loop (%s)", "train" if self.is_training else "eval", self.scheduler_name)
        return "running"

    def _finish(self, reason: str) -> str:
        self._stop_reason = reason
        if self.is_training and self._trainer_version_to_publish is not None and self.latency_list:
            self.publish_stats(
                {
                    "published_samples": self.total_published_samples,
                    "submitted_groups": self.total_submitted_groups,
                    "finished_groups": self.total_finished_groups,
                    "pending_rollouts": self.manager.active_count,
                    "active_rollouts": self.manager.active_count,
                    "time_since_start": time.time() - self._loop_start_time,
                    "trainer_model_version": self._trainer_version_to_publish,
                }
            )
            self._trainer_version_to_publish = None
        self._is_running = False
        logger.info("%s loop finished: reason=%s started=%s finished=%s", self.scheduler_name, reason, self._started_rollouts, self._finished_rollouts)
        return reason

    def length_penalty_adjustment(self, rollout_results: list[RolloutResult]) -> None:
        def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
            if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
                return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
            return 0.0

        if self.is_training and self.dap_cfg and self.dap_cfg.enabled and self.buffer_tokens > 0 and self.max_completion_tokens is not None:
            group_solve_rate = sum(r.metrics.success for r in rollout_results) / len(rollout_results)
            gamma = self.dap_cfg.gamma
            failure_scale = getattr(self.dap_cfg, "failure_scale", 1.0)
            success_scale = group_solve_rate ** gamma
            for result in rollout_results:
                rollout_scale = success_scale if result.metrics.success else failure_scale
                metrics_delta = 0.0
                for text in result.training_texts:
                    scale = 1.0 if text.output_tokens >= self.max_completion_tokens and not text.finished else rollout_scale
                    original_penalty = length_penalty(self.max_completion_tokens, text.output_tokens, self.buffer_tokens)
                    adjusted_penalty = original_penalty * scale
                    penalty_delta = adjusted_penalty - original_penalty
                    text.reward += penalty_delta
                    metrics_delta += penalty_delta
                result.metrics.reward += metrics_delta
                result._penalty_delta = metrics_delta
        else:
            for result in rollout_results:
                result._penalty_delta = 0.0

    def step(self) -> str:
        if not self._is_running:
            return self._stop_reason

        trainer_finished = _is_trainer_finished(self.cfg, self.trainer_state)
        if self.is_training and int(self.trainer_state.propagated_weight_version or 0) > self._last_trainer_version:
            if self.max_lag is not None:
                assert self.groups_per_update is not None
                self.can_submit_before_update += self.groups_per_update
            self._trainer_version_to_publish = self._last_trainer_version
            self._last_trainer_version = int(self.trainer_state.propagated_weight_version or 0)

        while True:
            if self._retry_requests:
                request = self._retry_requests[0]
                if not self._submit_request(request):
                    break
                self._retry_requests.popleft()
                continue

            if self.is_training:
                if trainer_finished:
                    break
                if self._group_rollout_index == self.attempts:
                    blocked_by_lag = self.total_submitted_groups >= self.can_submit_before_update
                    if blocked_by_lag:
                        break
                    self._group_id += 1

                    if self.domain_sampler is not None:
                        self._current_item = self.domain_sampler.sample()
                    else:
                        self._current_item = self.dataset[random.randrange(len(self.dataset))]
                        
                    self._group_rollouts[self._group_id] = []
                    self._group_rollout_index = 0
                    self._run_submitted_groups += 1
                    self.total_submitted_groups += 1
                assert self._current_item is not None
                request = self._make_request(group_id=self._group_id, item=self._current_item, rollout_index=self._group_rollout_index)
                if not self._submit_request(request):
                    break
                self._group_rollout_index += 1
            else:
                if self._next_item_index >= len(self.dataset):
                    break
                self._group_id += 1
                item = self.dataset[self._next_item_index]
                self._next_item_index += 1
                self._group_rollouts[self._group_id] = []
                self._run_submitted_groups += 1
                self.total_submitted_groups += 1
                request = self._make_request(group_id=self._group_id, item=item, rollout_index=0)
                if not self._submit_request(request):
                    self._next_item_index -= 1
                    self._group_id -= 1
                    self._run_submitted_groups -= 1
                    self.total_submitted_groups -= 1
                    self._group_rollouts.pop(self._group_id + 1, None)
                    break

        if self.manager.active_count == 0:
            if self._retry_requests:
                time.sleep(0.01)
                return "running"
            if self.is_training and trainer_finished:
                return self._finish("trainer_finished")
            if (not self.is_training) and self._next_item_index >= len(self.dataset):
                return self._finish("completed")
            time.sleep(0.01)
            return "running"

        timeout = self.rollout_timeout_s if self.rollout_timeout_s and self.rollout_timeout_s > 0 else 0.01
        try:
            completed = self.manager.wait_completed(timeout_s=timeout)
        except RolloutExecutionError as exc:
            logger.warning("Rollout failed and will be retried: request_id=%s", exc.request.request_id, exc_info=True)
            self._release_request_llm(exc.request)
            self._retry_requests.append(exc.request)
            return "running"
        if not completed:
            if time.time() - self._last_logged > 10.0 and self.manager.active_count:
                logger.info(
                    "%s: active=%s groups_in_progress=%s started=%s finished=%s published_samples=%s finished_groups=%s",
                    self.scheduler_name,
                    self.manager.active_count,
                    len(self._group_rollouts),
                    self._started_rollouts,
                    self._finished_rollouts,
                    self.total_published_samples,
                    self.total_finished_groups,
                )
                self._last_logged = time.time()
            return "running"

        for item in completed:
            request = item.request
            result = item.result
            local_group_id = int(request.extras["local_group_id"])
            self._release_request_llm(request)
            result.model_version = request.model_version
            result.group_id = request.group_id
            if not result.training_texts:
                logger.warning("Dropping empty rollout result and retrying: request_id=%s group_id=%s", request.request_id, request.group_id)
                self._retry_requests.append(request)
                continue
            for step_index, sample in enumerate(result.training_texts):
                sample.metadata["model_version"] = request.model_version
                sample.metadata["rollout_index"] = request.rollout_index
                sample.metadata["step_index"] = step_index
                sample.group_id = request.group_id
            self._group_rollouts[local_group_id].append(result)
            self._finished_rollouts += 1
            if len(self._group_rollouts[local_group_id]) == self.attempts:
                group_results = self._group_rollouts.pop(local_group_id)
                if self.attempts > 1:
                    random.shuffle(group_results)

                # Track completions per domain for adaptive sampling
                if self.domain_sampler is not None:
                    for r in group_results:
                        if r.domain:
                            self.domain_sampler.record_completion(r.domain)

                self.length_penalty_adjustment(group_results)
                group_samples = _write_group_result(group_results, self.attempts, self.data_writer)
                self._run_finished_groups += 1
                self.total_finished_groups += 1
                self.total_published_samples += group_samples
                self.update_stats(group_results)
                should_publish_train_stats = self.is_training and (self._trainer_version_to_publish is not None or self.debug_mode)
                should_publish_eval_stats = (not self.is_training) and self._run_finished_groups == self._expected_groups
                if should_publish_train_stats or should_publish_eval_stats:
                    if self.is_training:
                        loop_stats = {
                            "published_samples": self.total_published_samples,
                            "submitted_groups": self.total_submitted_groups,
                            "finished_groups": self.total_finished_groups,
                            "pending_rollouts": self.manager.active_count,
                            "active_rollouts": self.manager.active_count,
                            "time_since_start": time.time() - self._loop_start_time,
                            "trainer_model_version": self._trainer_version_to_publish,
                        }
                        self._trainer_version_to_publish = None
                    else:
                        loop_stats = {"trainer_model_version": self._last_trainer_version}
                    self.publish_stats(loop_stats)

        if (not self.is_training) and self._run_finished_groups == self._expected_groups and self.manager.active_count == 0 and not self._retry_requests:
            return self._finish("completed")
        return "running"


def _worker_config(cfg: DictConfig, mode: str) -> dict[str, Any]:
    actor_cfg = OmegaConf.to_container(cfg.actor, resolve=True)
    reward_shaping_cfg = OmegaConf.to_container(getattr(cfg, "reward_shaping", {}), resolve=True)
    return {
        "mode": mode,
        "actor": actor_cfg,
        "output_dir": str(cfg.output_dir),
        "reward_shaping": reward_shaping_cfg,
    }


def _manager_ray_options(cfg: DictConfig, *, mode: str) -> dict[str, Any]:
    actor_cfg = OmegaConf.to_container(cfg.actor, resolve=True)
    options = actor_cfg.get("ray_options", {})
    if not isinstance(options, dict):
        raise ValueError("actor.ray_options must be a mapping")
    worker_num_cpus = float(getattr(cfg.actor, "ray_worker_num_cpus", 0.25))
    options.setdefault("num_cpus", worker_num_cpus)
    return options


def _launch_ray_worker_log_collector(cfg: DictConfig, exp_path: Path) -> Any | None:
    if not bool(getattr(cfg.actor, "ray_worker_log_enabled", True)):
        return None
    worker_log_path = getattr(cfg.actor, "ray_worker_log_path", None) or str(exp_path / "actor" / "ray_workers.jsonl")
    collector = RayWorkerLogCollector.remote(str(worker_log_path))
    logger.info("Ray worker logs will be collected in %s", worker_log_path)
    return collector


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
    train_manager = None
    eval_manager = None
    train_dataset = None
    eval_dataset = None

    if cfg.wandb.use_wandb:
        run = init_wandb(cfg, exp_path / "actor", flatten_dict_config(cfg))  # type: ignore[arg-type]
        if run is None:
            raise ValueError("Failed to initialize wandb run")

    llm_urls = [url for url in str(cfg.me.llm_urls).split("+") if url]
    if not llm_urls:
        raise ValueError("No actor llm URLs were provided")

    train_workers = int(getattr(cfg.actor, "ray_workers"))
    if train_workers < 1:
        raise ValueError("actor.ray_workers must be >= 1")
    eval_cfg = getattr(cfg.actor, "eval", None)
    eval_workers = int(getattr(eval_cfg, "ray_workers", 0) or 0)
    train_worker_num_cpus = float(getattr(cfg.actor, "ray_worker_num_cpus", 0.25))
    eval_worker_num_cpus = train_worker_num_cpus
    reduced_train_workers = max(1, train_workers - eval_workers) if eval_workers > 0 else train_workers
    required_ray_cpus = max(1, int(math.ceil(max(train_workers * train_worker_num_cpus, reduced_train_workers * train_worker_num_cpus + eval_workers * eval_worker_num_cpus))))
    max_concurrent_workers = max(train_workers, reduced_train_workers + eval_workers)
    check_local_ray_worker_resources(cfg, instances=max_concurrent_workers, worker_num_cpus=max(train_worker_num_cpus, eval_worker_num_cpus), required_ray_cpus=required_ray_cpus)
    should_shutdown_ray = _init_ray_runtime(cfg, owner_name="ray_actor", min_num_cpus=required_ray_cpus)
    ray_worker_log_collector = _launch_ray_worker_log_collector(cfg, exp_path)

    try:
        trainer_state = TrainerState(exp_path)
        if cfg.debug.mode:
            trainer_state.propagated_weight_version = 0
        else:
            trainer_state.start_listening()

        finetune_model_path = exp_path / "finetune" / "current"
        actor_model_path = finetune_model_path if finetune_model_path.exists() else Path(cfg.model_path)
        train_llms = _build_train_llms(cfg, llm_urls, actor_model_path)
        eval_llms = _build_eval_llms(cfg, llm_urls, actor_model_path)
        dataset_loader = hydra.utils.get_method(cfg.dataset_loader)
        dataset_loader_params = cfg.get('dataset_loader_params', {})

        train_dataset = dataset_loader(cfg.train_dataset_names, **dataset_loader_params)
        eval_dataset = dataset_loader(cfg.test_dataset_names, **dataset_loader_params)

        logger.info("Loaded %d train rollout items", len(train_dataset))
        logger.info("Loaded %d eval rollout items", len(eval_dataset) if eval_dataset is not None else 0)

        llm_max_rollouts = int(cfg.actor.llm_max_rollouts)
        llm_load_balancer = LLMLoadBalancer(max_rollouts_per_url=llm_max_rollouts)
        logger.info("Using actor-local LLM load balancer: %s", llm_load_balancer.snapshot())

        worker_cls = hydra.utils.get_class(cfg.actor.rollout_policy)
        rollout_backend = str(getattr(cfg.actor, "rollout_backend", "ray"))
        logger.info("Using rollout backend: %s", rollout_backend)
        train_manager = RayRolloutManager(
            worker_cls=worker_cls,
            worker_config=_worker_config(cfg, "train"),
            num_workers=train_workers,
            ray_options=_manager_ray_options(cfg, mode="train"),
            execution_backend=rollout_backend,
            log_collector=ray_worker_log_collector,
            worker_name_prefix="train_rollout_worker",
        )

        wait_for_inference_servers(llm_urls)
        if not cfg.debug.mode:
            trainer_state.wait_for_model_version()

        data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor")
        stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats")
        eval_data_stream = SingleStreamSpec(exp_path=exp_path, topic="actor_test")
        eval_stats_stream = SingleStreamSpec(exp_path=exp_path, topic="stats_test")

        with (
            write_to_streams(data_stream, "a") as data_writer,
            write_to_streams(stats_stream, "a") as stats_writer,
            write_to_streams(eval_data_stream, "a") as eval_data_writer,
            write_to_streams(eval_stats_stream, "a") as eval_stats_writer,
        ):
            train_loop = RayActorLoop(
                cfg=cfg,
                dataset=train_dataset,
                manager=train_manager,
                trainer_state=trainer_state,
                data_writer=data_writer,
                stats_writer=stats_writer,
                scheduler_name="ray_train_scheduler",
                is_training=True,
                llm_load_balancer=llm_load_balancer,
                llms=train_llms,
            )
            train_loop.start()

            last_regular_eval = -1
            current_eval = -1
            eval_loop: RayActorLoop | None = None
            eval_every_n_versions = int(getattr(cfg, "eval_every_n_versions", 0) or 0)
            resource_policy = str(getattr(eval_cfg, "resource_policy", "elastic_train") if eval_cfg else "elastic_train")
            while True:
                next_regular_eval = int(trainer_state.propagated_weight_version or 0) if last_regular_eval == -1 else last_regular_eval + eval_every_n_versions
                should_eval = eval_every_n_versions > 0 and not cfg.debug.mode and eval_dataset is not None and len(eval_dataset) > 0
                if should_eval and eval_loop is None and int(trainer_state.propagated_weight_version or 0) >= next_regular_eval:
                    current_eval = next_regular_eval
                    if eval_workers > 0 and resource_policy == "elastic_train":
                        reduced_train_workers = max(1, train_workers - eval_workers)
                        logger.info("Starting eval v%s with elastic train capacity: train workers %d -> %d, eval workers %d", current_eval, train_workers, reduced_train_workers, eval_workers)
                        train_manager.set_target_workers(reduced_train_workers)
                        while train_manager.worker_count > reduced_train_workers:
                            train_loop.step()
                    eval_manager = RayRolloutManager(
                        worker_cls=worker_cls,
                        worker_config=_worker_config(cfg, "eval"),
                        num_workers=max(1, eval_workers),
                        ray_options=_manager_ray_options(cfg, mode="eval"),
                        execution_backend=rollout_backend,
                        log_collector=ray_worker_log_collector,
                        worker_name_prefix=f"eval_v{current_eval}_rollout_worker",
                    )
                    if eval_dataset:
                        eval_loop = RayActorLoop(
                            cfg=cfg,
                            dataset=eval_dataset,
                            manager=eval_manager,
                            trainer_state=trainer_state,
                            data_writer=eval_data_writer,
                            stats_writer=eval_stats_writer,
                            scheduler_name=f"ray_eval_scheduler_v{current_eval}",
                            is_training=False,
                            llm_load_balancer=llm_load_balancer,
                            llms=eval_llms,
                        )
                        eval_loop.start()

                try:
                    if eval_loop is not None:
                        eval_status = eval_loop.step()
                        if eval_status == "completed":
                            eval_loop = None
                            last_regular_eval = current_eval
                            if eval_manager is not None:
                                eval_manager.close()
                                eval_manager = None
                            train_manager.set_target_workers(train_workers)
                    train_status = train_loop.step()
                except Exception as exc:
                    if is_expected_ray_shutdown(exc):
                        logger.info("Stopping Ray actor loop because a Ray worker node is shutting down")
                        break
                    raise
                if train_status == "trainer_finished":
                    break
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception:
                logger.exception("Failed to finish W&B run")
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if eval_manager is not None:
            eval_manager.close()
        if train_manager is not None:
            train_manager.close()
        if ray_worker_log_collector is not None:
            close_ray_actor_best_effort(ray_worker_log_collector, logger, "Ray worker log collector")
            kill_ray_actor_best_effort(ray_worker_log_collector, logger, "Ray worker log collector")
        for dataset in (eval_dataset, train_dataset):
            close = getattr(dataset, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    logger.exception("Failed to close rollout dataset")
        if should_shutdown_ray and ray.is_initialized():
            try:
                ray.shutdown()
            except Exception:
                logger.exception("Failed to shut down Ray")
