from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from omegaconf import DictConfig
import ray
from pipelinerl.cube_rl.ray_worker_logging import (
    configure_ray_worker_logging,
    reset_worker_rollout_log_context,
    start_worker_rollout_log_context,
)
from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText
from pipelinerl.async_llm import (
    MASKED_TOKEN_ID,
    extract_images_from_messages,
    get_processor,
    normalize_chat_template_messages,
)

if TYPE_CHECKING:
    from cube_harness.llm import LLMCall

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CubeRuntimeSpec:
    cube_id: str
    benchmark_cfg: dict[str, Any]
    agent_cfg: dict[str, Any]
    split: str
    hint_condition: str | None = None


def _copy_model(obj: Any) -> Any:
    if hasattr(obj, "model_copy"):
        return obj.model_copy(deep=True)
    return obj


# rollout_key is built in cube_rl.launch._submit_one_rollout as
#   "{scheduler}:v{model_version}:g{group_id}:r{rollout_index}:{cube_id}:{task_id}".
# Dropping the ":r{rollout_index}" segment yields a key shared by all `attempts` of one
# GRPO group (same scheduler/version/group/cube/task), so every attempt hashes to one seed
# while different groups (different group_id, and different model_version each loop) differ.
_ROLLOUT_INDEX_RE = re.compile(r":r\d+:")


def _group_seed_from_rollout_key(rollout_key: str) -> int:
    """Stable non-negative 31-bit seed shared across a GRPO group's attempts.

    Uses a hashlib digest of the group key (forbidden: time/random) so the value is
    deterministic and identical for every attempt of the group. 31 bits keeps it a safe
    positive int for ``Math.seedrandom`` and JSON.
    """
    group_key = _ROLLOUT_INDEX_RE.sub(":", rollout_key, count=1)
    digest = hashlib.sha256(group_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def set_agent_llm_config(agent_config: Any, llm: dict) -> None:
    llm_config = getattr(agent_config, "llm_config")

    ## main config
    llm_config.api_base = llm["base_url"]
    if not llm_config.api_base.endswith("/v1"):
        llm_config.api_base += "/v1"

    llm_config.api_key = "EMPTY"
    llm_config.model_name = llm.get("served_model_name") or llm["model_name"]
    llm_config.tokenizer_name = llm.get("tokenizer_name", llm_config.model_name)
    if not llm_config.model_name.startswith("openai/"):
        llm_config.model_name = f"openai/{llm_config.model_name}"

    llm_config.logprobs = llm["collect_logprobs"]
    if llm_config.logprobs:
        llm_config.include_stop_str_in_output = True
        llm_config.skip_special_tokens = False

    # parameters config
    llm_parameters = llm.get("parameters", {})
    for param_name, param_value in llm_parameters.items():
        if hasattr(llm_config, param_name):
            setattr(llm_config, param_name, param_value)
        else:
            logger.warning(
                "Cube-harness Agent LLM parameters does not have attribute '%s', skipping",
                param_name,
            )


def _resolve_task_dataset_name(benchmark_obj: Any, task_config: Any) -> str:
    task_metadata = None
    benchmark_task_metadata = getattr(benchmark_obj, "task_metadata", None)
    if isinstance(benchmark_task_metadata, dict):
        task_metadata = benchmark_task_metadata.get(task_config.task_id)

    if task_metadata is not None:
        extra_info = getattr(task_metadata, "extra_info", None)
        if isinstance(extra_info, dict):
            dataset_name = extra_info.get("dataset")
            if dataset_name:
                return str(dataset_name)

    return ""


def make_training_text(llm_tokenizer: Any, llm_call: LLMCall) -> TrainingText:
    # Extract visual features if present
    images = []
    use_processor = False
    visual_features = None
    assistant_msg: dict = {
        "role": "assistant",
        "content": llm_call.output.content or "",
    }
    if llm_call.output.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in llm_call.output.tool_calls
        ]
    prompt_messages = normalize_chat_template_messages(llm_call.prompt.messages)
    full_messages = prompt_messages + [assistant_msg]

    if hasattr(llm_call.prompt, "messages"):
        images = extract_images_from_messages(prompt_messages)
        if images:
            use_processor = True

    if use_processor:
        # Use processor for vision-language models
        processor = get_processor(llm.model_name)

        try:
            # Apply chat template using processor for proper image token handling
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Create full conversation with assistant response
            text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
            )

            # Process prompt with images to get token IDs with image placeholders
            prompt_inputs = processor(
                text=processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                ),
                images=images,
                return_tensors=None,
            )

            # prompt_inputs["input_ids"] is a list of list
            prompt_token_ids = prompt_inputs["input_ids"][0]

            # Process images to get visual features
            processed = processor(
                text=[prompt_text], images=images, padding=True, return_tensors=None
            )
            visual_features = {
                key: value
                for key, value in processed.items()
                if isinstance(value, np.ndarray)
                and key not in ["input_ids", "attention_mask"]
            }

        except Exception as e:
            raise ValueError(f"Failed to process with vision-language processor: {e}")
    else:
        tools_kwarg = {"tools": llm_call.prompt.tools} if llm_call.prompt.tools else {}
        prompt_text = llm_tokenizer.apply_chat_template(
            conversation=prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **tools_kwarg,
        )

        text = llm_tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            **tools_kwarg,
        )
        prompt_token_ids = llm_tokenizer.apply_chat_template(
            prompt_messages,
            add_special_tokens=True,
            add_generation_prompt=True,
            **tools_kwarg,
        )

    output_text = text[len(prompt_text) :]

    tokenizer = processor.tokenizer if use_processor else llm_tokenizer

    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token) :]

    if not llm_call.logprobs:
        raise ValueError("Logprobs are required to make training data for RL")

    # We add the exact token ids and logprobs to "training_text" to ensure inference/training consistency
    labels = llm_call.completion_token_ids
    logprobs = llm_call.logprobs
    input_ids = prompt_token_ids + labels
    # Apply masking to input tokens that aren't generated
    labels = [MASKED_TOKEN_ID] * len(prompt_token_ids) + labels

    prompt_tokens = llm_call.prompt_tokens
    output_tokens = llm_call.output_tokens

    return TrainingText(
        text=text,
        n_predicted=len(output_text),
        input_ids=input_ids,
        labels=labels,
        logprobs=logprobs,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        visual_features=visual_features,
    )


def length_penalty(max_length: int, sequence_length: int, buffer_tokens: int) -> float:
    """
    Compute the overlong penalty
    """
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.0


@ray.remote(max_restarts=0, max_task_retries=0)
class CubeBenchmarkWorker:
    """Generic cube worker with lazy benchmark materialization.

    Interface:
    - setup()
    - rollout(cube_id, task_id)
    - health()
    - close()
    """

    def __init__(
        self,
        *,
        # do not save the ref to the whole cfg
        cfg: DictConfig,
        cube_specs: list[dict[str, Any]],
        seed: int,
        worker_name: str,
        llm: dict[str, Any],
        test_llm: dict[str, Any] | None = None,
        llm_router: Any | None = None,
        ray_worker_log_collector: Any | None = None,
        hint_refresh_actor: Any | None = None,
    ):
        self._cube_specs = {
            str(spec["cube_id"]): CubeRuntimeSpec(
                cube_id=str(spec["cube_id"]),
                benchmark_cfg=spec["benchmark_cfg"],
                agent_cfg=spec["agent_cfg"],
                split=str(spec["split"]),
                hint_condition=spec.get("hint_condition"),
            )
            for spec in cube_specs
        }
        self._seed = int(seed)
        self._worker_name = worker_name
        self._train_llm = llm
        self._test_llm = test_llm or llm
        self._llm_router = llm_router
        self._ray_worker_log_collector = ray_worker_log_collector
        # In-training hint refresh (Exp 1.5 / v2): live hint maps fetched from the
        # HintRefreshActor, cached per (condition, version). version=None entries (no
        # per-group pin from the scheduler) expire after a TTL as an approximation.
        self._hint_refresh_actor = hint_refresh_actor
        hint_refresh_cfg = getattr(getattr(cfg, "cube_params", {}), "hint_refresh", None)
        self._hint_cache_ttl_s = float(getattr(hint_refresh_cfg, "cache_ttl_s", 30.0) if hint_refresh_cfg else 30.0)
        self._hint_cache: dict[tuple[str, int | None], tuple[float, int, dict[str, str]]] = {}

        worker_log_level = str(getattr(cfg.actor, "ray_worker_log_level", "ERROR"))
        worker_litellm_log_level = str(
            getattr(cfg.actor, "ray_worker_litellm_log_level", "WARNING")
        )

        configure_ray_worker_logging(
            worker_name=self._worker_name,
            log_collector=self._ray_worker_log_collector,
            log_level=worker_log_level,
            litellm_log_level=worker_litellm_log_level,
        )

        self._ready = False
        self._setup_error: str | None = None

        self._current_cube_id: str | None = None
        self._benchmark = None
        self._task_by_id: dict[str, dict] = {}

        self._runtime_context = None
        self._container_backend = None
        self._llm_tokenizer = None

        # Multi-instance MiniWoB: when true, the per-rollout path overrides the deep-copied
        # task_config.seed with a stable per-GRPO-group seed (all `attempts` of a group share
        # one seed -> one instance; different groups -> different instances). Default False
        # leaves task_config.seed at its cube default (42), reproducing the historical pin.
        self._vary_instance_seed = bool(getattr(cfg.actor, "vary_instance_seed", False))

        ## Optional config for extra reward
        self._buffer_tokens = int(getattr(cfg.actor, "buffer_tokens", 0))
        self._discount_factor = float(getattr(cfg.actor, "discount_factor", 1.0))
        ## LaMer multi-episode reflection rollout (k_episodes==1 -> classic single-episode rollout).
        self._lamer_k_episodes = int(getattr(cfg.actor, "lamer_k_episodes", 1))
        # Eval/test rollouts may use a different episode budget (0 = same as train). Lets a
        # single-episode baseline (train k=1) be evaluated at k>1 i.i.d. for a held-out pass@1..k curve.
        self._eval_k_episodes = int(getattr(cfg.actor, "eval_k_episodes", 0) or 0)
        self._lamer_discount_gamma = float(
            getattr(cfg.actor, "lamer_discount_gamma", 1.0)
        )
        # Credit scheme: "outcome" (legacy eventual-best; math runs depend on it) or "forward"
        # (LaMer paper Eq. 4 cross-episode return-to-go — discounts reflections by distance to success).
        self._lamer_credit_scheme = str(getattr(cfg.actor, "lamer_credit_scheme", "outcome"))
        artifact_cfg = getattr(
            getattr(cfg, "cube_params", {}), "rollout_artifacts", None
        )
        self._persist_rollout_artifacts = (
            bool(getattr(artifact_cfg, "enabled", False)) if artifact_cfg else False
        )
        artifact_dir = getattr(artifact_cfg, "path", None) if artifact_cfg else None
        self._rollout_artifact_dir = (
            Path(artifact_dir)
            if artifact_dir
            else Path(cfg.output_dir) / "actor" / "rollout_artifacts"
        )

    def setup(self) -> dict[str, Any]:
        try:
            from pipelinerl.llm import TrainableLLM

            temp_llm = TrainableLLM(**self._train_llm)
            temp_llm.load_tokenizer()
            self._llm_tokenizer = temp_llm.tokenizer
            self._ready = True
            self._setup_error = None
            logger.info(
                "%s ready with %d cube specs", self._worker_name, len(self._cube_specs)
            )
            return self.health()
        except Exception as exc:
            self._ready = False
            self._setup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s failed during setup", self._worker_name)
            raise

    def _close_current_cube(self) -> None:
        if self._benchmark is not None:
            try:
                self._benchmark.close()
            except Exception as exc:
                logger.warning(
                    "%s failed to close cube %s: %s",
                    self._worker_name,
                    self._current_cube_id,
                    exc,
                )
        self._current_cube_id = None
        self._benchmark = None
        self._task_by_id = {}
        self._runtime_context = None
        self._container_backend = None

    def _prepare_cube(self, cube_id: str) -> None:
        if self._current_cube_id == cube_id:
            return
        if cube_id not in self._cube_specs:
            raise KeyError(f"Unknown cube_id: {cube_id}")

        import hydra

        self._close_current_cube()
        spec = self._cube_specs[cube_id]
        benchmark_obj = hydra.utils.instantiate(spec.benchmark_cfg)
        benchmark_obj.install()
        benchmark_obj.setup()

        self._runtime_context = getattr(benchmark_obj, "_runtime_context", None)
        self._container_backend = getattr(benchmark_obj, "container_backend", None)
        agent_cfg_template = hydra.utils.instantiate(spec.agent_cfg)
        task_llm = self._test_llm if spec.split == "test" else self._train_llm

        task_by_id: dict[str, dict] = {}
        for task_config in benchmark_obj.get_task_configs():
            agent_config = _copy_model(agent_cfg_template)
            set_agent_llm_config(agent_config, task_llm)
            dataset_name = _resolve_task_dataset_name(benchmark_obj, task_config)
            task_by_id[task_config.task_id] = {
                "task_config": _copy_model(task_config),
                "agent_config": agent_config,
                "domain": cube_id,
                "dataset": cube_id if not dataset_name else f"{cube_id}/{dataset_name}",
            }

        self._current_cube_id = cube_id
        self._benchmark = benchmark_obj
        self._task_by_id = task_by_id
        logger.info(
            "%s prepared cube %s with %d tasks",
            self._worker_name,
            cube_id,
            len(task_by_id),
        )

    def rollout(
        self,
        *,
        cube_id: str,
        task_id: str,
        rollout_key: str,
        is_training: bool = True,
        hint_version: int | None = None,
    ) -> dict:
        rollout_log_context = start_worker_rollout_log_context(f"{cube_id}:{task_id}")
        try:
            if not self._ready:
                raise RuntimeError(f"{self._worker_name} not ready")
            self._prepare_cube(cube_id)
            if task_id not in self._task_by_id:
                raise KeyError(f"Unknown task_id for cube {cube_id}: {task_id}")

            base_task = self._task_by_id[task_id]
            spec = self._cube_specs[cube_id]
            # Test rollouts use eval_k_episodes when set (e.g. a k=1 baseline evaluated at k>1 i.i.d.);
            # train rollouts (and unset eval_k) use the training k.
            k_episodes = self._eval_k_episodes if (not is_training and self._eval_k_episodes) else self._lamer_k_episodes
            task = {
                "task_config": _copy_model(base_task["task_config"]),
                "agent_config": _copy_model(base_task["agent_config"]),
                "domain": base_task.get("domain", None),
                "dataset": base_task.get("dataset", None),
                "runtime_context": self._runtime_context,
                "container_backend": self._container_backend,
                "k_episodes": k_episodes,
                # Hint refresh: train-split episodes feed the miner's transcript buffers.
                "report_hints": self._hint_refresh_actor is not None and spec.split == "train" and is_training,
            }
            if self._hint_refresh_actor is not None and spec.hint_condition is not None:
                version, hints = self._fetch_hints(spec.hint_condition, hint_version)
                if version > 0:  # version 0 = no refresh yet; keep the statically seeded bank hints
                    task["agent_config"].task_hints = dict(hints)

            result = self._rollout(task=_copy_model(task), rollout_key=rollout_key)
            return result.model_dump()
        except Exception:
            logger.exception(
                "%s rollout failed for cube_id=%s task_id=%s; dropping (empty result, will be retried)",
                self._worker_name,
                cube_id,
                task_id,
            )
            # Don't crash the actor on a single failed rollout (e.g. a transient vLLM 5xx — whose
            # litellm exception also can't be unpickled across Ray, so re-raising surfaces as an
            # opaque RaySystemError). Return an empty result; the train loop drops + retries it.
            return RolloutResult(
                training_texts=[],
                metrics=BaseMetrics(
                    reward=0.0, success=False, no_error=False, no_answer=True
                ),
                latency=0.0,
            ).model_dump()
        finally:
            reset_worker_rollout_log_context(rollout_log_context)

    def _fetch_hints(self, condition: str, hint_version: int | None) -> tuple[int, dict[str, str]]:
        """Hint map for one condition from the HintRefreshActor, cached per (condition, version).

        ``hint_version`` is the per-GRPO-group pin from the scheduler — exact-version
        entries never expire (the actor keeps a short map history), so every rollout of
        a group sees identical hints. ``hint_version=None`` falls back to a TTL cache.
        """
        key = (condition, hint_version)
        now = time.monotonic()
        cached = self._hint_cache.get(key)
        if cached is not None and (hint_version is not None or now - cached[0] < self._hint_cache_ttl_s):
            return cached[1], cached[2]
        try:
            version, hints = ray.get(self._hint_refresh_actor.get_hints.remote(condition, hint_version))
        except Exception:
            logger.warning("%s failed to fetch hints for condition %s", self._worker_name, condition, exc_info=True)
            return (cached[1], cached[2]) if cached is not None else (0, {})
        if len(self._hint_cache) > 64:
            self._hint_cache.clear()
        self._hint_cache[key] = (now, int(version), dict(hints))
        return int(version), dict(hints)

    def _episode_training_texts(
        self, trajectory: Any, task: dict, episode_index: int
    ) -> tuple[list, float, list, dict]:
        """One episode's trajectory -> TrainingTexts (reward assigned later at the rollout level).

        Returns (training_texts, episode_reward, agent_errors, last_step_info). Each llm_call
        becomes a training text tagged with ``episode_index``; the finalize reflection LLMCall
        (an AgentOutput step) is included like any other, so reflections are trainable.
        """
        from cube.core import EnvironmentOutput
        from cube_harness.core import AgentOutput, TerminationReason

        task_config = task["task_config"]
        agent_outputs = [
            step.output
            for step in trajectory.steps
            if isinstance(step.output, AgentOutput)
        ]
        agent_errors = [
            output.error for output in agent_outputs if output.error is not None
        ]
        if agent_errors:
            logger.error(
                "Cube rollout agent error: task_id=%s episode=%d termination=%s error=%s",
                getattr(task_config, "task_id", None),
                episode_index,
                trajectory.termination_reason,
                agent_errors[-1],
            )

        # The final evaluate() EnvironmentOutput carries task metrics. It is the last step for a
        # plain agent, but a LaMer agent's finalize() appends a reflection AgentOutput AFTER it,
        # so take the LAST EnvironmentOutput rather than steps[-1].
        env_outputs = [
            step.output
            for step in trajectory.steps
            if isinstance(step.output, EnvironmentOutput)
        ]
        if not env_outputs:
            raise ValueError("trajectory has no EnvironmentOutput (evaluate) step")
        last_step_info = env_outputs[-1].info

        episode_reward = trajectory.reward_info["reward"]
        finished = trajectory.termination_reason == TerminationReason.ENV_DONE
        training_texts = []
        for step_i, step in enumerate(trajectory.steps):
            step_output = step.output
            if not isinstance(step_output, AgentOutput):
                continue
            for call_i, call in enumerate(step_output.llm_calls):
                training_text = make_training_text(self._llm_tokenizer, call)
                training_text.finished = (
                    finished  # reward set later (rollout-level credit)
                )
                training_text.metadata.update(call.metadata or {})
                training_text.metadata.update(
                    {
                        "llm_call_id": call.id,
                        "llm_call_tag": call.tag,
                        "cube_id": task.get("domain"),
                        "task_id": getattr(task_config, "task_id", None),
                        "trajectory_id": trajectory.id,
                        "episode_index": episode_index,
                        "agent_step_index": step_i,
                        "llm_call_index": call_i,
                        "dataset_name": task.get("dataset"),
                        "termination_reason": str(trajectory.termination_reason),
                        "finish_reason": call.finish_reason,
                        "episode_reward": episode_reward,
                        "llm_prompt_tokens": call.prompt_tokens,
                        "llm_output_tokens": call.output_tokens,
                    }
                )
                training_texts.append(training_text)
        return training_texts, episode_reward, agent_errors, last_step_info

    def _rollout(self, task: dict, rollout_key: str) -> RolloutResult:
        from cube_harness.agents.lamer import episodes_to_success, lamer_rollout_credit, run_multi_episode_rollout
        from cube_harness.jefhinter import render_trajectory

        start = time.perf_counter()

        task_config = task["task_config"]
        agent_config = task["agent_config"]
        # Multi-instance: override the deep-copied task_config seed with a per-GRPO-group
        # seed so all `attempts` of a group share one instance, but groups differ. Only when
        # the flag is on AND the cube's task_config exposes a `seed` (e.g. MiniWoB); the
        # task_id is left untouched so hint-keying is unaffected.
        if self._vary_instance_seed and hasattr(task_config, "seed"):
            task_config.seed = _group_seed_from_rollout_key(rollout_key)
        agent_llm_config = getattr(agent_config, "llm_config")
        rollout_router = self._llm_router.with_affinity(rollout_key)
        agent_llm_config.router = rollout_router

        k_episodes = max(1, int(task.get("k_episodes", self._lamer_k_episodes)))
        training_texts: list = []
        episode_rewards: list[float] = []
        agent_errors: list = []
        last_step_info: dict = {}
        try:
            # The multi-episode loop, per-rollout cross-episode memory file, and early-stop-on-success
            # are the LaMer agent's shared helper (the same one the recipes use) — single source of the
            # loop. domain stays generic: it only maps each returned trajectory onto TrainingTexts +
            # credit below.
            trajectories = run_multi_episode_rollout(
                agent_config,
                task_config,
                self._runtime_context,
                k_episodes,
                exp_name="default",
                persist_episode=False,
                container_backend=self._container_backend,
            )
            for episode_index, trajectory in enumerate(trajectories):
                if self._persist_rollout_artifacts:
                    self._write_rollout_artifact(trajectory, task)
                ep_texts, ep_reward, ep_errors, ep_info = self._episode_training_texts(
                    trajectory, task, episode_index
                )
                training_texts.extend(ep_texts)
                episode_rewards.append(ep_reward)
                agent_errors.extend(ep_errors)
                last_step_info = ep_info
                if task.get("report_hints"):
                    # Fire-and-forget: feed the hint miner's rolling transcript buffers.
                    try:
                        self._hint_refresh_actor.report.remote(
                            str(getattr(task_config, "task_id", "")),
                            float(ep_reward),
                            render_trajectory(trajectory),
                        )
                    except Exception:
                        logger.warning("Failed to report episode transcript for hint refresh", exc_info=True)
                logger.info(
                    "Episode %d/%d done: reward=%s termination=%s",
                    episode_index + 1,
                    len(trajectories),
                    ep_reward,
                    trajectory.termination_reason,
                )
        finally:
            try:
                rollout_router.finish_affinity()
            except Exception:
                logger.warning("Failed to finish vLLM rollout affinity", exc_info=True)

        # Rollout-level credit is the LaMer agent's policy (cube_harness.lamer_rollout_credit), selected
        # by self._lamer_credit_scheme: "outcome" (legacy eventual-best, math runs) or "forward" (paper
        # Eq. 4 cross-episode return-to-go). domain stays generic — it only maps training texts onto that
        # pure policy. outcome_reward below is the rollout headline used only for metrics.
        outcome_reward = max(episode_rewards) if episode_rewards else 0.0
        turns = [
            (int(t.metadata.get("episode_index", 0)), t.metadata.get("llm_call_tag") == "reflection")
            for t in training_texts
        ]
        credit = lamer_rollout_credit(episode_rewards, turns, gamma=self._lamer_discount_gamma, scheme=self._lamer_credit_scheme)
        for t, r in zip(training_texts, credit):
            t.reward = r

        # Extra rewards (preserved): token-length discount + length penalty across all texts.
        total_output_tokens = sum(
            getattr(c, "output_tokens", 0) for c in training_texts
        )
        if self._discount_factor != 1.0:
            for t in training_texts:
                t.reward *= self._discount_factor**total_output_tokens
        max_completion_tokens = int(
            getattr(agent_llm_config, "max_completion_tokens", 0)
        )
        if self._buffer_tokens and max_completion_tokens > 0:
            len_reward = length_penalty(
                max_completion_tokens, total_output_tokens, self._buffer_tokens
            )
            for t in training_texts:
                t.reward += len_reward

        if not training_texts:
            logger.warning(
                "Cube rollout produced empty training_texts: task_id=%s episodes=%d episode_rewards=%s",
                getattr(task_config, "task_id", None),
                len(episode_rewards),
                episode_rewards,
            )

        latency = time.perf_counter() - start
        last_step_info.pop("profiling", {})
        # BaseMetrics requires success/no_error/no_answer. Derive them generically so cubes
        # whose task info doesn't emit them (e.g. arithmetic-cube) work; task-provided values
        # in last_step_info still override these defaults (e.g. math-tool-use).
        # episodes_to_success (cube_harness.agents.lamer): attempts until first solve, 0 if never.
        # Logged as episodes_to_success_mean = average attempts-to-success, blended with failures-as-0.
        metrics_kwargs = {
            "reward": outcome_reward,
            "num_steps": len(training_texts),
            "episodes_to_success": episodes_to_success(episode_rewards),
            "success": bool(outcome_reward > 0),
            "no_error": len(agent_errors) == 0,
            "no_answer": False,
            **last_step_info,
        }
        metrics = BaseMetrics(**metrics_kwargs)

        return RolloutResult(
            training_texts=training_texts,
            metrics=metrics,
            latency=latency,
            dataset_name=task["dataset"],
            domain=task["domain"],
        )

    def _write_rollout_artifact(self, trajectory: Any, task: dict) -> None:
        from cube_harness.core import AgentOutput

        try:
            self._rollout_artifact_dir.mkdir(parents=True, exist_ok=True)
            llm_calls = []
            for step_i, step in enumerate(trajectory.steps):
                if not isinstance(step.output, AgentOutput):
                    continue
                for call_i, call in enumerate(step.output.llm_calls):
                    llm_calls.append(
                        {
                            "step_index": step_i,
                            "llm_call_index": call_i,
                            "llm_call_id": call.id,
                            "tag": call.tag,
                            "timestamp": call.timestamp,
                            "prompt_tokens": call.prompt_tokens,
                            "output_tokens": call.output_tokens,
                            "finish_reason": call.finish_reason,
                            "metadata": call.metadata,
                        }
                    )

            payload = {
                "trajectory_id": trajectory.id,
                "cube_id": task.get("domain"),
                "dataset_name": task.get("dataset"),
                "task_id": getattr(task.get("task_config"), "task_id", None),
                "termination_reason": str(trajectory.termination_reason),
                "reward_info": trajectory.reward_info,
                "summary_stats": trajectory.summary_stats,
                "n_steps": len(trajectory.steps),
                "llm_calls": llm_calls,
            }
            artifact_path = (
                self._rollout_artifact_dir
                / f"{self._worker_name}_{trajectory.id}_{time.time_ns()}.json"
            )
            artifact_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            logger.exception(
                "%s failed to write rollout artifact for %s",
                self._worker_name,
                trajectory.id,
            )

    def health(self) -> dict[str, Any]:
        return {
            "worker_name": self._worker_name,
            "ready": self._ready,
            "current_cube_id": self._current_cube_id,
            "n_tasks": len(self._task_by_id),
            "error": self._setup_error,
        }

    def close(self) -> None:
        self._close_current_cube()
