from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from pipelinerl.async_llm import (
    MASKED_TOKEN_ID,
    extract_images_from_messages,
    get_processor,
    normalize_chat_template_messages,
)
from pipelinerl.rollouts import BaseMetrics, RolloutResult, TrainingText

if TYPE_CHECKING:
    from cube_harness.llm import LLMCall

logger = logging.getLogger(__name__)


def make_training_text(llm_tokenizer: Any, llm_call: LLMCall, processor_name: str | None = None) -> TrainingText:
    images = []
    use_processor = False
    visual_features = None
    assistant_msg: dict = {"role": "assistant", "content": llm_call.output.content or ""}
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
        processor = get_processor(processor_name or getattr(llm_tokenizer, "name_or_path", ""))

        try:
            prompt_text = processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            text = processor.apply_chat_template(
                full_messages,
                tokenize=False,
            )

            prompt_inputs = processor(
                text=processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                ),
                images=images,
                return_tensors=None,
            )
            prompt_token_ids = prompt_inputs["input_ids"][0]

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

    labels = llm_call.completion_token_ids
    logprobs = llm_call.logprobs
    input_ids = prompt_token_ids + labels
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
    if sequence_length > (max_length - buffer_tokens) and sequence_length <= max_length:
        return ((max_length - buffer_tokens) - sequence_length) / buffer_tokens
    return 0.


class CubeEpisodeRunner:
    """PipelineRL adapter for running cube-harness episodes.

    This class owns conversion from cube-harness trajectories into PipelineRL
    rollout outputs. Ray lifecycle and task scheduling stay in CubeRolloutWorker.
    """

    def __init__(
        self,
        *,
        llm_tokenizer: Any,
        llm_config: dict[str, Any],
        llm_router: Any | None,
        worker_name: str,
        runtime_context: Any | None,
        container_backend: Any | None,
        buffer_tokens: int = 0,
        discount_factor: float = 1.0,
        persist_rollout_artifacts: bool = False,
        rollout_artifact_dir: Path | None = None,
    ) -> None:
        self.llm_tokenizer = llm_tokenizer
        self.llm_config = llm_config
        self.llm_router = llm_router
        self.worker_name = worker_name
        self.runtime_context = runtime_context
        self.container_backend = container_backend
        self.buffer_tokens = int(buffer_tokens)
        self.discount_factor = float(discount_factor)
        self.persist_rollout_artifacts = bool(persist_rollout_artifacts)
        self.rollout_artifact_dir = rollout_artifact_dir

    def run_terminal(
        self,
        *,
        task: dict[str, Any],
        agent_config: Any,
        rollout_key: str,
    ) -> RolloutResult:
        """Run one complete cube-harness episode and return terminal training samples."""
        from cube_harness.episode import Episode, MAX_STEPS

        start = time.perf_counter()
        task_config = task["task_config"]
        agent_llm_config = getattr(agent_config, "llm_config")
        rollout_router = self.llm_router.with_affinity(rollout_key) if self.llm_router is not None else None
        if rollout_router is not None:
            agent_llm_config.router = rollout_router

        ep = Episode(
            id=0,
            output_dir="",
            agent_config=agent_config,
            task_config=task_config,
            exp_name="default",
            max_steps=task.get("max_steps", MAX_STEPS),
            persist_episode=False,
            runtime_context=self.runtime_context,
            container_backend=self.container_backend,
        )
        try:
            trajectory = ep.run()
        finally:
            try:
                if rollout_router is not None:
                    rollout_router.finish_affinity()
            except Exception:
                logger.warning("Failed to finish vLLM rollout affinity", exc_info=True)

        if self.persist_rollout_artifacts:
            self.write_rollout_artifact(trajectory, task)

        latency = time.perf_counter() - start
        return self.trajectory_to_rollout_result(
            trajectory=trajectory,
            task=task,
            agent_config=agent_config,
            latency=latency,
        )

    def trajectory_to_rollout_result(
        self,
        *,
        trajectory: Any,
        task: dict[str, Any],
        agent_config: Any,
        latency: float,
    ) -> RolloutResult:
        from cube.core import EnvironmentOutput
        from cube_harness.core import AgentOutput, TerminationReason

        task_config = task["task_config"]
        logger.info("Trajectory completed due to %s", trajectory.termination_reason)

        agent_outputs = [
            step.output
            for step in trajectory.steps
            if isinstance(step.output, AgentOutput)
        ]
        agent_llm_calls = sum(len(output.llm_calls) for output in agent_outputs)
        agent_errors = [output.error for output in agent_outputs if output.error is not None]
        if agent_errors:
            logger.error(
                "Cube rollout agent error: task_id=%s termination=%s steps=%d "
                "agent_outputs=%d llm_calls=%d error=%s",
                getattr(task_config, "task_id", None),
                trajectory.termination_reason,
                len(trajectory.steps),
                len(agent_outputs),
                agent_llm_calls,
                agent_errors[-1],
            )

        last_step = trajectory.steps[-1].output
        if not isinstance(last_step, EnvironmentOutput):
            raise ValueError(
                "Last step is expected to be an EnvironmentOutput because "
                f"Episode._run_loop() ends with evaluate(), got {type(last_step)}"
            )
        last_step_info = last_step.info

        final_reward = trajectory.reward_info["reward"]
        finished = trajectory.termination_reason == TerminationReason.ENV_DONE
        training_texts = self.trajectory_to_training_texts(
            trajectory=trajectory,
            task=task,
            final_reward=final_reward,
            finished=finished,
        )

        self.apply_rollout_rewards(
            training_texts=training_texts,
            agent_config=agent_config,
        )

        if not training_texts:
            logger.warning(
                "Cube rollout produced empty training_texts: task_id=%s termination=%s "
                "steps=%d agent_outputs=%d llm_calls=%d summary=%s",
                getattr(task_config, "task_id", None),
                trajectory.termination_reason,
                len(trajectory.steps),
                len(agent_outputs),
                agent_llm_calls,
                trajectory.summary_stats,
            )

        profiling = last_step_info.pop("profiling", {})
        filtered_last_step_info = {
            key: value
            for key, value in last_step_info.items()
            if isinstance(value, list) or isinstance(value, (float, bool, int, str))
        }
        metrics_kwargs = {
            "reward": final_reward,
            "success": bool(final_reward),
            "no_error": not bool(agent_errors),
            "no_answer": not bool(training_texts),
            "num_steps": len(training_texts),
            **filtered_last_step_info,
        }
        metrics = BaseMetrics(**metrics_kwargs)

        return RolloutResult(
            training_texts=training_texts,
            metrics=metrics,
            latency=latency,
            dataset_name=task["dataset"],
            domain=task["domain"],
        )

    def trajectory_to_training_texts(
        self,
        *,
        trajectory: Any,
        task: dict[str, Any],
        final_reward: float,
        finished: bool,
        validate_per_step: bool = False,
    ) -> list[TrainingText]:
        from cube.core import EnvironmentOutput
        from cube_harness.core import AgentOutput

        task_config = task["task_config"]
        training_texts: list[TrainingText] = []
        processor_name = self.llm_config.get("tokenizer_name", self.llm_config.get("model_name"))

        for step_i, step in enumerate(trajectory.steps):
            step_output = step.output
            if not isinstance(step_output, AgentOutput):
                continue

            step_reward = final_reward
            if validate_per_step:
                for candidate_step in trajectory.steps[step_i + 1 :]:
                    if isinstance(candidate_step.output, EnvironmentOutput):
                        step_reward = float(candidate_step.output.reward)
                        break

            for call_i, call in enumerate(step_output.llm_calls):
                training_text = make_training_text(
                    self.llm_tokenizer,
                    call,
                    processor_name=processor_name,
                )
                training_text.reward = step_reward
                training_text.finished = finished
                training_text.metadata.update(call.metadata or {})
                training_text.metadata.update(
                    {
                        "llm_call_id": call.id,
                        "llm_call_tag": call.tag,
                        "cube_id": task.get("domain"),
                        "task_id": getattr(task_config, "task_id", None),
                        "trajectory_id": trajectory.id,
                        "agent_step_index": step_i,
                        "llm_call_index": call_i,
                        "dataset_name": task.get("dataset"),
                        "termination_reason": str(trajectory.termination_reason),
                        "finish_reason": call.finish_reason,
                        "final_reward": final_reward,
                        "step_reward": step_reward,
                        "llm_prompt_tokens": call.prompt_tokens,
                        "llm_output_tokens": call.output_tokens,
                    }
                )
                training_texts.append(training_text)

        return training_texts

    def apply_rollout_rewards(
        self,
        *,
        training_texts: list[TrainingText],
        agent_config: Any,
    ) -> None:
        total_output_tokens = sum(getattr(c, "output_tokens", 0) for c in training_texts)
        if self.discount_factor != 1.0:
            for text in training_texts:
                text.reward *= self.discount_factor ** total_output_tokens

        agent_llm_config = getattr(agent_config, "llm_config")
        max_completion_tokens = int(getattr(agent_llm_config, "max_completion_tokens", 0))
        if self.buffer_tokens and max_completion_tokens > 0:
            len_reward = length_penalty(max_completion_tokens, total_output_tokens, self.buffer_tokens)
            for text in training_texts:
                text.reward += len_reward

    def write_rollout_artifact(self, trajectory: Any, task: dict[str, Any]) -> None:
        from cube_harness.core import AgentOutput

        if self.rollout_artifact_dir is None:
            return

        try:
            self.rollout_artifact_dir.mkdir(parents=True, exist_ok=True)
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
                self.rollout_artifact_dir / f"{self.worker_name}_{trajectory.id}_{time.time_ns()}.json"
            )
            artifact_path.write_text(json.dumps(payload, indent=2, default=str))
        except Exception:
            logger.exception("%s failed to write rollout artifact for %s", self.worker_name, trajectory.id)
