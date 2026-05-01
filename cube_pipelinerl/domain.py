from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

import ray
from cube_pipelinerl.ray_worker_logging import (
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

def _copy_model(obj: Any) -> Any:
    if hasattr(obj, "model_copy"):
        return obj.model_copy(deep=True)
    return obj

def _configure_agent_llm(agent_config: Any, llm: dict) -> None:
    llm_config = getattr(agent_config, "llm_config")

    ## main config
    llm_config.api_base = llm["base_url"]
    if not llm_config.api_base.endswith("/v1"):
        llm_config.api_base += "/v1"

    llm_config.api_key = "EMPTY"
    llm_config.model_name = llm["served_model_name"] or llm["model_name"]
    if not llm_config.model_name.startswith("openai/"):
        llm_config.model_name = f"openai/{llm_config.model_name}"

    llm_config.logprobs = llm['collect_logprobs']
    if llm_config.logprobs:
        llm_config.include_stop_str_in_output = True
        llm_config.skip_special_tokens = False

    # parameters config
    llm_parameters = llm.get("parameters", {})
    for param_name, param_value in llm_parameters.items():
        if hasattr(llm_config, param_name):
            setattr(llm_config, param_name, param_value)
        else:
            logger.warning("Cube-harness Agent LLM parameters does not have attribute '%s', skipping", param_name)

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

@ray.remote(max_restarts=0, max_task_retries=0)
class CubeBenchmarkActor:
    """Cube benchmark lifecycle + rollout execution actor.

    Interface:
    - setup()
    - get_task_ids()
    - rollout(task_id, llm)
    - health()
    - close()
    """

    def __init__(
        self,
        *,
        benchmark_cfg: dict,
        agent_cfg: dict,
        cube_name: str,
        train_dataset_names: list[str],
        test_dataset_names: list[str],
        seed: int,
        actor_name: str,
        ray_worker_log_collector: Any | None = None,
        ray_worker_log_level: str = "ERROR",
        litellm_log_level: str = "WARNING",
    ):
        self._benchmark_cfg = benchmark_cfg
        self._agent_cfg = agent_cfg
        self._cube_name = cube_name
        self._seed = int(seed)
        self._actor_name = actor_name
        self._ray_worker_log_collector = ray_worker_log_collector
        self._ray_worker_log_level = ray_worker_log_level
        self._litellm_log_level = litellm_log_level
        configure_ray_worker_logging(
            actor_name=self._actor_name,
            log_collector=self._ray_worker_log_collector,
            log_level=self._ray_worker_log_level,
            litellm_log_level=self._litellm_log_level,
        )

        self._ready = False
        self._setup_error: str | None = None

        self._benchmark = None
        self._task_ids: list[str] = []
        self._task_by_id: dict[str, dict] = {}
        self._train_task_ids: list[str] = []
        self._test_task_ids: list[str] = []

        self._runtime_context = None
        self._container_backend = None
        self._llm_tokenizer = None

        self._train_dataset_names = train_dataset_names
        self._test_dataset_names = test_dataset_names

    def setup(self) -> dict[str, Any]:
        import hydra

        try:
            benchmark_obj = hydra.utils.instantiate(self._benchmark_cfg)
            benchmark_obj.install()
            benchmark_obj.setup()

            self._runtime_context = getattr(benchmark_obj, "_runtime_context", None)
            self._container_backend = getattr(benchmark_obj, "container_backend", None)

            agent_cfg_template = hydra.utils.instantiate(self._agent_cfg)

            all_task_configs = list(benchmark_obj.get_task_configs())

            self._train_task_ids = []
            self._test_task_ids = []
            task_by_id: dict[str, dict] = {}

            for task_config in all_task_configs:
                task_id = task_config.task_id
                dataset_name = _resolve_task_dataset_name(benchmark_obj, task_config)
                task_by_id[task_config.task_id] = {
                    "task_config": _copy_model(task_config),
                    "agent_config": _copy_model(agent_cfg_template),
                    "domain": self._cube_name,
                    "dataset": dataset_name if dataset_name else self._cube_name,
                }
                if dataset_name in self._train_dataset_names or dataset_name == "":
                    self._train_task_ids.append(task_id)
                elif dataset_name in self._test_dataset_names:
                    self._test_task_ids.append(task_id)
                else:
                    logger.warning(
                        "%s task_id %s has dataset '%s' which is not in train or test dataset lists, assigning to train by default",
                        self._actor_name,
                        task_id,
                        dataset_name,
                    )
                    self._train_task_ids.append(task_id)
        
            self._task_by_id = task_by_id
            self._task_ids = list(self._task_by_id.keys())
            self._benchmark = benchmark_obj
            self._ready = True
            self._setup_error = None
            logger.info("%s ready with %d tasks", self._actor_name, len(self._task_ids))
            return self.health()
        except Exception as exc:
            self._ready = False
            self._setup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("%s failed during setup", self._actor_name)
            raise

    def get_task_ids(self) -> list[str]:
        return list(self._task_ids)

    def get_train_task_ids(self) -> list[str]:
        return self._train_task_ids

    def get_test_task_ids(self) -> list[str]:
        return self._test_task_ids

    def rollout(self, task_id: str, llm: dict) -> dict:
        from pipelinerl.llm import TrainableLLM

        rollout_log_context = start_worker_rollout_log_context(str(task_id))
        try:
            if not self._ready:
                raise RuntimeError(f"{self._actor_name} not ready")
            if task_id not in self._task_by_id:
                raise KeyError(f"Unknown task_id: {task_id}")

            if self._llm_tokenizer is None:
                temp_llm = TrainableLLM(**llm)
                temp_llm.load_tokenizer()
                self._llm_tokenizer = temp_llm.tokenizer

            base_task = self._task_by_id[task_id]
            task = {
                "task_config": _copy_model(base_task["task_config"]),
                "agent_config": _copy_model(base_task["agent_config"]),
                "domain": base_task.get("domain", None),
                "dataset": base_task.get("dataset", None),
                "runtime_context": self._runtime_context,
                "container_backend": self._container_backend,
            }

            result = self._rollout(task=_copy_model(task), llm=llm)
            return result.model_dump()
        except Exception:
            logger.exception("%s rollout failed for task_id=%s", self._actor_name, task_id)
            raise
        finally:
            reset_worker_rollout_log_context(rollout_log_context)

    def _rollout(self, task: dict, llm: dict) -> RolloutResult:
        from cube_harness.episode import Episode, MAX_STEPS
        from cube.core import EnvironmentOutput
        from cube_harness.core import AgentOutput, TerminationReason

        start = time.perf_counter()

        task_config = task["task_config"]
        agent_config = task["agent_config"]
        _configure_agent_llm(agent_config, llm)
        validate_per_step = False

        ep = Episode(
            id=0,
            output_dir="",
            agent_config=agent_config,
            task_config=task_config,
            exp_name="default",
            max_steps=MAX_STEPS,
            persist_episode=False,
            runtime_context=self._runtime_context,
            container_backend=self._container_backend,
        )
        trajectory = ep.run()
        logger.info(f"Trajectory completed due to {trajectory.termination_reason}")
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

        # last step is always an EnvironmentOutput since Episode._run_loop() ends with evaluate method.
        last_step = trajectory.steps[-1].output
        if not isinstance(last_step, EnvironmentOutput):
            raise ValueError(f"""Last step is always an EnvironmentOutput
                              since Episode._run_loop() ends with evaluate method., got {type(last_step)}""")
        last_step_info = last_step.info

        final_reward = trajectory.reward_info['reward']
        finished = trajectory.termination_reason == TerminationReason.ENV_DONE
        training_texts = []
        # trajectory.steps contain a list of AgentOutput/EnvironmentOutput objects in the order they were executed. \\
        # Within an AgentOutput there is a list of llm_calls. for each llm_call \\
        # we want to capture a training example, and assign it a reward value. If validate_per_step is True, we instead \\ 
        # assign each llm_call the reward of the EnvironmentOutput that immediately follows it, which allows for per-step rewards if the task provides them. Otherwise, we assign all calls the final reward of the trajectory.
        for step_i, step in enumerate(trajectory.steps):
            step_output = step.output
            if isinstance(step_output, AgentOutput):
                step_reward = final_reward
                if validate_per_step:
                    for j in trajectory.steps[step_i + 1:]:
                        if isinstance(j, EnvironmentOutput):
                            step_reward = float(j.reward)
                            break
                
                for call in step_output.llm_calls:
                    training_text = make_training_text(self._llm_tokenizer, call)
                    training_text.reward = step_reward
                    training_text.finished = finished
                    training_texts.append(training_text)

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

        latency = time.perf_counter() - start
        profiling = last_step_info.pop("profiling", {})
        metrics_kwargs = {'reward': final_reward, 'num_steps': len(training_texts), **last_step_info}
        metrics = BaseMetrics(**metrics_kwargs)

        return RolloutResult(
            training_texts=training_texts,
            metrics=metrics,
            latency=latency,
            dataset_name=task["dataset"],
            domain=task["domain"],
        )

    def health(self) -> dict[str, Any]:
        return {
            "actor_name": self._actor_name,
            "ready": self._ready,
            "n_tasks": len(self._task_ids),
            "domain": self._cube_name,
            "error": self._setup_error,
        }

    def close(self) -> None:
        if self._benchmark is not None:
            try:
                self._benchmark.close()
            except Exception as exc:
                logger.warning("%s failed to close benchmark: %s", self._actor_name, exc)
