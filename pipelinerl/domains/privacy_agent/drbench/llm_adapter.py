"""PipelineRL LLM adapter for the vendored privacy_agent DRBench slice."""


from dataclasses import dataclass
from typing import Any

from pipelinerl.async_llm import make_training_text as make_rl_training_text
from pipelinerl.llm import LLMCall, Prompt, TrainableLLM

PLANNING_LOG_NAMES = {
    "query_planner",
    "action_plan_initial",
    "action_plan_fallback",
    "action_dependencies",
}
PLANNING_LOG_PREFIXES = ("adaptive_iter",)


@dataclass
class CapturedLLMCall:
    llm_call: LLMCall
    log_name: str
    requested_model: str | None


class PrivacyAgentLLMAdapter:
    """Route vendored DRBench prompts through the rollout LLM.

    The vendored agent code passes prompt strings into this adapter. The adapter keeps
    PipelineRL's normal token accounting and `TrainingText` creation while exposing a
    small interface that is easy for the copied DRBench modules to use.
    """

    def __init__(self, llm: TrainableLLM, capture_mode: str = "all_calls"):
        if capture_mode not in {"all_calls", "planning_only"}:
            raise ValueError(f"Unsupported capture_mode: {capture_mode}")

        self.llm = llm
        self.capture_mode = capture_mode
        self.captured_calls: list[CapturedLLMCall] = []
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.captured_prompt_tokens = 0
        self.captured_output_tokens = 0

    def _should_capture(self, log_name: str) -> bool:
        if self.capture_mode == "all_calls":
            return True
        if log_name in PLANNING_LOG_NAMES:
            return True
        return any(log_name.startswith(prefix) for prefix in PLANNING_LOG_PREFIXES)

    def _run_generation(self, prompt: Prompt, requested_model: str | None) -> LLMCall:
        if requested_model and requested_model != self.llm.model_name:
            raise ValueError(
                "privacy_agent routes all vendored DRBench calls through the rollout LLM. "
                f"Requested model '{requested_model}' does not match rollout model '{self.llm.model_name}'."
            )

        llm_stream = self.llm.generate(prompt)
        return llm_stream.get_llm_call()

    def generate_text(self, prompt: str, model: str | None = None, log_name: str = "llm") -> str:
        prompt_obj = Prompt(messages=[{"role": "user", "content": str(prompt)}])
        llm_call = self._run_generation(prompt_obj, requested_model=model)

        prompt_tokens = max(0, llm_call.prompt_length_tokens)
        output_tokens = max(0, llm_call.output_length_tokens)

        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_output_tokens += output_tokens

        if self._should_capture(log_name):
            self.captured_prompt_tokens += prompt_tokens
            self.captured_output_tokens += output_tokens
            self.captured_calls.append(
                CapturedLLMCall(
                    llm_call=llm_call,
                    log_name=log_name,
                    requested_model=model,
                )
            )

        return llm_call.output.content or ""

    def make_training_texts(self, group_id: str | None, base_metadata: dict[str, Any]) -> list:
        traces = []
        for idx, captured in enumerate(self.captured_calls):
            if self.llm.collect_logprobs and captured.llm_call.logprobs:
                trace = make_rl_training_text(self.llm, captured.llm_call)
            else:
                trace = self.llm.make_training_text(
                    captured.llm_call.prompt,
                    captured.llm_call.output,
                )
                trace.prompt_tokens = max(0, captured.llm_call.prompt_length_tokens)
                trace.output_tokens = max(0, captured.llm_call.output_length_tokens)

            trace.group_id = group_id
            trace.metadata.setdefault("privacy_agent", {}).update(
                {
                    **base_metadata,
                    "captured_call_index": idx,
                    "log_name": captured.log_name,
                    "requested_model": captured.requested_model,
                    "capture_mode": self.capture_mode,
                }
            )
            traces.append(trace)

        return traces
