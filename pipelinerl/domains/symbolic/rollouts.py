from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Sequence

import sympy as sp
from sympy.core.sympify import SympifyError

from pipelinerl.rollouts import BaseMetrics, RolloutResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import aiohttp
    from omegaconf import DictConfig
    from tapeagents.llms.trainable import TrainableLLM
    from pipelinerl.domains.math.rollouts import RewardTable

_DEFAULT_MAX_LEN = 1200
_ALLOWED_FUNCS = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
}
_ALLOWED_CONSTS = {
    "pi": sp.pi,
    "E": sp.E,
}


@dataclass
class SymbolicEvaluation:
    status: str
    error: str | None = None


def extract_boxed_expression(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^}]*)\}", text, flags=re.DOTALL)
    if not matches:
        return None
    candidate = matches[-1].strip()
    return candidate or None


def _build_locals(variable_names: Sequence[str]) -> dict:
    locals_dict: dict[str, sp.Basic] = {}
    for name in variable_names:
        symbol = sp.symbols(name)
        locals_dict[name] = symbol
    locals_dict.update(_ALLOWED_FUNCS)
    locals_dict.update(_ALLOWED_CONSTS)
    return locals_dict


def _parse_expression(expr_text: str, variable_names: Sequence[str]) -> sp.Expr:
    normalized = expr_text.replace("^", "**").strip()
    locals_dict = _build_locals(variable_names)
    return sp.sympify(normalized, locals=locals_dict, evaluate=True)


def expressions_equivalent(lhs: sp.Expr, rhs: sp.Expr) -> bool:
    try:
        delta = sp.simplify(sp.expand(lhs - rhs))
        return bool(delta == 0)
    except Exception:
        return False


def evaluate_symbolic_prediction(
    problem: dict,
    prediction: str | None,
    max_prediction_length: int = _DEFAULT_MAX_LEN,
) -> SymbolicEvaluation:
    candidate_text = extract_boxed_expression(prediction)
    if not candidate_text:
        return SymbolicEvaluation(status="no_answer", error="missing_boxed_answer")
    if len(candidate_text) > max_prediction_length:
        return SymbolicEvaluation(status="unparsable", error="too_long")
    variables = problem.get("variables") or ["x"]
    try:
        expected_expr = _parse_expression(problem["target"], variables)
    except (KeyError, SympifyError, ValueError) as exc:
        return SymbolicEvaluation(status="unparsable", error=f"invalid_gold:{exc}")
    try:
        predicted_expr = _parse_expression(candidate_text, variables)
    except SympifyError as exc:
        return SymbolicEvaluation(status="unparsable", error=f"sympify_error:{exc}")
    except Exception as exc:  # pragma: no cover - safety net
        return SymbolicEvaluation(status="unparsable", error=str(exc))
    if expressions_equivalent(expected_expr, predicted_expr):
        return SymbolicEvaluation(status="correct")
    return SymbolicEvaluation(status="wrong")


def _score_status(status: str, finished: bool, rewards: "RewardTable") -> float:
    match (status, finished):
        case ("wrong", False):
            return rewards.wrong_answer_not_finished
        case ("wrong", True):
            return rewards.wrong_answer_finished
        case ("no_answer", False):
            return rewards.no_answer_not_finished
        case ("no_answer", True):
            return rewards.no_answer_finished
        case ("unparsable", False):
            return rewards.unparsable_not_finished
        case ("unparsable", True):
            return rewards.unparsable_finished
        case ("correct", False):
            return rewards.correct_answer_not_finished
        case ("correct", True):
            return rewards.correct_answer_finished
        case _:
            raise ValueError(f"Unsupported symbolic status '{status}'")


async def generate_symbolic_rollout(
    cfg: "DictConfig",
    llm: "TrainableLLM",
    problem: dict,
    session: "aiohttp.ClientSession",
) -> RolloutResult:
    from pipelinerl.async_llm import llm_async_generate, make_training_text
    from pipelinerl.domains.math.rollouts import RewardTable, length_penalty
    from tapeagents.core import Prompt

    messages = []
    if cfg.actor.system_prompt:
        messages.append({"role": "system", "content": cfg.actor.system_prompt})
    messages.append({"role": "user", "content": problem["task"]})
    prompt = Prompt(messages=messages)

    t0 = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - t0

    rewards = RewardTable(**dict(cfg.rewards))
    trace = make_training_text(llm, llm_call)
    discount_factor = cfg.actor.discount_factor
    max_len = getattr(cfg.actor, "symbolic_max_prediction_length", _DEFAULT_MAX_LEN)

    evaluation = evaluate_symbolic_prediction(problem, llm_call.output.content, max_prediction_length=max_len)
    reward = _score_status(evaluation.status, trace.finished, rewards)
    reward *= discount_factor ** llm_call.output_length_tokens
    overlong_penalty = 0.0
    if rewards.buffer_tokens > 0:
        overlong_penalty = length_penalty(
            llm.parameters["max_tokens"],
            llm_call.output_length_tokens,
            rewards.buffer_tokens,
        )
        reward += overlong_penalty

    trace.reward = reward
    trace.metadata.setdefault("symbolic", {})
    trace.metadata["symbolic"].update(
        {
            "status": evaluation.status,
            "error": evaluation.error,
            "target": problem.get("target"),
            "task_type": problem.get("task_type"),
            "overlong_penalty": overlong_penalty,
        }
    )

    metrics = BaseMetrics(
        reward=reward,
        success=evaluation.status == "correct",
        no_error=evaluation.status != "unparsable",
        no_answer=evaluation.status == "no_answer",
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
    )


__all__ = [
    "SymbolicEvaluation",
    "evaluate_symbolic_prediction",
    "expressions_equivalent",
    "extract_boxed_expression",
    "generate_symbolic_rollout",
]
