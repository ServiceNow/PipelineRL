from typing import Any, Literal
from pydantic import Field
from tapeagents.core import StopStep


class MathAnswer(StopStep):
    """
    Action that indicates the agent has finished solving a math problem.
    The final answer must be contained within \\boxed{} format.
    """

    kind: Literal["math_answer_action"] = "math_answer_action"
    answer: Any = Field(description="Final answer in \\boxed{} format")