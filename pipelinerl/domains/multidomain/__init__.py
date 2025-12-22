"""Debug utilities for multi-domain experimentation."""

from collections.abc import Iterable
from typing import Any

DOMAIN = "multi"


_MATH_SAMPLE = {
    "id": 0,
    "dataset": "math_debug",
    "task": "Compute 2 + 2.",
    "answer": "\\boxed{4}",
    "domain": "math",
}

_GUESSING_SAMPLE = {
    "id": 0,
    "dataset": "guessing_debug",
    "task": "Hidden number between 1 and 3. Start guessing.",
    "answer": 2,
    "domain": "guessing",
}

_CODING_SAMPLE = {
    "id": 0,
    "dataset": "coding_debug",
    "question": "Implement class Solution with method addTwoNumbers(a: int, b: int) that returns a + b.",
    "starter_code": "class Solution:\n    def addTwoNumbers(self, a: int, b: int) -> int:\n        pass",
    "tests": [
        {"id": 0, "type": "functional", "input": "1\n2", "output": "3"},
        {"id": 1, "type": "functional", "input": "5\n7", "output": "12"},
    ],
    "entry_point": "addTwoNumbers",
    "domain": "coding",
}


def load_problems(dataset_names: Iterable[str] | None = None, **_: Any) -> list[dict]:
    """Return tiny synthetic problems for smoke-testing multi-domain dispatch."""
    if dataset_names is None:
        dataset_names = ["math_debug", "guessing_debug"]

    problems: list[dict] = []
    for name in dataset_names:
        if name == "math_debug":
            problems.append(dict(_MATH_SAMPLE))
        elif name == "guessing_debug":
            problems.append(dict(_GUESSING_SAMPLE))
        elif name == "coding_debug":
            problems.append(dict(_CODING_SAMPLE))
        else:
            raise ValueError(f"Unknown debug dataset '{name}'")
    return problems
