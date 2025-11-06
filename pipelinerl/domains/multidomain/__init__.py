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
        else:
            raise ValueError(f"Unknown debug dataset '{name}'")
    return problems
