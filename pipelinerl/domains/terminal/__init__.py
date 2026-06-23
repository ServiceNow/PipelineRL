"""TMax-style terminal-agent domain: proot sandbox, bash agent, pytest verifier."""

from .load_tasks import load_problems
from .rollouts import generate_terminal_rollout

__all__ = [
    "generate_terminal_rollout",
    "load_problems",
]
