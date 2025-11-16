"""Symbolic algebra/calculus domain with CAS-checked rewards."""

from .dataset import load_problems

DOMAIN = "symbolic"


def generate_symbolic_rollout(*args, **kwargs):
    from .rollouts import generate_symbolic_rollout as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "DOMAIN",
    "generate_symbolic_rollout",
    "load_problems",
]
