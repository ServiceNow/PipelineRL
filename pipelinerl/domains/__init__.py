"""Domain utilities and rollouts."""

from .dispatcher import generate_multidomain_rollout, register_domain_rollout

__all__ = [
    "generate_multidomain_rollout",
    "register_domain_rollout",
]
