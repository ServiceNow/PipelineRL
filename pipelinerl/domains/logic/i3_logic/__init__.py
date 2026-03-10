"""Vendored i3_logic package from Prime Intellect.

Source: https://github.com/PrimeIntellect/i3-logic
License: Apache 2.0

This package provides verifiers for 87+ logic task types used in the
INTELLECT-3-RL dataset. Vendored to remove the external dependency.
"""

from .task2verifier import verifier_classes
from .base import Data

__all__ = ["verifier_classes", "Data"]
