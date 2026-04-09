"""This package contains a modified subset of code from Prime Intellect's i3-logic project.

Source: https://github.com/PrimeIntellect/i3-logic
License: Apache 2.0

Modified for PipelineRL by retaining verifier-related files only.
"""

from .task2verifier import verifier_classes
from .base import Data

__all__ = ["verifier_classes", "Data"]
