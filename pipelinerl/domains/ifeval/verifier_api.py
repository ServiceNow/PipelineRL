"""IFEval instruction following verification.

Uses the instruction_following_eval package for verification.
Install with: pip install git+https://github.com/josejg/instruction_following_eval.git
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid dependency issues
_IFEVAL_LOADED = False
_test_instruction_following = None
_InputExample = None


def _lazy_import_ifeval():
    """Lazily import instruction_following_eval to avoid import overhead."""
    global _IFEVAL_LOADED, _test_instruction_following, _InputExample

    if _IFEVAL_LOADED:
        return True

    try:
        from instruction_following_eval.evaluation import test_instruction_following
        from instruction_following_eval.evaluation import InputExample

        _test_instruction_following = test_instruction_following
        _InputExample = InputExample
        _IFEVAL_LOADED = True
        logger.info("instruction_following_eval loaded successfully")
        return True
    except ImportError as e:
        logger.error(
            "Failed to import instruction_following_eval: %s. "
            "Install with: pip install git+https://github.com/josejg/instruction_following_eval.git",
            e,
        )
        return False


@dataclass
class IFEvalVerificationResult:
    """Result of IFEval instruction verification."""

    followed_all: bool
    followed_count: int
    total_count: int
    instruction_results: list[bool]
    error: str | None = None

    @property
    def score(self) -> float:
        """Return fraction of instructions followed."""
        if self.total_count == 0:
            return 0.0
        return self.followed_count / self.total_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "followed_all": self.followed_all,
            "followed_count": self.followed_count,
            "total_count": self.total_count,
            "instruction_results": self.instruction_results,
            "score": self.score,
            "error": self.error,
        }


def verify_ifeval_response(
    response: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict | None],
    strict: bool = False,
) -> IFEvalVerificationResult:
    """Verify if a response follows the given instructions.

    Args:
        response: The model's response to verify.
        instruction_id_list: List of instruction IDs to check.
        kwargs_list: List of kwargs for each instruction (or None).
        strict: If True, use strict matching. If False, try response variants.

    Returns:
        IFEvalVerificationResult with verification details.
    """
    if not instruction_id_list:
        return IFEvalVerificationResult(
            followed_all=True,
            followed_count=0,
            total_count=0,
            instruction_results=[],
        )

    if not _lazy_import_ifeval():
        return IFEvalVerificationResult(
            followed_all=False,
            followed_count=0,
            total_count=len(instruction_id_list),
            instruction_results=[False] * len(instruction_id_list),
            error="instruction_following_eval not installed",
        )

    if not response or not response.strip():
        return IFEvalVerificationResult(
            followed_all=False,
            followed_count=0,
            total_count=len(instruction_id_list),
            instruction_results=[False] * len(instruction_id_list),
            error="empty_response",
        )

    # Ensure kwargs_list matches instruction_id_list length
    if len(kwargs_list) < len(instruction_id_list):
        kwargs_list = list(kwargs_list) + [{}] * (len(instruction_id_list) - len(kwargs_list))

    # Normalize kwargs - replace None with empty dict
    kwargs_list = [kw if kw is not None else {} for kw in kwargs_list]

    try:
        # Create InputExample for the verifier
        example = _InputExample(
            key=0,
            instruction_id_list=instruction_id_list,
            prompt="",  # Not needed for verification
            kwargs=kwargs_list,
        )

        # Run verification
        result = _test_instruction_following(example, response, strict=strict)

        # Extract individual instruction results
        instruction_results = []
        for instr_id in instruction_id_list:
            # The result has follow_instruction_list which is a list of bools
            # corresponding to each instruction
            idx = instruction_id_list.index(instr_id)
            if idx < len(result.follow_instruction_list):
                instruction_results.append(result.follow_instruction_list[idx])
            else:
                instruction_results.append(False)

        followed_count = sum(instruction_results)

        return IFEvalVerificationResult(
            followed_all=result.follow_all_instructions,
            followed_count=followed_count,
            total_count=len(instruction_id_list),
            instruction_results=instruction_results,
        )

    except Exception as e:
        logger.warning("IFEval verification error: %s", e)
        return IFEvalVerificationResult(
            followed_all=False,
            followed_count=0,
            total_count=len(instruction_id_list),
            instruction_results=[False] * len(instruction_id_list),
            error=str(e),
        )


def verify_ifeval_from_context(
    response: str,
    reward_context: dict[str, Any],
    strict: bool = False,
) -> IFEvalVerificationResult:
    """Verify response using reward_context from dataset.

    Args:
        response: The model's response to verify.
        reward_context: Dict with instruction_id_list and kwargs.
        strict: If True, use strict matching.

    Returns:
        IFEvalVerificationResult with verification details.
    """
    instruction_id_list = reward_context.get("instruction_id_list", [])
    kwargs_list = reward_context.get("kwargs", [])

    return verify_ifeval_response(
        response=response,
        instruction_id_list=instruction_id_list,
        kwargs_list=kwargs_list,
        strict=strict,
    )
