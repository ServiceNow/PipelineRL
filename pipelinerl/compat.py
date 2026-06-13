from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_litellm_context_window_exception_for_pickle() -> None:
    """
    Make LiteLLM ContextWindowExceededError tolerant to pickle-based reconstruction.

    Ray reconstructs exceptions using positional args from ``BaseException.args``.
    LiteLLM's ContextWindowExceededError requires ``model`` and ``llm_provider``,
    so deserialization can fail when only a message is present.
    """
    try:
        from litellm import exceptions as litellm_exceptions
    except Exception:
        return

    cls = getattr(litellm_exceptions, "ContextWindowExceededError", None)
    if cls is None or getattr(cls, "_pipelinerl_pickle_patch_applied", False):
        return

    original_init = cls.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        if "message" in kwargs:
            message = kwargs.pop("message")
        elif args:
            message = args[0]
            args = args[1:]
        else:
            message = "Context window exceeded"

        if "model" in kwargs:
            model = kwargs.pop("model")
        elif args:
            model = args[0]
            args = args[1:]
        else:
            model = None

        if "llm_provider" in kwargs:
            llm_provider = kwargs.pop("llm_provider")
        elif args:
            llm_provider = args[0]
            args = args[1:]
        else:
            llm_provider = None

        response = kwargs.pop("response", args[0] if args else None)
        litellm_debug_info = kwargs.pop("litellm_debug_info", args[1] if len(args) > 1 else None)

        original_init(
            self,
            message=message,
            model=model,
            llm_provider=llm_provider,
            response=response,
            litellm_debug_info=litellm_debug_info,
            **kwargs,
        )

    cls.__init__ = _patched_init
    cls._pipelinerl_pickle_patch_applied = True
    logger.info("Applied ContextWindowExceededError pickle compatibility patch")

