"""Compatibility helpers for dealing with transformers regressions."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def ensure_mistral3_auto_causal_lm_registered() -> None:
    """Register Mistral 3 config for AutoModelForCausalLM when missing."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from transformers.models.auto import modeling_auto
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logger.debug("transformers unavailable; skipping Mistral 3 auto registration: %s", exc)
        return

    if modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get("mistral3"):
        _PATCHED = True
        return

    modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["mistral3"] = "Mistral3ForConditionalGeneration"
    # Some utilities rely on MODEL_WITH_LM_HEAD, keep them in sync.
    modeling_auto.MODEL_WITH_LM_HEAD_MAPPING_NAMES["mistral3"] = "Mistral3ForConditionalGeneration"

    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

        # Touch the lazy mapping once so the entry is registered without materializing weights.
        _ = AutoModelForCausalLM._model_mapping[Mistral3Config]
    except Exception as exc:  # pragma: no cover - optional dependency guard
        logger.debug("Unable to prime Mistral 3 causal LM mapping: %s", exc)

    _PATCHED = True
