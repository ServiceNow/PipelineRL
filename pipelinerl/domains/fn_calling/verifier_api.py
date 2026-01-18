"""BFCL v3 AST-based verifier for function calling domain."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Literal, Optional, cast

import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

AnswerStatus = Literal["correct", "wrong", "no_answer", "unparsable"]
_VALID_STATUSES: set[str] = {"correct", "wrong", "no_answer", "unparsable"}

# Pattern to extract tool_calls from text (fallback for non-structured output)
_TOOL_BLOCK = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL | re.IGNORECASE)


def _lazy_import_bfcl():
    """Lazily import BFCL modules."""
    try:
        from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
        from bfcl_eval.constants.enums import Language
        from bfcl_eval.utils import is_function_calling_format_output, is_java, is_js
        return {
            "ast_checker": ast_checker,
            "Language": Language,
            "is_function_calling_format_output": is_function_calling_format_output,
            "is_java": is_java,
            "is_js": is_js,
        }
    except ImportError as e:
        logger.error(f"Failed to import bfcl_eval: {e}")
        raise ImportError(
            "bfcl_eval package required for fn_calling verification. "
            "Install with: pip install bfcl-eval"
        ) from e


def _convert_to_gorilla(oai_tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool_calls format to BFCL Gorilla format.

    {"function": {"name": "add", "arguments": "{\"a\": 1, \"b\": 2}"}}
    -> {"add": {"a": 1, "b": 2}}
    """
    decoded_output = []
    for tool_call in oai_tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                params = json.loads(arguments)
            except json.JSONDecodeError:
                params = {}
        else:
            params = arguments if isinstance(arguments, dict) else {}
        decoded_output.append({name: params})
    return decoded_output


def _parse_tool_calls_from_text(generation: str) -> List[Dict[str, Any]]:
    """Parse tool_calls from text format (fallback for non-structured outputs)."""
    match = _TOOL_BLOCK.search(generation)
    if not match:
        return []

    payload = match.group(1).strip()
    if not payload:
        return []

    try:
        data = json.loads(payload)
        if not isinstance(data, list):
            data = [data]
        # Convert to OpenAI format
        oai_calls = []
        for item in data:
            if isinstance(item, dict):
                # Check if it's already in OAI format
                if "function" in item:
                    oai_calls.append(item)
                else:
                    # Gorilla format: {name: args}
                    for name, args in item.items():
                        oai_calls.append({
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                            }
                        })
        return oai_calls
    except json.JSONDecodeError:
        return []


def _extract_tool_calls(
    generation: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Extract tool calls from structured output or text."""
    # Prefer structured tool_calls if provided
    if tool_calls:
        return tool_calls
    # Fallback to text parsing
    return _parse_tool_calls_from_text(generation)


def verify_fn_calling_answer(
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "model",
) -> AnswerStatus:
    """Verify a function calling answer using BFCL AST checker.

    Args:
        generation: The model's text generation.
        reward_context: Contains 'category', 'function', 'ground_truth', 'is_relevance'.
        tool_calls: Optional structured tool_calls from the model output.
        model_name: Model name for BFCL's function name conversion.

    Returns:
        AnswerStatus: "correct", "wrong", "no_answer", or "unparsable".
    """
    category = reward_context.get("category", "simple")
    is_relevance = reward_context.get("is_relevance", False)
    function_def = reward_context.get("function", [])
    ground_truth = reward_context.get("ground_truth")

    # Extract tool calls
    oai_tool_calls = _extract_tool_calls(generation, tool_calls)

    # Handle relevance/irrelevance tests (no AST check needed)
    if is_relevance:
        has_tool_calls = len(oai_tool_calls) > 0
        if "irrelevance" in category:
            # Should NOT make any tool calls
            return "correct" if not has_tool_calls else "wrong"
        else:
            # Should make at least one tool call
            return "correct" if has_tool_calls else "no_answer"

    # No tool calls when expected
    if not oai_tool_calls:
        return "no_answer"

    # Convert to Gorilla format for BFCL checker
    try:
        gorilla_tool_calls = _convert_to_gorilla(oai_tool_calls)
    except Exception as e:
        logger.debug(f"Failed to convert tool calls to Gorilla format: {e}")
        return "unparsable"

    # Validate format
    try:
        bfcl = _lazy_import_bfcl()
        is_valid_format = bfcl["is_function_calling_format_output"](gorilla_tool_calls)
        if not is_valid_format:
            logger.debug("Invalid tool call format")
            return "unparsable"
    except Exception as e:
        logger.debug(f"Format validation failed: {e}")
        return "unparsable"

    # No ground truth to compare against
    if ground_truth is None:
        logger.warning(f"No ground truth for category '{category}'")
        return "unparsable"

    # Determine language for checker
    if bfcl["is_java"](category):
        language = bfcl["Language"].JAVA
    elif bfcl["is_js"](category):
        language = bfcl["Language"].JAVASCRIPT
    else:
        language = bfcl["Language"].PYTHON

    # Run AST checker
    try:
        checker_result = bfcl["ast_checker"](
            function_def,
            gorilla_tool_calls,
            ground_truth,
            language,
            category,
            model_name,
        )
    except Exception as e:
        logger.debug(f"AST checker failed: {e}")
        return "unparsable"

    if checker_result.get("valid", False):
        return "correct"
    else:
        logger.debug(
            f"AST check failed: {checker_result.get('error', 'unknown')} "
            f"({checker_result.get('error_type', 'unknown')})"
        )
        return "wrong"


# ---------------------------------------------------------------------------
# RPC / FastAPI server for remote verification
# ---------------------------------------------------------------------------


class FnCallingVerificationRequest(BaseModel):
    """Request payload for function calling verification."""

    generation: str
    reward_context: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    model_name: str = "model"


def _execute_verification(
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]],
    model_name: str,
) -> AnswerStatus:
    """Execute verification in process pool."""
    return verify_fn_calling_answer(
        generation=generation,
        reward_context=reward_context,
        tool_calls=tool_calls,
        model_name=model_name,
    )


async def verify_fn_calling_answer_rpc(
    *,
    session: aiohttp.ClientSession,
    host: str,
    port: int,
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "model",
) -> AnswerStatus:
    """Verify function calling answer via RPC call.

    Args:
        session: aiohttp session.
        host: Server hostname.
        port: Server port.
        generation: Model's text generation.
        reward_context: Context with function defs and ground truth.
        tool_calls: Structured tool calls from model output.
        model_name: Model name for verification.

    Returns:
        AnswerStatus from the verification server.
    """
    payload = {
        "generation": generation,
        "reward_context": reward_context,
        "tool_calls": tool_calls,
        "model_name": model_name,
    }
    async with session.post(f"http://{host}:{port}/verify_answer", json=payload) as response:
        body = await response.text()
        if response.status != 200:
            logger.error(f"fn_calling verifier returned {response.status}: {body[:512]}")
            raise ValueError("fn_calling verifier request failed")
        data = json.loads(body)
        status = str(data.get("answer_status", "")).strip().lower()
        if status not in _VALID_STATUSES:
            raise ValueError(f"fn_calling verifier produced invalid status '{status}'")
        return cast(AnswerStatus, status)


class BFCLEnvironment:
    """FastAPI wrapper for BFCL v3 function calling verifier."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        keepalive_timeout_s: int = 60,
    ) -> None:
        self._max_workers = max_workers
        self._keepalive_timeout_s = keepalive_timeout_s

    def launch(self, port: int) -> None:
        """Launch the verification server."""
        app = FastAPI()

        with ProcessPoolExecutor(max_workers=self._max_workers) as process_pool:

            @app.post("/verify_answer")
            async def verify(request: FnCallingVerificationRequest):
                loop = asyncio.get_running_loop()
                try:
                    answer_status = await loop.run_in_executor(
                        process_pool,
                        _execute_verification,
                        request.generation,
                        dict(request.reward_context),
                        request.tool_calls,
                        request.model_name,
                    )
                except Exception as exc:
                    logger.exception("fn_calling verification failed")
                    raise HTTPException(status_code=500, detail=str(exc))
                return JSONResponse(content={"answer_status": answer_status})

            @app.get("/health")
            async def health():
                return {"status": "ok"}

            uvicorn.run(
                app,
                host="0.0.0.0",
                port=port,
                timeout_keep_alive=self._keepalive_timeout_s,
            )


# Alias for backwards compatibility with existing config
AgenticToolsEnvironment = BFCLEnvironment
