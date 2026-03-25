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

# Categories where multiple function calls are expected
_PARALLEL_CATEGORIES = {"parallel", "parallel_multiple", "live_parallel", "live_parallel_multiple"}

# Pattern to extract tool_calls from text (fallback for non-structured output)
_TOOL_BLOCK = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL | re.IGNORECASE)


def _lazy_import_bfcl():
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
    """
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
    matches = _TOOL_BLOCK.findall(generation)
    if not matches:
        return []

    # Use the LAST match (actual model output, not template examples)
    payload = matches[-1].strip()
    if not payload:
        # If last match is empty, try earlier matches in reverse order
        for match in reversed(matches[:-1]):
            if match.strip():
                payload = match.strip()
                break
        if not payload:
            return []

    try:
        data = json.loads(payload)
        if not isinstance(data, list):
            data = [data]
        oai_calls = []
        for item in data:
            if isinstance(item, dict):
                # Check if it's already in OAI format
                if "function" in item:
                    oai_calls.append(item)
                # Handle {"name": "func_name", "arguments": {...}} format
                elif "name" in item and "arguments" in item:
                    oai_calls.append({
                        "function": {
                            "name": item["name"],
                            "arguments": json.dumps(item["arguments"]) if isinstance(item["arguments"], dict) else str(item["arguments"]),
                        }
                    })
                else:
                    # Gorilla format: {func_name: {args}}
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
    # Prefer structured tool_calls if provided
    if tool_calls:
        return tool_calls
    # Fallback to text parsing
    return _parse_tool_calls_from_text(generation)


def _compute_partial_score(
    func_descriptions: list,
    gorilla_tool_calls: list,
    ground_truth: list,
    language: Any,
    model_name_for_bfcl: str,
) -> float:
    """Compute partial credit for parallel categories by checking each expected call individually.

    Returns fraction of ground truth calls matched (0.0 to 1.0).
    """
    from bfcl_eval.eval_checker.ast_eval.ast_checker import simple_function_checker, find_description

    if not ground_truth:
        return 0.0

    total = len(ground_truth)
    matched_indices: list[int] = []
    passed = 0

    for expected in ground_truth:
        func_name_expected = list(expected.keys())[0]
        func_description = find_description(func_descriptions, func_name_expected)
        if func_description is None:
            continue

        for idx in range(len(gorilla_tool_calls)):
            if idx in matched_indices:
                continue
            try:
                result = simple_function_checker(
                    func_description,
                    gorilla_tool_calls[idx],
                    expected,
                    language,
                    model_name_for_bfcl,
                )
                if result["valid"]:
                    matched_indices.append(idx)
                    passed += 1
                    break
            except Exception:
                continue

    return passed / total


def verify_fn_calling_answer(
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Verify a function calling answer using BFCL AST checker.

    Returns:
        Dict with 'answer_status' (AnswerStatus) and 'partial_score' (float, 0.0-1.0).
    """
    category = reward_context.get("category", "simple")
    is_relevance = reward_context.get("is_relevance", False)
    function_def = reward_context.get("function", [])
    ground_truth = reward_context.get("ground_truth")

    oai_tool_calls = _extract_tool_calls(generation, tool_calls)
    bfcl = _lazy_import_bfcl()

    if is_relevance:
        has_tool_calls = len(oai_tool_calls) > 0
        if "irrelevance" in category:
            # Should NOT make any tool calls
            status = "correct" if not has_tool_calls else "wrong"
        else:
            # Should make at least one tool call
            status = "correct" if has_tool_calls else "no_answer"
        return {"answer_status": status, "partial_score": 1.0 if status == "correct" else 0.0}

    if not oai_tool_calls:
        return {"answer_status": "no_answer", "partial_score": 0.0}

    # Note: Java name normalization is handled by BFCL when using a -FC model
    # (BFCL converts ground truth dots→underscores to match model output)

    try:
        gorilla_tool_calls = _convert_to_gorilla(oai_tool_calls)
    except Exception as e:
        logger.debug(f"Failed to convert tool calls to Gorilla format: {e}")
        return {"answer_status": "unparsable", "partial_score": 0.0}

    try:
        is_valid_format = bfcl["is_function_calling_format_output"](gorilla_tool_calls)
        if not is_valid_format:
            logger.debug("Invalid tool call format")
            return {"answer_status": "unparsable", "partial_score": 0.0}
    except Exception as e:
        logger.debug(f"Format validation failed: {e}")
        return {"answer_status": "unparsable", "partial_score": 0.0}

    # Always use a known -FC model for BFCL's AST checker.
    # The -FC suffix means underscore_to_dot=True, which normalizes ground truth dots→underscores
    # to match model output (OpenAI API doesn't allow dots in function names).
    model_name_for_bfcl = "gpt-4o-2024-11-20-FC"

    if ground_truth is None:
        logger.warning(f"No ground truth for category '{category}'")
        return {"answer_status": "unparsable", "partial_score": 0.0}

    # BFCL returns {'id': ..., 'ground_truth': [...]} but AST checker expects just the list
    if isinstance(ground_truth, dict) and "ground_truth" in ground_truth:
        ground_truth = ground_truth["ground_truth"]

    if bfcl["is_java"](category):
        language = bfcl["Language"].JAVA
    elif bfcl["is_js"](category):
        language = bfcl["Language"].JAVASCRIPT
    else:
        language = bfcl["Language"].PYTHON

    try:
        checker_result = bfcl["ast_checker"](
            function_def,
            gorilla_tool_calls,
            ground_truth,
            language,
            category,
            model_name_for_bfcl,
        )
    except Exception as e:
        logger.debug(f"AST checker failed: {e}")
        return {"answer_status": "unparsable", "partial_score": 0.0}

    if checker_result.get("valid", False):
        return {"answer_status": "correct", "partial_score": 1.0}

    # For parallel categories, compute partial credit per-call
    partial_score = 0.0
    if category in _PARALLEL_CATEGORIES and isinstance(ground_truth, list) and len(ground_truth) > 1:
        try:
            partial_score = _compute_partial_score(
                function_def, gorilla_tool_calls, ground_truth, language, model_name_for_bfcl,
            )
        except Exception as e:
            logger.debug(f"Partial score computation failed: {e}")

    logger.debug(
        f"AST check failed: {checker_result.get('error', 'unknown')} "
        f"({checker_result.get('error_type', 'unknown')})"
    )
    return {"answer_status": "wrong", "partial_score": partial_score}


class FnCallingVerificationRequest(BaseModel):
    generation: str
    reward_context: Dict[str, Any] = Field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    model_name: str = "model"


def _execute_verification(
    generation: str,
    reward_context: Dict[str, Any],
    tool_calls: Optional[List[Dict[str, Any]]],
    model_name: str,
) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
    """Returns dict with 'answer_status' (AnswerStatus) and 'partial_score' (float)."""
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
        return {
            "answer_status": cast(AnswerStatus, status),
            "partial_score": float(data.get("partial_score", 1.0 if status == "correct" else 0.0)),
        }


class BFCLEnvironment:
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
                    result = await loop.run_in_executor(
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
                return JSONResponse(content=result)

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
