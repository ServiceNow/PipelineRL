"""
Tool parser plugin for RL tool calling format.
"""

import json
import re
from typing import Any  # noqa: F401
import logging

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, 
    ExtractedToolCallInformation,
    ToolCall,
    FunctionCall
)


@ToolParserManager.register_module("rl_tool")
class HermesRLToolParser(ToolParser):
    """
    Tool parser for RL tool calling format using <tool_call></tool_call> markers.
    Supports both standard format and Apriel-style formats:
    - <tool_calls>[{...}, {...}]</tool_calls> (preferred if present)
    - [BEGIN FINAL RESPONSE] ... [END FINAL RESPONSE] wrapper
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # Tool call markers
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        
        # Regex pattern for parsing tool calls
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        
        # Apriel-specific patterns
        self.apriel_final_response_regex = re.compile(
            r"\[BEGIN FINAL RESPONSE\](.*?)\[END FINAL RESPONSE\]", re.DOTALL
        )
        # Prefer parsing aggregated tool calls from <tool_calls>...</tool_calls>
        # Be lenient: case-insensitive; tolerate missing closing tag by capturing to end.
        self.apriel_tool_calls_regex = re.compile(
            r"<tool_calls>\s*(.*?)\s*(?:</tool_calls>|$)", re.DOTALL | re.IGNORECASE
        )
        
        # State for streaming
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []
    
    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """
        Extract tool calls from the model output.
        
        Args:
            model_output: The raw model output string
            request: The request object
            
        Returns:
            ExtractedToolCallInformation with tool calls and metadata
        """
        logger = logging.getLogger("pipelinerl.tool_parser")
        # Ensure variable exists for any fallback references below
        final_response_match = None

        try:
            # 1) Apriel aggregated tool calls block has priority
            tool_calls_matches = list(self.apriel_tool_calls_regex.finditer(model_output))
            if tool_calls_matches:
                # Use the last match (in case of multiple blocks)
                last_match = tool_calls_matches[-1]
                tool_calls_json = last_match.group(1).strip()
                parsed_calls = []
                try:
                    parsed_calls = json.loads(tool_calls_json) if tool_calls_json else []
                except Exception:
                    logger.debug("Failed to parse aggregated <tool_calls> JSON; falling back", exc_info=True)
                    parsed_calls = []

                tool_calls: list[ToolCall] = []
                for i, pc in enumerate(parsed_calls):
                    try:
                        name = pc.get("name", "")
                        args_obj = pc.get("arguments", {})
                        if not isinstance(args_obj, (dict, list, str, int, float, bool)):
                            args_obj = {}
                        args_str = json.dumps(args_obj, ensure_ascii=False)
                        call_id = pc.get("id", f"call_{i}")
                        tool_calls.append(
                            ToolCall(
                                id=call_id,
                                type="function",
                                function=FunctionCall(name=str(name), arguments=args_str),
                            )
                        )
                    except Exception:
                        logger.debug("Skipping malformed aggregated tool call", exc_info=True)
                        continue

                # Prefer final response content if present; otherwise empty string
                final_response_match = self.apriel_final_response_regex.search(model_output)
                content = final_response_match.group(1).strip() if final_response_match else ""

                return ExtractedToolCallInformation(
                    tools_called=bool(tool_calls),
                    tool_calls=tool_calls,
                    content=content,
                )

            # 2) Try bare JSON tool-calls (no tags), but only if tools are declared in the request
            #    Accept either a list of {name, arguments} or a single dict
            try:
                tools_declared = bool(getattr(request, "tools", None))
            except Exception:
                tools_declared = False

            if tools_declared:
                candidate_strings: list[str] = []
                final_response_match = self.apriel_final_response_regex.search(model_output)
                if final_response_match:
                    candidate_strings.append(final_response_match.group(1).strip())
                candidate_strings.append(model_output.strip())

                for candidate in candidate_strings:
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        continue
                    parsed_list = []
                    if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
                        parsed_list = [parsed]
                    elif isinstance(parsed, list) and all(isinstance(it, dict) for it in parsed):
                        parsed_list = [it for it in parsed if "name" in it and "arguments" in it]
                    if not parsed_list:
                        continue
                    tool_calls: list[ToolCall] = []
                    for i, pc in enumerate(parsed_list):
                        try:
                            name = pc.get("name", "")
                            args_obj = pc.get("arguments", {})
                            if not isinstance(args_obj, (dict, list, str, int, float, bool)):
                                args_obj = {}
                            args_str = json.dumps(args_obj, ensure_ascii=False)
                            call_id = pc.get("id", f"call_{i}")
                            tool_calls.append(
                                ToolCall(
                                    id=call_id,
                                    type="function",
                                    function=FunctionCall(name=str(name), arguments=args_str),
                                )
                            )
                        except Exception:
                            logger.debug("Skipping malformed bare-JSON tool call", exc_info=True)
                            continue
                    content = final_response_match.group(1).strip() if final_response_match else ""
                    return ExtractedToolCallInformation(
                        tools_called=bool(tool_calls),
                        tool_calls=tool_calls,
                        content=content,
                    )

            # 3) Fallback: look for single <tool_call> blocks (legacy / other models)
            content_to_search = model_output
            final_response_match = self.apriel_final_response_regex.search(model_output)
            if final_response_match:
                final_response_content = final_response_match.group(1).strip()
                if self.tool_call_start_token in final_response_content:
                    content_to_search = final_response_content
                elif self.tool_call_start_token not in model_output:
                    # No tool calls found, return final response as content
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=final_response_content
                    )

            # Quick check to avoid unnecessary processing
            if self.tool_call_start_token not in content_to_search:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )

            # Find all tool call matches
            function_call_tuples = self.tool_call_regex.findall(content_to_search)

            # Parse JSON from matches
            tool_calls = []
            for i, match in enumerate(function_call_tuples):
                json_str = match[0] if match[0] else match[1]
                try:
                    parsed_call = json.loads(json_str.strip())
                    args_obj = parsed_call.get("arguments", {})
                    if not isinstance(args_obj, (dict, list, str, int, float, bool)):
                        args_obj = {}
                    tool_call = ToolCall(
                        id=f"call_{i}",
                        type="function",
                        function=FunctionCall(
                            name=str(parsed_call.get("name", "")),
                            arguments=json.dumps(args_obj, ensure_ascii=False)
                        )
                    )
                    tool_calls.append(tool_call)
                except Exception:
                    logger.debug("Skipping malformed <tool_call> JSON", exc_info=True)
                    continue

            # Determine content based on whether we found tool calls
            if tool_calls and final_response_match:
                # If we found tool calls in final response, use just the tool calls
                content = ""
            elif final_response_match:
                # If we have final response but no tool calls there, use final response
                content = final_response_match.group(1).strip()
            else:
                # Standard processing
                content = model_output

            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content
            )

        except Exception:
            # Never propagate exceptions to the server; log and return a safe fallback.
            logger.exception("Tool parser encountered an exception; returning safe fallback.")
            if final_response_match:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=final_response_match.group(1).strip()
                )
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    