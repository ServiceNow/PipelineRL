"""
Tool parser plugin for RL tool calling format.
"""

import json
import logging
import re

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParser
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ExtractedToolCallInformation,
    ToolCall,
    FunctionCall,
)

_JSON_SCALAR_TYPES = (dict, list, str, int, float, bool)


def _build_tool_call(index: int, parsed: dict, *, force_id: str | None = None) -> ToolCall | None:
    try:
        args_obj = parsed.get("arguments", {})
        if not isinstance(args_obj, _JSON_SCALAR_TYPES):
            args_obj = {}
        call_id = force_id if force_id is not None else parsed.get("id", f"call_{index}")
        return ToolCall(
            id=call_id,
            type="function",
            function=FunctionCall(
                name=str(parsed.get("name", "")),
                arguments=json.dumps(args_obj, ensure_ascii=False),
            ),
        )
    except Exception:
        logging.getLogger("pipelinerl.tool_parser").debug(
            "Skipping malformed tool call", exc_info=True
        )
        return None


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

        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"

        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )

        self.apriel_final_response_regex = re.compile(
            r"\[BEGIN FINAL RESPONSE\](.*?)\[END FINAL RESPONSE\]", re.DOTALL
        )
        # Lenient match: case-insensitive and tolerate a missing closing tag.
        self.apriel_tool_calls_regex = re.compile(
            r"<tool_calls>\s*(.*?)\s*(?:</tool_calls>|$)", re.DOTALL | re.IGNORECASE
        )

        # vLLM streaming hooks expect these attributes on the parser instance.
        self.current_tool_name_sent = False
        self.prev_tool_call_arr = []
        self.current_tool_id = -1
        self.streamed_args_for_tool = []

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        logger = logging.getLogger("pipelinerl.tool_parser")
        final_response_match = None

        try:
            tool_calls_matches = list(self.apriel_tool_calls_regex.finditer(model_output))
            if tool_calls_matches:
                last_match = tool_calls_matches[-1]
                tool_calls_json = last_match.group(1).strip()
                parsed_calls = []
                try:
                    parsed_calls = json.loads(tool_calls_json) if tool_calls_json else []
                except Exception:
                    logger.debug("Failed to parse aggregated <tool_calls> JSON; falling back", exc_info=True)
                    parsed_calls = []

                tool_calls = [
                    tc for tc in (_build_tool_call(i, pc) for i, pc in enumerate(parsed_calls))
                    if tc is not None
                ]

                final_response_match = self.apriel_final_response_regex.search(model_output)
                content = final_response_match.group(1).strip() if final_response_match else ""

                return ExtractedToolCallInformation(
                    tools_called=bool(tool_calls),
                    tool_calls=tool_calls,
                    content=content,
                )

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
                    tool_calls = [
                        tc for tc in (_build_tool_call(i, pc) for i, pc in enumerate(parsed_list))
                        if tc is not None
                    ]
                    content = final_response_match.group(1).strip() if final_response_match else ""
                    return ExtractedToolCallInformation(
                        tools_called=bool(tool_calls),
                        tool_calls=tool_calls,
                        content=content,
                    )

            # Fallback: legacy <tool_call> blocks.
            content_to_search = model_output
            final_response_match = self.apriel_final_response_regex.search(model_output)
            if final_response_match:
                final_response_content = final_response_match.group(1).strip()
                if self.tool_call_start_token in final_response_content:
                    content_to_search = final_response_content
                elif self.tool_call_start_token not in model_output:
                    return ExtractedToolCallInformation(
                        tools_called=False,
                        tool_calls=[],
                        content=final_response_content
                    )

            if self.tool_call_start_token not in content_to_search:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )

            function_call_tuples = self.tool_call_regex.findall(content_to_search)

            tool_calls = []
            for i, match in enumerate(function_call_tuples):
                json_str = match[0] if match[0] else match[1]
                try:
                    parsed_call = json.loads(json_str.strip())
                except Exception:
                    logger.debug("Skipping malformed <tool_call> JSON", exc_info=True)
                    continue
                tc = _build_tool_call(i, parsed_call, force_id=f"call_{i}")
                if tc is not None:
                    tool_calls.append(tc)

            if tool_calls and final_response_match:
                content = ""
            elif final_response_match:
                content = final_response_match.group(1).strip()
            else:
                content = model_output

            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content
            )

        except Exception:
            # Never propagate to the vLLM server.
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
    