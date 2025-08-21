"""
Tool parser plugin for RL tool calling format.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union, Sequence

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
        # Quick check to avoid unnecessary processing
        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Find all tool call matches
            function_call_tuples = self.tool_call_regex.findall(model_output)
            
            # Parse JSON from matches
            tool_calls = []
            for i, match in enumerate(function_call_tuples):
                json_str = match[0] if match[0] else match[1]
                try:
                    parsed_call = json.loads(json_str.strip())
                    
                    tool_call = ToolCall(
                        id=f"call_{i}",
                        type="function",
                        function=FunctionCall(
                            name=parsed_call.get("name", ""),
                            arguments=json.dumps(
                                parsed_call.get("arguments", {}),
                                ensure_ascii=False
                            )
                        )
                    )
                    tool_calls.append(tool_call)
                except json.JSONDecodeError:
                    continue
            
            # Extract content before first tool call
            content = model_output#[:model_output.find(self.tool_call_end_token)].strip()
            if not content:
                content = None
                
            return ExtractedToolCallInformation(
                tools_called=bool(tool_calls),
                tool_calls=tool_calls,
                content=content
            )
            
        except Exception:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    
    def extract_tool_calls_streaming(
        self, 
        previous_text: str, 
        current_text: str, 
        delta_text: str, 
        request
    ) -> Optional[Dict[str, Any]]:
        """
        Extract tool calls in streaming mode.
        
        Args:
            previous_text: The previous text
            current_text: The current complete text
            delta_text: The new text delta
            request: The request object
            
        Returns:
            Dictionary with streaming tool call information
        """
        # Simple streaming implementation
        if self.tool_call_start_token not in current_text:
            return {"content": delta_text}
        
        # Check if we're starting a new tool call
        if self.tool_call_start_token in delta_text:
            self.current_tool_id += 1
            return {
                "tool_calls": [{
                    "index": self.current_tool_id,
                    "type": "function",
                    "id": f"call_{self.current_tool_id}",
                    "function": {"name": ""}
                }]
            }
        
        return {"content": delta_text}