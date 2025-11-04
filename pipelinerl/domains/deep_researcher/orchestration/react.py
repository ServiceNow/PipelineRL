"""ReAct-style orchestration: Reasoning + Acting in loop."""
import re
import logging
import asyncio
from typing import Dict
import aiohttp

from .base import BaseOrchestrator
from .registry import register_strategy
from pipelinerl.async_llm import llm_async_generate
from tapeagents.core import Prompt

logger = logging.getLogger(__name__)

@register_strategy("react")
class ReActOrchestrator(BaseOrchestrator):
    
    async def execute(
        self,
        question: str,
        session: aiohttp.ClientSession
    ) -> Dict:
        llm_calls = []
        tool_calls = 0
        
        messages = [
            {"role": "system", "content": self._get_react_system_prompt()},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        final_answer = None
        
        for turn in range(self.max_turns):
            llm_call = await llm_async_generate(
                self.llm,
                Prompt(messages=messages),
                session
            )
            llm_calls.append(llm_call)

            has_tool_calls = hasattr(llm_call.output, 'tool_calls') and llm_call.output.tool_calls
            has_content = llm_call.output.content is not None and llm_call.output.content.strip()
            
            if has_tool_calls:
                logger.debug(f"Turn {turn + 1}: Model returned {len(llm_call.output.tool_calls)} tool calls")
                
                messages.append({
                    "role": "assistant",
                    "content": llm_call.output.content or "",
                    "tool_calls": llm_call.output.tool_calls
                })
                
                async def execute_single_tool(tool_call):
                    tool_name = tool_call.function.name
                    tool_args = tool_call.function.arguments
                    
                    logger.debug(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    try:
                        # Map tool names to our registry
                        if tool_name.lower() in ["search", "web_search"]:
                            result = await self.tool_registry.execute_tool(
                                "web_search",
                                {"query": tool_args.get("query", "")},
                                session
                            )
                        elif tool_name.lower() in ["read", "web_read"]:
                            result = await self.tool_registry.execute_tool(
                                "web_read",
                                {"url": tool_args.get("url", "")},
                                session
                            )
                        else:
                            result = f"Error: Unknown tool '{tool_name}'"
                        
                        return {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": result,
                            "success": True
                        }
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        return {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": f"Error: {str(e)}",
                            "success": False
                        }
                
                tool_responses = await asyncio.gather(
                    *[execute_single_tool(tc) for tc in llm_call.output.tool_calls],
                    return_exceptions=True
                )
                
                for response in tool_responses:
                    if isinstance(response, Exception):
                        logger.error(f"Tool execution raised exception: {response}")
                        # Add error response
                        messages.append({
                            "role": "tool",
                            "content": f"Error: {str(response)}"
                        })
                    else:
                        messages.append(response)
                        if response.get("success"):
                            tool_calls += 1
            
            elif has_content:
                # Model returned content (final answer or reasoning)
                output = llm_call.output.content

                messages.append({"role": "assistant", "content": output})
                
                logger.debug(f"Turn {turn + 1}: {output[:200]}...")

                final_answer = output.strip()
                logger.info(f"Final answer received: {final_answer[:100]}...")
                break
            
            else:
                logger.error(f"Turn {turn + 1}: LLM returned neither tool_calls nor content")
                messages.append({
                    "role": "user",
                    "content": "Please provide either a tool call or your final answer."
                })
        
        if final_answer is None:
            logger.warning(f"Max turns ({self.max_turns}) reached without final answer")
            if llm_calls:
                last_output = llm_calls[-1].output.content or ""
                if last_output.strip():
                    final_answer = last_output.strip()
                    logger.info(f"Using last output as final answer: {final_answer[:100]}...")
                else:
                    final_answer = ""
            else:
                final_answer = ""
        
        return {
            "llm_calls": llm_calls,
            "final_answer": final_answer,
            "metadata": {
                "num_turns": len(llm_calls),
                "num_tool_calls": tool_calls,
                "strategy_used": "react"
            }
        }
    
    def _get_react_system_prompt(self) -> str:
        return """You are a helpful research assistant. Your goal is to answer questions accurately using available tools.

You have access to the following tools:
- search: Search the web for information. Use when you need to find general information.
  Arguments: {"query": "your search query"}
- read: Read and extract content from a webpage. Use when you have a specific URL.
  Arguments: {"url": "https://example.com"}

How to respond:
1. If you need more information, call one or more tools
2. If you have enough information to answer, return your final answer directly (no tool calls)

Behavior:
- Use tools to gather information when needed
- After receiving tool results, either call more tools or provide your final answer
- Be concise and accurate in your final answer
- If you cannot find the information, say so clearly

Now, answer the user's question using tools as needed."""
    
    def get_strategy_name(self) -> str:
        return "react"
