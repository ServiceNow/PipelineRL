import logging
import re
from typing import Any, Generator, Union, Literal
from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import Prompt, Step, Tape, Observation, LLMOutputParsingFailureAction, SetNextNode, StopStep
from tapeagents.llms import LLM
from tapeagents.nodes import Node
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.code_executor import PythonCodeAction, CodeExecutionResult

logger = logging.getLogger(__name__)


class Task(Observation):
    kind: Literal["task"] = "task"
    task: str
    template: str = Field(default="{task}", description="Template for the task with {task} placeholder")

    def llm_view(self, indent: int | None = 2) -> str:
        return self.template.format(task=self.task)


class AnswerAction(StopStep):
    kind: Literal["answer_action"] = "answer_action"
    text: str
    value: Union[float, int, str]


class CodeExecutionNode(Node):
    """Node that generates Python code to solve math problems with iterative reasoning."""
    
    system_prompt: str = Field(default="", description="System prompt for the node")
    
    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Build conversation with task and previous code/results
        task = tape.steps[0]
        assert isinstance(task, Task), f"Expected a Task, got {task.__class__.__name__}"
        
        conversation_content = task.llm_view()
        
        # Add previous code execution attempts and results
        for step in tape.steps[1:]:
            if isinstance(step, PythonCodeAction):
                conversation_content += f"\n\n```python\n{step.code}\n```output\n"
            elif isinstance(step, CodeExecutionResult):
                result = step.result.output.strip()
                if "\n\nstdout:" in result:
                    result = result.split("\n\nstdout:")[0].strip()
                # Clean up result formatting
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                conversation_content += f"{result}\n```"
            elif isinstance(step, ActionExecutionFailure):
                conversation_content += f"Error: {step.error}\n```"
        
        messages.append({"role": "user", "content": conversation_content})
        
        # Load tokenizer if needed
        llm = agent.llms.get("default")
        if llm and llm.tokenizer is None:
            llm.load_tokenizer()
        
        if llm and llm.tokenizer:
            prompt_token_ids = llm.tokenizer.apply_chat_template(
                messages, add_special_tokens=True, add_generation_prompt=True
            )
        else:
            prompt_token_ids = None
        
        return Prompt(messages=messages, token_ids=prompt_token_ids)

    def generate_steps(self, agent: Any, tape: Tape, llm_stream) -> Generator[Step, None, None]:
        # Parse LLM output for Python code or final answer
        output_text = llm_stream.get_output().content
        if not output_text:
            yield LLMOutputParsingFailureAction(error="Empty LLM output", llm_output=output_text)
            yield SetNextNode(next_node="code_exec")
            return
        
        # Check for boxed answer first
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_match = re.search(boxed_pattern, output_text)
        if boxed_match:
            value_str = boxed_match.group(1).strip()
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
            logger.info(f"Found final boxed answer: {value}")
            yield AnswerAction(text=f"The answer is {value}", value=value)
            return
        
        # Look for Python code blocks
        python_code_pattern = r'```python\s*\n(.*?)\n```'
        code_matches = re.findall(python_code_pattern, output_text, re.DOTALL)
        
        if code_matches:
            # Take the last code block
            code = code_matches[-1].strip()
            logger.info(f"Extracted Python code: {code[:100]}...")
            yield PythonCodeAction(
                name="math_solution.py",
                code=code,
                input_files=[]
            )
            yield SetNextNode(next_node="code_exec")
        else:
            # No code or boxed answer - try to extract answer from text
            has_execution_results = any(isinstance(step, CodeExecutionResult) for step in tape.steps)
            
            if has_execution_results:
                # Look for answer patterns in the output
                answer_patterns = [
                    r"(?:answer|result)\s+is\s+([+-]?\d*\.?\d+)",
                    r"([+-]?\d*\.?\d+)$",
                    r"(?:final|answer):\s*([+-]?\d*\.?\d+)",
                ]
                
                for pattern in answer_patterns:
                    match = re.search(pattern, output_text, re.IGNORECASE)
                    if match:
                        try:
                            value = float(match.group(1))
                            logger.info(f"Extracted answer from text: {value}")
                            yield AnswerAction(text=f"The answer is {value}", value=value)
                            return
                        except ValueError:
                            continue
            
            # Continue iterating
            yield LLMOutputParsingFailureAction(error="No Python code or clear answer found, continuing", llm_output=output_text)
            yield SetNextNode(next_node="code_exec")


TIRMathTape = Tape[
    None,
    Union[
        Task,
        PythonCodeAction,
        CodeExecutionResult,
        ActionExecutionFailure,
        LLMOutputParsingFailureAction,
        SetNextNode,
        AnswerAction,
    ],
]


class TIRMathAgent(Agent):
    """TIR (Tool Integrated Reasoning) agent for mathematical problem solving."""
    
    def __init__(self, system_prompt: str = "", max_iterations: int = 8, **kwargs):
        # Create nodes with the system prompt
        nodes = [
            CodeExecutionNode(
                name="code_exec",
                system_prompt=system_prompt
            ),
        ]
        super().__init__(nodes=nodes, max_iterations=max_iterations, **kwargs)
        self.store_llm_calls = True
    
    @classmethod
    def create(cls, system_prompt: str, llm: LLM, max_prompt_length: int, max_iterations: int = 8):
        agent = cls(
            system_prompt=system_prompt,
            llms={"default": llm},
            max_iterations=max_iterations,
        )
        agent.store_llm_calls = True
        if agent.llms["default"].tokenizer is None:
            agent.llms["default"].load_tokenizer()
        return agent

    def get_steps_description(self) -> str:
        return "Generate Python code iteratively to solve math problems, execute it, analyze results, and provide final answer."


def extract_result_value(sample: dict) -> dict:
    """Extract numerical result from dataset sample."""
    # Compatibility wrapper - actual implementation is in datasets.py
    from .datasets import extract_result_value as datasets_extract_result_value
    return datasets_extract_result_value(sample)


def solve_task(agent: Agent, env, task: dict, tape_file: str = "") -> Tape:
    """Solve a single math task using the TIR agent."""
    from tapeagents.orchestrator import main_loop
    from tapeagents.io import save_json_tape
    import os
    
    tmp_tape_file = f"{tape_file}.tmp" if tape_file else None
    start_step = Task(task=task["question"])
    tape = TIRMathTape(steps=[start_step], context=None)
    metadata = task.copy()

    for event in main_loop(agent, tape, env, max_loops=30):
        step = None
        if event.agent_event and event.agent_event.step:
            step = event.agent_event.step
        elif event.observation:
            step = event.observation
        if step:
            tape = tape.append(step)
            if tmp_tape_file:
                save_json_tape(tape, tmp_tape_file)
    
    if tmp_tape_file:
        os.unlink(tmp_tape_file)
    
    metadata["solved"] = False
    if isinstance(tape[-1], AnswerAction):
        # Use same verification logic as generate_tir_rollout
        try:
            from pipelinerl.domains.math.verifier_api import verify_math
            predicted_answer = f"\\boxed{{{tape[-1].value}}}"
            target_answer = task.get("answer", "")
            answer_status = verify_math(predicted_answer, target_answer, strict=True)
            metadata["solved"] = (answer_status == "correct")
        except Exception as e:
            logger.warning(f"Math verification failed: {e}")
            # Fallback to numerical comparison
            task_value = task.get("value")
            tape_value = tape[-1].value
            if task_value is not None and tape_value is not None:
                metadata["solved"] = abs(float(task_value) - float(tape_value)) < 1e-6
            else:
                metadata["solved"] = False
    
    tape.metadata.result = metadata
    return tape