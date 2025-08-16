import logging
import math
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
    
    def _extract_numerical_value(self, text: str):
        """Extract numerical value from text using multiple parsing strategies."""
        if not text or not isinstance(text, str):
            return None
            
        text = text.strip()
        if not text:
            return None
        
        # option 1: Simple integer
        if re.match(r'^[+-]?\d+$', text):
            try:
                return int(text)
            except ValueError:
                pass
        
        # option 2: Simple float
        if re.match(r'^[+-]?\d+\.\d+$', text):
            try:
                return float(text)
            except ValueError:
                pass
        
        # option 3: Scientific notation
        if re.match(r'^[+-]?\d+(?:\.\d+)?[eE][+-]?\d+$', text):
            try:
                return float(text)
            except ValueError:
                pass
        
        # option 4: Simple fraction
        if '/' in text and len(text.split('/')) == 2:
            try:
                parts = text.split('/')
                num = float(parts[0].strip())
                den = float(parts[1].strip())
                if den != 0:
                    value = num / den
                    if abs(value - round(value)) < 0.001:
                        return round(value)
                    return value
            except ValueError:
                pass
        
        # option 5: Try to evaluate simple arithmetic expressions
        try:
            import ast
            import operator
            
            # Simple arithmetic operators
            ops = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }
            
            def safe_eval(node):
                if isinstance(node, ast.Constant):  # Python 3.8+
                    return node.value
                elif isinstance(node, ast.Num):  # Python < 3.8
                    return node.n
                elif isinstance(node, ast.BinOp):
                    return ops[type(node.op)](safe_eval(node.left), safe_eval(node.right))
                elif isinstance(node, ast.UnaryOp):
                    return ops[type(node.op)](safe_eval(node.operand))
                else:
                    raise ValueError(f"Unsupported operation: {type(node)}")
            
            if re.match(r'^[0-9+\-*/().\s]+$', text):
                tree = ast.parse(text, mode='eval')
                result = safe_eval(tree.body)
                if isinstance(result, (int, float)) and not (math.isnan(result) or math.isinf(result)):
                    if abs(result - round(result)) < 0.001:
                        return round(result)
                    return result
        except Exception:
            pass
        
        # option 6: Try SymPy parsing (if available)
        try:
            import sympy as sp
            
            expr = sp.sympify(text)
            
            if expr.is_number:
                result = float(expr.evalf())
                if not (math.isnan(result) or math.isinf(result)):
                    if abs(result - round(result)) < 0.001:
                        return round(result)
                    return result
            
            elif expr.free_symbols:
                substitutions = {}
                for symbol in expr.free_symbols:
                    var_name = str(symbol)
                    if var_name in ['x', 'y', 'z']:
                        substitutions[symbol] = 1
                    elif var_name in ['t', 'time']:
                        substitutions[symbol] = 1
                    elif var_name in ['n', 'i', 'j', 'k']:
                        substitutions[symbol] = 1
                
                if substitutions:
                    try:
                        substituted = expr.subs(substitutions)
                        if substituted.is_number:
                            result = float(substituted.evalf())
                            if not (math.isnan(result) or math.isinf(result)):
                                if abs(result - round(result)) < 0.001:
                                    return round(result)
                                return result
                    except Exception as e:
                        logger.warning(f"Error evaluating SymPy expression: {e}")
                        pass
        except Exception as e:
            logger.warning(f"Error evaluating SymPy expression: {e}")
            pass
        
        return None
    
    def make_prompt(self, agent: Any, tape: Tape) -> Prompt:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        task = tape.steps[0]
        assert isinstance(task, Task), f"Expected a Task, got {task.__class__.__name__}"
        
        messages.append(
            {"role": "user", "content": task.llm_view()}
        )
        
        reasoning_steps = []
        assistant_output_content = ""
        
        for step in tape.steps[1:]:
            if isinstance(step, (PythonCodeAction, CodeExecutionResult, ActionExecutionFailure)):
                reasoning_steps.append(step)
        
        for step in reasoning_steps:
            if isinstance(step, PythonCodeAction):
                assistant_output_content += f"\n\n```python\n{step.code}\n```"
            elif isinstance(step, CodeExecutionResult):
                result = step.result.output.strip()
                if "\n\nstdout:" in result:
                    result = result.split("\n\nstdout:")[0].strip()
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                if len(result) > 2000:
                    lines = result.split('\n')
                    if len(lines) > 20:
                        kept_lines = lines[:10] + [f"... [{len(lines)-20} lines omitted] ..."] + lines[-10:]
                        result = '\n'.join(kept_lines)
                    else:
                        result = result[:2000] + "... [output truncated]"
                assistant_output_content += f"\n```output\n{result}\n```"
            elif isinstance(step, ActionExecutionFailure):
                assistant_output_content += f"\n```output\nError: {step.error}\n```"
        
        if assistant_output_content:
            messages.append({"role": "assistant", "content": assistant_output_content})
        
        llm = agent.llms.get("default")
        if llm and llm.tokenizer is None:
            llm.load_tokenizer()
        
        if llm and llm.tokenizer:
            if messages[-1]["role"] == "user":
                prompt_token_ids = llm.tokenizer.apply_chat_template(
                    messages, add_special_tokens=True, add_generation_prompt=True
                )
            else:
                prompt_token_ids = llm.tokenizer.apply_chat_template(
                    messages, add_special_tokens=True, add_generation_prompt=False
                )
        else:
            prompt_token_ids = None
        
        return Prompt(messages=messages, token_ids=prompt_token_ids)

    def generate_steps(self, agent: Any, tape: Tape, llm_stream) -> Generator[Step, None, None]:
        output_text = llm_stream.get_output().content
        if not output_text:
            yield LLMOutputParsingFailureAction(error="Empty LLM output", llm_output=output_text)
            yield SetNextNode(next_node="code_exec")
            return
        
        # extract Python code and boxed answer
        python_code_pattern = r'```python\s*\n(.*?)```'
        code_matches = re.findall(python_code_pattern, output_text, re.DOTALL)
        
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        boxed_match = re.search(boxed_pattern, output_text)
        
        has_execution_results = any(isinstance(step, CodeExecutionResult) for step in tape.steps)
        last_action_was_verification = False
        if tape.steps:
            for step in reversed(tape.steps):
                if isinstance(step, PythonCodeAction):
                    last_action_was_verification = step.name == "verification.py"
                    break
        
        # CASE 1: both code and boxed answer present?
        if code_matches and boxed_match and not last_action_was_verification:
            code = code_matches[-1].strip()
            boxed_value = boxed_match.group(1).strip()
            logger.info(f"Found complete solution with code and boxed answer: {boxed_value}")
            
            yield PythonCodeAction(name="verification.py", code=code, input_files=[])
            yield SetNextNode(next_node="code_exec")
            return
        
        # CASE 1b: executed verification code - extract result or use boxed answer?
        elif last_action_was_verification and has_execution_results:
            last_result = None
            for step in reversed(tape.steps):
                if isinstance(step, CodeExecutionResult):
                    result = step.result.output.strip()
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                    last_result = result
                    break
            
            if last_result:
                logger.info(f"Last execution result for answer extraction: '{last_result}'")
                lines = last_result.strip().split('\n')
                for i, line in enumerate(reversed(lines)):
                    line = line.strip()
                    logger.info(f"Checking line {i}: '{line}'")
                    
                    extracted_value = self._extract_numerical_value(line)
                    if extracted_value is not None:
                        logger.info(f"Using execution result as answer: {extracted_value}")
                        yield AnswerAction(text=f"The answer is {extracted_value}", value=extracted_value)
                        return
            
            # fallback: find boxed answer from original complete solution in tape history
            original_boxed_answer = None
            if hasattr(agent, 'llm_calls') and agent.llm_calls:
                for llm_call in reversed(agent.llm_calls):
                    if hasattr(llm_call, 'response') and llm_call.response:
                        content = llm_call.response.content
                        if '```python' in content and '\\boxed{' in content:
                            boxed_match_history = re.search(r'\\boxed\{([^}]+)\}', content)
                            if boxed_match_history:
                                original_boxed_answer = boxed_match_history.group(1).strip()
                                break
            
            if original_boxed_answer:
                logger.info(f"Falling back to original boxed answer: '{original_boxed_answer}'")
                try:
                    if '/' in original_boxed_answer and len(original_boxed_answer.split('/')) == 2:
                        parts = original_boxed_answer.split('/')
                        value = float(parts[0]) / float(parts[1])
                    else:
                        value = float(original_boxed_answer)
                except ValueError:
                    value = original_boxed_answer
                yield AnswerAction(text=f"The answer is {value}", value=value)
                return
            
            # something went wrong?
            logger.warning("Failed to extract answer from verification step")
            yield AnswerAction(text="Unable to determine answer", value=0)
            return
        
        # CASE 2: only code present?
        elif code_matches:
            code = code_matches[-1].strip()
            logger.info(f"Extracted Python code for iteration: {code[:100]}...")
            
            # why are we still generating code?
            reasoning_attempts = len([s for s in tape.steps if isinstance(s, PythonCodeAction)])
            
            recent_results = []
            recent_errors = []
            for step in tape.steps[-8:]:  # last 8 steps
                if isinstance(step, CodeExecutionResult):
                    result = step.result.output.strip()
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                    recent_results.append(result.lower())
                elif isinstance(step, ActionExecutionFailure):
                    recent_errors.append(step.error)
            
            none_outputs = sum(1 for r in recent_results if r in ['none', '', 'null'])
            same_outputs = len(recent_results) - len(set(recent_results)) if recent_results else 0
            
            if (none_outputs >= 2 and reasoning_attempts >= 3) or \
               (same_outputs >= 2 and reasoning_attempts >= 4) or \
               reasoning_attempts >= 6:
                logger.warning(f"Stopping code execution: {reasoning_attempts} attempts, {none_outputs} None outputs, {same_outputs} repeated outputs")
                
                # look at previous outputs for an answer
                all_outputs = []
                for step in tape.steps:
                    if isinstance(step, CodeExecutionResult):
                        output = step.result.output.strip()
                        if output.startswith('"') and output.endswith('"'):
                            output = output[1:-1]
                        all_outputs.append(output)
                
                combined_output = "\n".join(all_outputs)
                number_patterns = [
                    r'answer[:\s=]+([+-]?\d+(?:\.\d+)?)',
                    r'result[:\s=]+([+-]?\d+(?:\.\d+)?)',
                    r'([+-]?\d+(?:\.\d+)?)\s*$',
                    r'([+-]?\d+(?:\.\d+)?)',
                ]
                
                for pattern in number_patterns:
                    numbers = re.findall(pattern, combined_output, re.IGNORECASE | re.MULTILINE)
                    if numbers:
                        extracted_value = self._extract_numerical_value(numbers[-1])
                        if extracted_value is not None:
                            logger.info(f"Extracted answer from history: {extracted_value}")
                            yield AnswerAction(text=f"The answer is {extracted_value}", value=extracted_value)
                            return
                
                all_numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', combined_output)
                if all_numbers:
                    for num_str in reversed(all_numbers):  # Try from last to first
                        extracted_value = self._extract_numerical_value(num_str)
                        if extracted_value is not None:
                            logger.info(f"Extracted fallback answer: {extracted_value}")
                            yield AnswerAction(text=f"Best guess answer: {extracted_value}", value=extracted_value)
                            return
                
                logger.warning("No numerical answer found in outputs")
                yield AnswerAction(text="Unable to determine answer", value="No answer")
                return
            
            yield PythonCodeAction(name="math_solution.py", code=code, input_files=[])
            yield SetNextNode(next_node="code_exec")
        
        # CASE 3: only boxed answer present?
        elif boxed_match:
            value_str = boxed_match.group(1).strip()
            logger.info(f"Found direct boxed answer: {value_str}")
            
            extracted_value = self._extract_numerical_value(value_str)
            if extracted_value is not None:
                yield AnswerAction(text=f"The answer is {extracted_value}", value=extracted_value)
                return
            else:
                yield AnswerAction(text=f"The answer is {value_str}", value=value_str)
                return
        
        # CASE 4: neither code nor answer? - keep going
        else:
            reasoning_attempts = len([s for s in tape.steps if isinstance(s, PythonCodeAction)])
            parse_failures = len([s for s in tape.steps if isinstance(s, LLMOutputParsingFailureAction)])
            
            # check for "None" outputs or empty results that indicate unproductive loops
            recent_results = []
            for step in tape.steps[-6:]:  # last 6 steps
                if isinstance(step, CodeExecutionResult):
                    result = step.result.output.strip()
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                    recent_results.append(result.lower())
            
            none_outputs = sum(1 for r in recent_results if r in ['none', '', 'null'])
            
            should_terminate = (
                (parse_failures >= 2 and reasoning_attempts >= 4) or
                (none_outputs >= 2 and reasoning_attempts >= 3) or
                (reasoning_attempts >= 8)  # hard limit
            )
            
            if should_terminate:
                logger.warning(f"Terminating: {reasoning_attempts} attempts, {parse_failures} parse failures, {none_outputs} None outputs")
                # try to extract any numerical answer from the accumulated outputs
                all_outputs = []
                for step in tape.steps:
                    if isinstance(step, CodeExecutionResult):
                        output = step.result.output.strip()
                        if output.startswith('"') and output.endswith('"'):
                            output = output[1:-1]
                        all_outputs.append(output)
                
                combined_output = "\n".join(all_outputs)
                number_patterns = [
                    r'answer[:\s=]+([+-]?\d+(?:\.\d+)?)',
                    r'result[:\s=]+([+-]?\d+(?:\.\d+)?)',
                    r'([+-]?\d+(?:\.\d+)?)\s*$',  # Number at end
                    r'([+-]?\d+(?:\.\d+)?)',  # Any number
                ]
                
                for pattern in number_patterns:
                    numbers = re.findall(pattern, combined_output, re.IGNORECASE | re.MULTILINE)
                    if numbers:
                        # Try the improved extraction on the last found number
                        extracted_value = self._extract_numerical_value(numbers[-1])
                        if extracted_value is not None:
                            logger.info(f"Extracting answer from execution history with pattern '{pattern}': {extracted_value}")
                            yield AnswerAction(text=f"The answer is {extracted_value}", value=extracted_value)
                            return
                
                logger.warning("No clear numerical answer found, providing default")
                yield AnswerAction(text="Unable to determine answer", value=0)
                return
            
            yield LLMOutputParsingFailureAction(error="No code or answer found", llm_output=output_text)
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
    # compatibility wrapper - actual implementation is in datasets.py
    from .datasets import extract_result_value as datasets_extract_result_value
    return datasets_extract_result_value(sample)


def solve_task(agent: Agent, env, task: dict, tape_file: str = "") -> Tape:
    """Solve a single math task using the TIR agent."""
    from tapeagents.orchestrator import main_loop
    from tapeagents.io import save_json_tape
    import os
    
    tmp_tape_file = f"{tape_file}.tmp" if tape_file else None
    start_step = Task(task=task["task"])
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
        try:
            from pipelinerl.domains.math.verifier_api import verify_math
            predicted_answer = f"\\boxed{{{tape[-1].value}}}"
            target_answer = task.get("answer", "")
            answer_status = verify_math(predicted_answer, target_answer, strict=True)
            metadata["solved"] = (answer_status == "correct")
        except Exception as e:
            logger.warning(f"Math verification failed: {e}")
            task_value = task.get("value")
            tape_value = tape[-1].value
            if task_value is not None and tape_value is not None:
                metadata["solved"] = abs(float(task_value) - float(tape_value)) < 1e-6
            else:
                metadata["solved"] = False
    
    tape.metadata.result = metadata
    return tape