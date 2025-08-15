import asyncio
import logging
import os

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tapeagents.remote_environment import EnvironmentServer

from tapeagents.environment import AsyncEnvironment, Environment
from tapeagents.core import Action
from tapeagents.tools.code_executor import PythonCodeAction, CodeExecutionResult
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.container_executor import CommandLineCodeResult

logger = logging.getLogger(__name__)


class MCPPythonEnvironment(Environment):
    """Environment using (Pydantic) MCP Run Python server for sandboxed code execution."""
    
    def __init__(self):
        super().__init__()
        
        # Set up environment variables for Deno to use a writable cache directory
        import tempfile
        
        # Create a temporary directory for Deno cache in a writable location
        self.deno_cache_dir = tempfile.mkdtemp(prefix="deno_cache_")
        
        # Set environment variables for Deno
        self.env_vars = {
            'DENO_DIR': self.deno_cache_dir,
            'DENO_CACHE_DIR': self.deno_cache_dir,
        }
        
        self.server_params = StdioServerParameters(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=node_modules',
                '-W=node_modules',
                '--node-modules-dir=auto',
                'jsr:@pydantic/mcp-run-python',
                'stdio',
            ],
            env=self.env_vars,
        )
        logger.info(f"MCP Python environment initialized with cache dir: {self.deno_cache_dir}")
    
    def __del__(self):
        """Clean up temporary cache directory."""
        try:
            import shutil
            if hasattr(self, 'deno_cache_dir') and os.path.exists(self.deno_cache_dir):
                shutil.rmtree(self.deno_cache_dir)
                logger.debug(f"Cleaned up Deno cache dir: {self.deno_cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up Deno cache dir: {e}")
    
    def launch(self, port: int):
        """Launch the environment as a server."""
        from omegaconf import OmegaConf
        
        env_server = EnvironmentServer(
            n_envs=1,
            host="0.0.0.0",
            port=port,
            max_session_inactivity_secs=600
        )
        
        env_server.launch(OmegaConf.create({
            "_target_": "pipelinerl.domains.tir.environment.MCPPythonEnvironment"
        }))
    
    def react(self, tape):
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        
        for action in actions:
            if not isinstance(action, PythonCodeAction):
                continue
                
            try:
                logger.info(f"Executing Python code via MCP: {repr(action.code[:100])}...")
                
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    
                    def run_in_thread():
                        return asyncio.run(self._execute_python_code(action.code))
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result(timeout=90)
                        
                except RuntimeError:
                    result = asyncio.run(self._execute_python_code(action.code))
                
                # logger.info(f"MCP execution result: {repr(result[:200])}...")
                logger.info(f"MCP execution result: {repr(result)}")
                
                output, success = self._parse_mcp_result(result)
                
                observation = CodeExecutionResult(
                    result=CommandLineCodeResult(
                        output=output,
                        exit_code=0 if success else 1
                    )
                )
                
                tape = tape.append(observation)
                
            except TimeoutError as e:
                logger.warning(f"Code execution timed out: {e}")
                tape = tape.append(ActionExecutionFailure(error=f"Timeout: {e}"))
                break
            except Exception as e:
                logger.error(f"MCP execution failed: {e}")
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
                
        return tape
    
    async def _execute_python_code(self, code: str) -> str:
        """Execute Python code using MCP Run Python server"""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Add timeout to the actual MCP call
                try:
                    result = await asyncio.wait_for(
                        session.call_tool('run_python_code', {'python_code': code}),
                        timeout=90.0  # 90 second timeout for individual code execution
                    )
                    return result.content[0].text
                except asyncio.TimeoutError:
                    raise TimeoutError("Code execution timed out after 90 seconds")
                except Exception as e:
                    logger.error(f"MCP execution failed: {e}")
                    raise e
    
    def _parse_mcp_result(self, mcp_output: str) -> tuple[str, bool]:
        """Parse MCP output to extract result and determine success."""
        if "<status>error</status>" in mcp_output:
            if "<stderr>" in mcp_output and "</stderr>" in mcp_output:
                start = mcp_output.find("<stderr>") + len("<stderr>")
                end = mcp_output.find("</stderr>")
                error_msg = mcp_output[start:end].strip()
                return f"Error: {error_msg}", False
            else:
                return "Error: Code execution failed", False
        
        # Check for <output> tags first (common in MCP responses)
        if "<output>" in mcp_output and "</output>" in mcp_output:
            start = mcp_output.find("<output>") + len("<output>")
            end = mcp_output.find("</output>")
            output = mcp_output[start:end].strip()
            return output if output else "No output produced", True
        
        if "<o>" in mcp_output and "</o>" in mcp_output:
            start = mcp_output.find("<o>") + len("<o>")
            end = mcp_output.find("</o>")
            output = mcp_output[start:end].strip()
            
            if output.startswith("[") and output.endswith("]"):
                output = output[1:-1].strip()
            
            return output if output else "No output produced", True
        
        elif "<return_value>" in mcp_output and "</return_value>" in mcp_output:
            start = mcp_output.find("<return_value>") + len("<return_value>")
            end = mcp_output.find("</return_value>")
            return_value = mcp_output[start:end].strip()
            
            if return_value.startswith("[") and return_value.endswith("]"):
                return_value = return_value[1:-1].strip()
            
            return return_value, True
        
        elif "<stderr>" in mcp_output and "</stderr>" in mcp_output:
            start = mcp_output.find("<stderr>") + len("<stderr>")
            end = mcp_output.find("</stderr>")
            error_msg = mcp_output[start:end].strip()
            
            if "Traceback" in error_msg:
                lines = error_msg.split('\n')
                last_line = lines[-1] if lines else error_msg
                return f"Error: {last_line}", False
            else:
                return f"Error: {error_msg}", False
        
        else:
            clean_output = mcp_output.strip()
            return clean_output if clean_output else "No output produced", True


class AsyncMCPPythonEnvironment(AsyncEnvironment):
    """Async Environment using (Pydantic) MCP Run Python server for sandboxed code execution."""
    
    def __init__(self):
        super().__init__()
        
        # Set up environment variables for Deno to use a writable cache directory
        import tempfile
        
        # Create a temporary directory for Deno cache in a writable location
        self.deno_cache_dir = tempfile.mkdtemp(prefix="deno_cache_")
        
        # Set environment variables for Deno
        self.env_vars = {
            'DENO_DIR': self.deno_cache_dir,
            'DENO_CACHE_DIR': self.deno_cache_dir,
        }
        
        self.server_params = StdioServerParameters(
            command='/home/toolkit/.deno/bin/deno',
            args=[
                'run',
                '-N',
                '-R=node_modules',
                '-W=node_modules',
                '--node-modules-dir=auto',
                'jsr:@pydantic/mcp-run-python',
                'stdio',
            ],
            env=self.env_vars,
        )
        logger.info(f"Async MCP Python environment initialized with cache dir: {self.deno_cache_dir}")
    
    def __del__(self):
        """Clean up temporary cache directory."""
        try:
            import shutil
            if hasattr(self, 'deno_cache_dir') and os.path.exists(self.deno_cache_dir):
                shutil.rmtree(self.deno_cache_dir)
                logger.debug(f"Cleaned up Deno cache dir: {self.deno_cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up Deno cache dir: {e}")
    
    def launch(self, port: int):
        """Launch the environment as a server."""
        from omegaconf import OmegaConf
        
        env_server = EnvironmentServer(
            n_envs=1,
            host="0.0.0.0",
            port=port,
            max_session_inactivity_secs=600
        )
        
        env_server.launch(OmegaConf.create({
            "_target_": "pipelinerl.domains.tir.environment.AsyncMCPPythonEnvironment"
        }))

    def react(self, tape):
        raise NameError("react not supported in async env, use areact")

    async def areact(self, tape):
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        
        for action in actions:
            if not isinstance(action, PythonCodeAction):
                continue
                
            try:
                logger.info(f"Executing Python code via MCP: {repr(action.code[:100])}...")
                
                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    
                    def run_in_thread():
                        return asyncio.run(self._execute_python_code(action.code))
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_thread)
                        result = future.result(timeout=90)
                        
                except RuntimeError:
                    result = asyncio.run(self._execute_python_code(action.code))
                
                # logger.info(f"MCP execution result: {repr(result[:200])}...")
                logger.info(f"MCP execution result: {repr(result)}")
                
                output, success = self._parse_mcp_result(result)
                
                observation = CodeExecutionResult(
                    result=CommandLineCodeResult(
                        output=output,
                        exit_code=0 if success else 1
                    )
                )
                
                tape = tape.append(observation)
                
            except TimeoutError as e:
                logger.warning(f"Code execution timed out: {e}")
                tape = tape.append(ActionExecutionFailure(error=f"Timeout: {e}"))
                break
            except Exception as e:
                logger.error(f"MCP execution failed: {e}")
                tape = tape.append(ActionExecutionFailure(error=str(e)))
                break
                
        return tape
    
    async def _execute_python_code(self, code: str) -> str:
        """Execute Python code using MCP Run Python server"""
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Add timeout to the actual MCP call
                try:
                    result = await asyncio.wait_for(
                        session.call_tool('run_python_code', {'python_code': code}),
                        timeout=90.0  # 90 second timeout for individual code execution
                    )
                    return result.content[0].text
                except asyncio.TimeoutError:
                    raise TimeoutError("Code execution timed out after 90 seconds")
                except Exception as e:
                    logger.error(f"MCP execution failed: {e}")
                    raise e
    
    def _parse_mcp_result(self, mcp_output: str) -> tuple[str, bool]:
        """Parse MCP output to extract result and determine success."""
        if "<status>error</status>" in mcp_output:
            if "<stderr>" in mcp_output and "</stderr>" in mcp_output:
                start = mcp_output.find("<stderr>") + len("<stderr>")
                end = mcp_output.find("</stderr>")
                error_msg = mcp_output[start:end].strip()
                return f"Error: {error_msg}", False
            else:
                return "Error: Code execution failed", False
        
        # Check for <output> tags first (common in MCP responses)
        if "<output>" in mcp_output and "</output>" in mcp_output:
            start = mcp_output.find("<output>") + len("<output>")
            end = mcp_output.find("</output>")
            output = mcp_output[start:end].strip()
            return output if output else "No output produced", True
        
        if "<o>" in mcp_output and "</o>" in mcp_output:
            start = mcp_output.find("<o>") + len("<o>")
            end = mcp_output.find("</o>")
            output = mcp_output[start:end].strip()
            
            if output.startswith("[") and output.endswith("]"):
                output = output[1:-1].strip()
            
            return output if output else "No output produced", True
        
        elif "<return_value>" in mcp_output and "</return_value>" in mcp_output:
            start = mcp_output.find("<return_value>") + len("<return_value>")
            end = mcp_output.find("</return_value>")
            return_value = mcp_output[start:end].strip()
            
            if return_value.startswith("[") and return_value.endswith("]"):
                return_value = return_value[1:-1].strip()
            
            return return_value, True
        
        elif "<stderr>" in mcp_output and "</stderr>" in mcp_output:
            start = mcp_output.find("<stderr>") + len("<stderr>")
            end = mcp_output.find("</stderr>")
            error_msg = mcp_output[start:end].strip()
            
            if "Traceback" in error_msg:
                lines = error_msg.split('\n')
                last_line = lines[-1] if lines else error_msg
                return f"Error: {last_line}", False
            else:
                return f"Error: {error_msg}", False
        
        else:
            clean_output = mcp_output.strip()
            return clean_output if clean_output else "No output produced", True
