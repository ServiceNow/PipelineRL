import asyncio
import logging
import os
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tapeagents.remote_environment import EnvironmentServer

from tapeagents.environment import Environment
from tapeagents.core import Action
from tapeagents.tools.code_executor import PythonCodeAction, CodeExecutionResult
from tapeagents.steps import ActionExecutionFailure
from tapeagents.tools.container_executor import CommandLineCodeResult

logger = logging.getLogger(__name__)

# Global shared Deno setup to avoid per-environment complexity
_global_deno_setup_lock = threading.Lock()
_global_deno_setup_done = False
_global_deno_work_dir = None

try:
    from filelock import FileLock
    _deno_file_lock = FileLock("/tmp/deno_mcp_start.lock")
except ImportError:
    logger.warning("filelock not available, using threading lock only")
    _deno_file_lock = None


@asynccontextmanager
async def _stdio_client_with_stderr(server_params):
    """Wrap stdio_client and try to capture any Deno process errors."""
    try:
        async with stdio_client(server_params) as pipes:
            yield pipes
    except Exception as e:
        logger.error(f"MCP stdio_client failed: {e}")
        try:
            result = subprocess.run([
                server_params.command,
                *server_params.args
            ], 
            env=server_params.env,
            cwd=server_params.cwd,
            capture_output=True, 
            text=True, 
            timeout=10,
            input=""
            )
            if result.stderr:
                logger.error("⇢ Deno stderr:\n%s\n⇠ end stderr", result.stderr.strip())
            if result.stdout:
                logger.error("⇢ Deno stdout:\n%s\n⇠ end stdout", result.stdout.strip())
            logger.error(f"Deno exit code: {result.returncode}")
        except Exception as debug_e:
            logger.error(f"Failed to debug Deno command: {debug_e}")
        raise e


def _ensure_global_deno_setup():
    """Ensure Deno and MCP package are set up globally once."""
    global _global_deno_setup_done, _global_deno_work_dir
    
    with _global_deno_setup_lock:
        if _global_deno_setup_done:
            return _global_deno_work_dir
            
        logger.info("Setting up global Deno environment for MCP")
        
        _global_deno_work_dir = tempfile.mkdtemp(prefix="deno_global_")
        
        env_vars = {
            'PATH': os.environ.get('PATH', ''),
            'DENO_NO_UPDATE_CHECK': '1',
        }
        
        # install Deno if not found
        deno_install_dir = os.environ.get('DENO_INSTALL', os.path.expanduser('~/.deno'))
        deno_bin_path = os.path.join(deno_install_dir, 'bin', 'deno')
        
        if not os.path.exists(deno_bin_path):
            logger.info(f"Installing Deno to {deno_install_dir}")
            try:
                install_cmd = f'curl -fsSL https://deno.land/install.sh | bash -s -- -q -d {deno_install_dir}'
                subprocess.run(install_cmd, shell=True, check=True, timeout=120, 
                             env={**env_vars, 'DENO_INSTALL': deno_install_dir})
                logger.info(f"Deno installed successfully to {deno_bin_path}")
            except Exception as e:
                logger.error(f"Deno installation failed: {e}")
                return None
        
        deno_bin_dir = os.path.join(deno_install_dir, 'bin')
        env_vars['PATH'] = f"{deno_bin_dir}:{env_vars['PATH']}"
        
        os.environ['PATH'] = env_vars['PATH']
        
        try:
            deno_test = subprocess.run(['deno', '--version'], env=env_vars, capture_output=True, text=True, timeout=10)
            logger.info(f"Deno version: {deno_test.stdout.strip()}")
        except Exception as e:
            logger.error(f"Deno test failed: {e}")
            return None
            
        # Pre-cache MCP package globally
        for attempt in range(2):
            try:
                subprocess.run([
                    'deno', 'cache', 
                    '--node-modules-dir=auto',
                    'jsr:@pydantic/mcp-run-python'
                ], env=env_vars, cwd=_global_deno_work_dir, check=True, timeout=120)
                logger.info("Global MCP package caching completed")
                _global_deno_setup_done = True
                return _global_deno_work_dir
            except Exception as e:
                if attempt == 0 and "Failed reading cache entry" in str(e):
                    logger.warning(f"Deno cache corruption detected, clearing cache and retrying: {e}")
                    try:
                        subprocess.run(['deno', 'cache', '--reload', 'jsr:@pydantic/mcp-run-python'], 
                                     env=env_vars, cwd=_global_deno_work_dir, timeout=120)
                    except Exception as clear_e:
                        logger.error(f"Cache clear failed: {clear_e}")
                else:
                    logger.error(f"Global MCP package caching failed: {e}")
                    return None


class MCPPythonEnvironment(Environment):
    """Environment using (Pydantic) MCP Run Python server for sandboxed code execution."""
    
    # Class-level lock to serialize Deno session creation within each Python worker
    _deno_lock = asyncio.Lock()
    
    def __init__(self):
        super().__init__()
        
        # do Deno setup lazily when needed
        self.work_dir = None
        
        self.env_vars = None
        
        self.server_params = None
        logger.info("MCP Python environment initialized (Deno setup will be done lazily)")
    
    def _ensure_setup(self):
        """Ensure Deno setup is complete (called lazily when needed)."""
        if self.work_dir is None:
            self.work_dir = _ensure_global_deno_setup()
            if not self.work_dir:
                raise RuntimeError("Failed to set up global Deno environment")
            
            deno_install_dir = os.environ.get('DENO_INSTALL', os.path.expanduser('~/.deno'))
            deno_bin_dir = os.path.join(deno_install_dir, 'bin')
            
            current_path = os.environ.get('PATH', '')
            if deno_bin_dir not in current_path:
                new_path = f"{deno_bin_dir}:{current_path}"
            else:
                new_path = current_path
                
            self.env_vars = {
                'PATH': new_path,
                'DENO_NO_UPDATE_CHECK': '1',
            }
            
            self.work_dir = tempfile.mkdtemp(prefix="mcp_env_")

            self.server_params = StdioServerParameters(
                command='deno',
                args=[
                    'run',
                    '-A',
                    '--quiet',
                    'jsr:@pydantic/mcp-run-python',
                    'stdio',
                ],
                env=self.env_vars,
                cwd=self.work_dir,
            )
            logger.info(f"MCP Python environment setup completed with work dir: {self.work_dir}")
    
    def __del__(self):
        """No cleanup needed - using global shared directory."""
        pass
    
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
        """Synchronous react method for backward compatibility."""
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
    
    async def areact(self, tape):
        """Async react method for use with async_execute_agent."""
        actions = [step for step in tape.steps[-tape.metadata.n_added_steps :] if isinstance(step, Action)]
        
        for action in actions:
            if not isinstance(action, PythonCodeAction):
                continue
                
            try:
                logger.info(f"Executing Python code via MCP: {repr(action.code[:100])}...")
                
                result = await self._execute_python_code(action.code)
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
        """Execute Python code using MCP Run Python server with retry mechanism and serialization."""
        self._ensure_setup()
        
        async with self._deno_lock:
            if _deno_file_lock:
                with _deno_file_lock:
                    return await self._run_mcp_session(code)
            else:
                return await self._run_mcp_session(code)
    
    async def _run_mcp_session(self, code: str) -> str:
        """Run MCP session with retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                logger.debug(f"MCP execution attempt {attempt + 1}/{max_retries}")
                
                test_result = subprocess.run([
                    self.server_params.command, '--version'
                ], env=self.server_params.env, capture_output=True, text=True, timeout=5)
                if test_result.returncode != 0:
                    logger.error(f"Deno version check failed: {test_result.stderr}")
                    raise RuntimeError(f"Deno not working: {test_result.stderr}")
                logger.debug(f"Deno version check passed: {test_result.stdout.strip()}")
                
                async with _stdio_client_with_stderr(self.server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        try:
                            result = await asyncio.wait_for(
                                session.call_tool('run_python_code', {'python_code': code}),
                                timeout=30.0
                            )
                            return result.content[0].text
                        except asyncio.TimeoutError:
                            raise TimeoutError("Code execution timed out after 30 seconds")
                        except Exception as e:
                            logger.error(f"MCP tool call failed on attempt {attempt + 1}: {e}")
                            if attempt == max_retries - 1:
                                raise e
                            await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"MCP session setup failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"Server params: {self.server_params}")
                    logger.error(f"Environment vars: {self.env_vars}")
                    raise RuntimeError(f"MCP execution failed after {max_retries} attempts. This indicates a serious issue with the Deno/MCP setup. Code execution cannot proceed safely without proper sandboxing.")
                elif "Failed reading cache entry" in str(e):
                    logger.warning(f"Deno cache corruption detected during MCP execution, clearing cache: {e}")
                    try:
                        subprocess.run([
                            'deno', 'cache', '--reload', 'jsr:@pydantic/mcp-run-python'
                        ], env=self.server_params.env, cwd=self.server_params.cwd, timeout=30)
                        logger.info("Cache cleared successfully")
                    except Exception as clear_e:
                        logger.error(f"Failed to clear Deno cache: {clear_e}")
                backoff_time = 0.5 + attempt * 0.5
                await asyncio.sleep(backoff_time)
    
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
