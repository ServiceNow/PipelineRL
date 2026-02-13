"""Integration tests for vllm1 with actual distributed setup."""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List
import time
import os
import subprocess
import sys
import signal

# torch is needed at top level for pytest.mark.skipif decorators
import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not available, process tree cleanup will be limited")


def stream_process_output(proc, name):
    """Start background threads to continuously stream process stdout/stderr.

    Args:
        proc: subprocess.Popen object
        name: Name for logging prefix (e.g., "vLLM Server", "Trainer")

    Returns:
        Tuple of (stdout_thread, stderr_thread)
    """
    import threading

    def read_stream(stream, prefix):
        """Read from stream and print with prefix."""
        try:
            for line in iter(stream.readline, ''):
                if line:
                    print(f"{prefix} {line.rstrip()}", flush=True)
        except Exception as e:
            print(f"{prefix} [Stream read error: {e}]", flush=True)

    stdout_thread = threading.Thread(
        target=read_stream,
        args=(proc.stdout, f"[{name} OUT]"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=read_stream,
        args=(proc.stderr, f"[{name} ERR]"),
        daemon=True,
    )

    stdout_thread.start()
    stderr_thread.start()

    return stdout_thread, stderr_thread


def kill_process_tree(pid, sig=signal.SIGKILL):
    """Kill a process and all its children/grandchildren.

    Args:
        pid: Process ID to kill
        sig: Signal to send (default SIGKILL)
    """
    if not HAS_PSUTIL:
        # Fallback: just kill the main process
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        return

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Kill children first
    for child in children:
        try:
            print(f"[Kill] Killing child process {child.pid}")
            child.send_signal(sig)
        except psutil.NoSuchProcess:
            pass

    # Kill parent
    try:
        parent.send_signal(sig)
    except psutil.NoSuchProcess:
        pass


def force_kill_process(proc, name):
    """Forcefully kill a process tree and collect output.

    SIGKILL always kills the process. If communicate() hangs, it's the PIPES
    that are stuck, not the process. We handle this with retries and timeouts.

    Returns:
        Tuple of (stdout, stderr, returncode)
    """
    # If already dead, try to get output
    if proc.poll() is not None:
        try:
            stdout, stderr = proc.communicate(timeout=2)
            return stdout, stderr, proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[Kill] {name} already dead but pipes hung, closing...")
            proc.stdout.close() if proc.stdout else None
            proc.stderr.close() if proc.stderr else None
            return "<pipes hung>", "<pipes hung>", proc.returncode

    # Kill entire process tree (including vLLM workers, trainer subprocesses, etc)
    print(f"[Kill] Killing {name} process tree (PID {proc.pid})...")
    kill_process_tree(proc.pid, signal.SIGKILL)

    # Wait for main process to actually die
    try:
        proc.wait(timeout=2)
        print(f"[Kill] {name} process tree killed")
    except subprocess.TimeoutExpired:
        print(f"[Kill] WARNING: {name} didn't die after SIGKILL")

    # Try to read output from pipes (this is what usually hangs)
    for attempt, timeout_val in enumerate([1, 2, 3], start=1):
        try:
            stdout, stderr = proc.communicate(timeout=timeout_val)
            print(f"[Kill] {name} output collected (attempt {attempt})")
            return stdout, stderr, proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[Kill] {name} communicate() timed out (attempt {attempt})")
            continue

    # Pipes are stuck - force close them
    print(f"[Kill] {name} pipes stuck, force closing...")
    try:
        proc.stdout.close() if proc.stdout else None
        proc.stderr.close() if proc.stderr else None
        proc.stdin.close() if proc.stdin else None
    except Exception as e:
        print(f"[Kill] Error closing pipes: {e}")

    return "<pipes stuck>", "<pipes stuck>", proc.returncode if proc.returncode else -999


async def wait_for_processes(processes_with_names, check_interval=0.5, timeout=60):
    """Wait for multiple subprocesses to complete, printing output in real-time.

    Args:
        processes_with_names: List of (subprocess.Popen, name) tuples
        check_interval: How often to check process status (seconds)
        timeout: Maximum time to wait for all processes (seconds)

    Raises:
        RuntimeError: If any process fails or timeout is reached
    """
    start_time = time.time()

    # Create async readers for each process's stdout and stderr
    async def read_stream(stream, prefix):
        """Read from a stream line-by-line and print with prefix."""
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, stream.readline)
                if not line:
                    break
                print(f"{prefix} {line.rstrip()}", flush=True)
        except Exception as e:
            print(f"{prefix} [Read error: {e}]", flush=True)

    # Start readers for all processes
    reader_tasks = []
    for proc, name in processes_with_names:
        reader_tasks.append(asyncio.create_task(read_stream(proc.stdout, f"[{name} OUT]")))
        reader_tasks.append(asyncio.create_task(read_stream(proc.stderr, f"[{name} ERR]")))

    try:
        while True:
            # Check if timeout exceeded
            if time.time() - start_time > timeout:
                print(f"\n{'='*60}", flush=True)
                print("TIMEOUT: Killing all processes", flush=True)
                print(f"{'='*60}\n", flush=True)

                # Kill all processes forcefully
                for proc, name in processes_with_names:
                    if proc.poll() is None:
                        print(f"[Main] Killing {name}...", flush=True)
                        kill_process_tree(proc.pid, signal.SIGKILL)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass

                raise RuntimeError(f"Timeout after {timeout} seconds waiting for processes")

            # Check each process
            crashed_proc = None
            crashed_name = None

            for proc, name in processes_with_names:
                returncode = proc.poll()
                if returncode is not None and returncode != 0:
                    crashed_proc = proc
                    crashed_name = name
                    print(f"\n{'='*60}", flush=True)
                    print(f"{name} process CRASHED with exit code {returncode}", flush=True)
                    print(f"{'='*60}\n", flush=True)
                    break

            # If a process crashed, kill the others
            if crashed_proc is not None:
                # Kill all other processes
                for proc, name in processes_with_names:
                    if proc != crashed_proc and proc.poll() is None:
                        print(f"[Main] Killing {name}...", flush=True)
                        kill_process_tree(proc.pid, signal.SIGKILL)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass

                raise RuntimeError(
                    f"{crashed_name} process failed with exit code {crashed_proc.returncode}"
                )

            # Check if all processes completed successfully
            all_done = all(proc.poll() is not None for proc, _ in processes_with_names)
            if all_done:
                # Wait for readers to finish draining pipes
                print("[Main] All processes completed, waiting for output to finish...", flush=True)
                await asyncio.sleep(1)  # Give readers time to finish

                print(f"\n{'='*60}", flush=True)
                print("✓ All processes completed successfully", flush=True)
                print(f"{'='*60}\n", flush=True)
                return

            # Sleep before next check
            await asyncio.sleep(check_interval)
    finally:
        # Cancel reader tasks
        for task in reader_tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellation
        await asyncio.gather(*reader_tasks, return_exceptions=True)


class TestBasicGeneration:
    """Test basic vLLM generation with worker extension."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    async def test_load_model_and_generate(self, vllm_engine_factory, simple_prompt, generation_config):
        """Test loading model and generating text."""
        from vllm import SamplingParams

        async with vllm_engine_factory(disable_weight_updates=True) as manager:
            # Generate text
            sampling_params = SamplingParams(
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                max_tokens=generation_config["max_tokens"],
                seed=generation_config["seed"],
            )

            request_id = "test_request_1"
            async for output in manager.engine.generate(
                simple_prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output

            assert final_output is not None
            assert len(final_output.outputs) > 0
            assert len(final_output.outputs[0].text) > 0

            print(f"Generated text: {final_output.outputs[0].text}")

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    async def test_deterministic_generation(self, vllm_engine_factory, simple_prompt, generation_config):
        """Test that generation is deterministic with same seed and temperature=0."""
        from vllm import SamplingParams

        async with vllm_engine_factory(disable_weight_updates=True) as manager:
            sampling_params = SamplingParams(
                temperature=generation_config["temperature"],
                top_p=generation_config["top_p"],
                max_tokens=generation_config["max_tokens"],
                seed=generation_config["seed"],
            )

            # Generate twice with same parameters
            outputs = []
            for i in range(2):
                request_id = f"test_request_{i}"
                async for output in manager.engine.generate(
                    simple_prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    final_output = output
                outputs.append(final_output.outputs[0].text)

            # Outputs should be identical
            assert outputs[0] == outputs[1], f"Outputs differ: '{outputs[0]}' vs '{outputs[1]}'"


class TestWorkerExtension:
    """Test WorkerExtension loading and methods."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
    async def test_extension_loaded(self, vllm_engine_factory):
        """Test that WorkerExtension is properly loaded."""
        from vllm.v1.engine.core_client import AsyncMPClient

        async with vllm_engine_factory(disable_weight_updates=True) as manager:
            # Check that engine has the extension methods
            assert isinstance(manager.engine.engine_core, AsyncMPClient)

            # Test that we can call the extension method
            # This verifies the extension is loaded on workers
            # collective_rpc_async returns a list of results (one per worker)
            results = await manager.is_extension_loaded()
            # Extension should be loaded on all workers
            assert isinstance(results, list)
            assert len(results) > 0  # At least one worker
            # Results are PIDs (integers > 0)
            assert all(isinstance(r, int) and r > 0 for r in results), f"Expected PIDs, got: {results}"
            print(f"WorkerExtension successfully loaded on {len(results)} worker(s)")
            print(f"Worker PIDs: {results}")
            print(f"Unique PIDs: {len(set(results))} (indicates {len(set(results))} separate processes)")


class TestWeightUpdateDistributed:
    """Test weight updates with 2-GPU distributed setup."""

    @pytest.mark.timeout(300)  # 5 minutes for init test
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    async def test_init_actor_update_group(
        self,
        model_name,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
    ):
        """Test initializing actor update group with 2 GPUs.

        This test verifies that the process group can be initialized correctly:
        - vLLM engine runs on GPU 0 as rank 1 (in subprocess)
        - Dummy trainer process runs on GPU 1 as rank 0 (in subprocess)

        Both run in subprocesses to ensure proper CUDA_VISIBLE_DEVICES isolation.
        """
        print("\n" + "="*60)
        print("Starting distributed process group initialization test")
        print("="*60)

        # Step 1: Start trainer subprocess FIRST with CUDA_VISIBLE_DEVICES=1
        trainer_env = os.environ.copy()
        trainer_env["CUDA_VISIBLE_DEVICES"] = "1"
        trainer_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting trainer process (rank 0, GPU 1)")
        trainer_proc = subprocess.Popen(
            [
                sys.executable,
                str(distributed_trainer_helper),
                "init",
                "--init-method", distributed_init_method,
                "--rank", "0",
                "--world-size", "2",
            ],
            env=trainer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give trainer a moment to start and begin initializing
        await asyncio.sleep(1)

        # Step 2: Start vLLM engine subprocess with CUDA_VISIBLE_DEVICES=0
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
        vllm_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting vLLM engine process (rank 1, GPU 0)")
        vllm_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_engine_helper),
                "init",  # Command argument
                "--model-name", model_name,
                "--init-method", distributed_init_method,
                "--actor-llm-idx", "0",
                "--world-size", "2",
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Step 3: Wait for both processes, killing all if one crashes
        await wait_for_processes([
            (trainer_proc, "Trainer"),
            (vllm_proc, "vLLM Engine"),
        ], timeout=180)  # Init test is faster, but give it 3 minutes to be safe

    @pytest.mark.timeout(1000)  # 1000 seconds for broadcasting 291 parameters
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    async def test_weight_update_same_weights(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        temp_dir,
    ):
        """Test that updating with same weights produces same output.

        This test:
        1. vLLM engine generates baseline output (in subprocess on GPU 0)
        2. Trainer waits for baseline, then broadcasts weights (in subprocess on GPU 1)
        3. vLLM engine receives update and generates again
        4. vLLM engine verifies outputs are identical

        Both run in subprocesses for proper CUDA_VISIBLE_DEVICES isolation.
        Uses file-based sync points for coordination.
        """
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting weight update test (same weights)")
        print("="*60)

        # Create sync directory for coordination
        sync_dir = create_sync_dir(temp_dir)
        print(f"[Main] Sync directory: {sync_dir}")

        # Step 1: Start vLLM engine subprocess with weight_update command
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
        # NOTE: needed to pass WeightUpdateRequest to collective
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        # Enable DEBUG logging in vllm1.py
        vllm_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting vLLM engine process (GPU 0)")
        vllm_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_engine_helper),
                "weight_update",
                "--model-name", model_name,
                "--init-method", distributed_init_method,
                "--actor-llm-idx", "0",
                "--world-size", "2",
                "--prompt", simple_prompt,
                "--max-tokens", str(generation_config["max_tokens"]),
                "--sync-dir", str(sync_dir),
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give vLLM engine a moment to start
        await asyncio.sleep(1)

        # Step 2: Start trainer subprocess (will wait for baseline_done sync point)
        trainer_env = os.environ.copy()
        trainer_env["CUDA_VISIBLE_DEVICES"] = "1"
        # Enable DEBUG logging in vllm1.py (for consistency)
        trainer_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting trainer process (GPU 1)")
        trainer_proc = subprocess.Popen(
            [
                sys.executable,
                str(distributed_trainer_helper),
                "broadcast",
                "--init-method", distributed_init_method,
                "--model-name", model_name,
                "--sync-dir", str(sync_dir),
            ],
            env=trainer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Step 3: Wait for both processes, killing all if one crashes
        # 291 parameters takes ~600 seconds, so use 900s (15 min) to be safe
        await wait_for_processes([
            (vllm_proc, "vLLM Engine"),
            (trainer_proc, "Trainer"),
        ], timeout=900)

    @pytest.mark.timeout(1000)  # 1000 seconds for broadcasting 290 parameters
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    async def test_weight_update_different_weights(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        temp_dir,
    ):
        """Test that updating with perturbed weights produces different output.

        This test:
        1. vLLM engine generates baseline output (in subprocess on GPU 0)
        2. Trainer broadcasts PERTURBED weights (in subprocess on GPU 1)
        3. vLLM engine receives update and generates again
        4. vLLM engine verifies outputs are DIFFERENT (perturbed weights changed output)

        Both run in subprocesses for proper CUDA_VISIBLE_DEVICES isolation.
        Uses file-based sync points for coordination.
        """
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting weight update test (perturbed weights)")
        print("="*60)

        # Create sync directory for coordination
        sync_dir = create_sync_dir(temp_dir)
        print(f"[Main] Sync directory: {sync_dir}")

        # Step 1: Start vLLM engine subprocess with weight_update command
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
        # NOTE: needed to pass WeightUpdateRequest to collective
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        # Enable DEBUG logging in vllm1.py
        vllm_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting vLLM engine process (GPU 0)")
        vllm_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_engine_helper),
                "weight_update",
                "--model-name", model_name,
                "--init-method", distributed_init_method,
                "--actor-llm-idx", "0",
                "--world-size", "2",
                "--prompt", simple_prompt,
                "--max-tokens", str(generation_config["max_tokens"]),
                "--sync-dir", str(sync_dir),
                "--expect-different",  # Flag to expect different outputs
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give vLLM engine a moment to start
        await asyncio.sleep(1)

        # Step 2: Start trainer subprocess with --perturb flag
        trainer_env = os.environ.copy()
        trainer_env["CUDA_VISIBLE_DEVICES"] = "1"
        # Enable DEBUG logging in vllm1.py (for consistency)
        trainer_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting trainer process (GPU 1) with --perturb")
        trainer_proc = subprocess.Popen(
            [
                sys.executable,
                str(distributed_trainer_helper),
                "broadcast",
                "--init-method", distributed_init_method,
                "--model-name", model_name,
                "--sync-dir", str(sync_dir),
                "--perturb",  # Perturb weights to test different outputs
            ],
            env=trainer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Step 3: Wait for both processes, killing all if one crashes
        # 290 parameters takes ~600 seconds, so use 900s (15 min) to be safe
        await wait_for_processes([
            (vllm_proc, "vLLM Engine"),
            (trainer_proc, "Trainer"),
        ], timeout=900)


    @pytest.mark.timeout(2000)  # 2000 seconds - this test does 2 full broadcasts
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    async def test_weight_update_cross_validation(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        temp_dir,
    ):
        """Cross-validation test: verify broadcast = load from disk.

        This test validates that:
        1. Broadcasting weights produces same results as loading from disk
        2. Round-trip works: original → modified → original

        Flow:
        - vLLM: Load original, generate res_un_1
        - Trainer: Save perturbed model to disk, broadcast perturbed weights
        - vLLM: Receive perturbed, generate res_mod_1
        - vLLM: Recreate engine with perturbed model from disk, generate res_mod_2
        - Trainer: Broadcast original weights
        - vLLM: Receive original, generate res_un_2

        Assertions:
        - res_un_1 == res_un_2 (original weights produce same output)
        - res_mod_1 == res_mod_2 (broadcast = load from disk)
        """
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting cross-validation test")
        print("="*60)

        # Create sync directory for coordination
        sync_dir = create_sync_dir(temp_dir)
        print(f"[Main] Sync directory: {sync_dir}")
        print(f"[Main] Temp directory: {temp_dir}")

        # Step 1: Start vLLM engine subprocess
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        vllm_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting vLLM engine process (GPU 0)")
        vllm_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_engine_helper),
                "cross_validation",
                "--model-name", model_name,
                "--init-method", distributed_init_method,
                "--actor-llm-idx", "0",
                "--world-size", "2",
                "--prompt", simple_prompt,
                "--max-tokens", str(generation_config["max_tokens"]),
                "--sync-dir", str(sync_dir),
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give vLLM engine a moment to start
        await asyncio.sleep(1)

        # Step 2: Start trainer subprocess
        trainer_env = os.environ.copy()
        trainer_env["CUDA_VISIBLE_DEVICES"] = "1"
        trainer_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting trainer process (GPU 1)")
        trainer_proc = subprocess.Popen(
            [
                sys.executable,
                str(distributed_trainer_helper),
                "cross_validation",
                "--init-method", distributed_init_method,
                "--model-name", model_name,
                "--sync-dir", str(sync_dir),
                "--temp-dir", str(temp_dir),
            ],
            env=trainer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Step 3: Wait for both processes
        # This test does 2 broadcasts, so double the timeout
        await wait_for_processes([
            (vllm_proc, "vLLM Engine"),
            (trainer_proc, "Trainer"),
        ], timeout=1800)  # 30 minutes


    @pytest.mark.timeout(2000)  # 2000 seconds - this test does 3 broadcasts
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    async def test_weight_update_back_and_forth(
        self,
        model_name,
        simple_prompt,
        generation_config,
        shared_distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        shared_test_dir,
    ):
        """Back-and-forth test: switch between original and perturbed weights.

        This test validates that:
        1. We can update weights multiple times
        2. We can switch back and forth between weight sets
        3. Updates are deterministic and reproducible

        Flow:
        - vLLM: Load original, generate res_or_1
        - Trainer: Broadcast perturbed weights
        - vLLM: Receive perturbed, generate res_mod_1
        - Trainer: Broadcast original weights
        - vLLM: Receive original, generate res_or_2
        - Trainer: Broadcast perturbed weights again (same as first)
        - vLLM: Receive perturbed, generate res_mod_2

        Assertions:
        - res_or_1 == res_or_2 (can restore original weights)
        - res_mod_1 == res_mod_2 (perturbed weights are consistent)
        """
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting back-and-forth test")
        print("="*60)

        # Create sync directory for coordination
        sync_dir = create_sync_dir(shared_test_dir)
        print(f"[Main] Sync directory: {sync_dir}")

        # Step 1: Start vLLM engine subprocess
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        vllm_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting vLLM engine process (GPU 0)")
        vllm_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_engine_helper),
                "back_and_forth",
                "--model-name", model_name,
                "--init-method", shared_distributed_init_method,
                "--actor-llm-idx", "0",
                "--world-size", "2",
                "--prompt", simple_prompt,
                "--max-tokens", str(generation_config["max_tokens"]),
                "--sync-dir", str(sync_dir),
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Give vLLM engine a moment to start
        await asyncio.sleep(1)

        # Step 2: Start trainer subprocess
        trainer_env = os.environ.copy()
        trainer_env["CUDA_VISIBLE_DEVICES"] = "1"
        trainer_env["PIPELINERL_DEBUG"] = "1"

        print("[Main] Starting trainer process (GPU 1)")
        trainer_proc = subprocess.Popen(
            [
                sys.executable,
                str(distributed_trainer_helper),
                "back_and_forth",
                "--init-method", shared_distributed_init_method,
                "--model-name", model_name,
                "--sync-dir", str(sync_dir),
            ],
            env=trainer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Step 3: Wait for both processes
        # This test does 3 broadcasts, so use longer timeout
        await wait_for_processes([
            (vllm_proc, "vLLM Engine"),
            (trainer_proc, "Trainer"),
        ], timeout=1800)  # 30 minutes

    @pytest.mark.timeout(2400)  # 40 minutes for server test
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
    async def test_server_weight_update_pattern(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        temp_dir,
    ):
        """Server integration test: verify weight update pattern with HTTP API.

        This test validates the real-world scenario where:
        1. vLLM server is running and serving HTTP requests
        2. Trainer broadcasts weight updates while server is active
        3. Server responses change based on weight updates

        Flow:
        - Start vLLM HTTP server (loads original model)
        - Continuously generate via HTTP API (deterministic)
        - Trainer: wait 15s → broadcast perturbed → wait 5s → broadcast original → wait 5s → broadcast perturbed
        - Verify generation pattern: original → perturbed → original → perturbed
        """
        import requests
        import time

        print("\n" + "="*60)
        print("Starting server weight update pattern test")
        print("="*60)

        # Start vLLM HTTP server
        server_port = 8000
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        print(f"[Main] Starting vLLM HTTP server on port {server_port} (GPU 0)")
        vllm_entry_point = Path(__file__).parent.parent / "pipelinerl" / "entrypoints" / "run_vllm1.py"
        server_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_entry_point),
                "--model", model_name,
                "--port", str(server_port),
                "--host", "127.0.0.1",
                "--actor-llm-idx", "0",
                "--weight-update-group-init-method", distributed_init_method,
                "--weight-update-group-world-size", "2",
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Start streaming server output in background threads
        print("[Main] Starting server output streaming...")
        server_stdout_thread, server_stderr_thread = stream_process_output(server_proc, "vLLM Server")

        # Give server a moment to start, then immediately start trainer
        # (they need to rendezvous for process group initialization)
        await asyncio.sleep(1)

        # Start trainer process immediately (needed for process group rendezvous)
        trainer_env = os.environ.copy()
        trainer_env["CUDA_VISIBLE_DEVICES"] = "1"

        print("[Main] Starting trainer process (GPU 1) for process group rendezvous")
        trainer_proc = subprocess.Popen(
            [
                sys.executable,
                str(distributed_trainer_helper),
                "timed_broadcast_server_test",
                "--init-method", distributed_init_method,
                "--model-name", model_name,
                "--server-url", f"http://127.0.0.1:{server_port}",
            ],
            env=trainer_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Start streaming trainer output in background threads
        print("[Main] Starting trainer output streaming...")
        trainer_stdout_thread, trainer_stderr_thread = stream_process_output(trainer_proc, "Trainer")

        try:
            # Wait for server to be ready
            print("[Main] Waiting for server to be ready...")
            server_ready = False
            for i in range(300):  # Wait up to 5 minutes
                # Check if server process crashed
                if server_proc.poll() is not None:
                    print(f"[Main] Server process terminated with code {server_proc.returncode}")
                    raise RuntimeError(f"Server process terminated with code {server_proc.returncode}")

                # Check if trainer process crashed
                if trainer_proc.poll() is not None:
                    print(f"[Main] Trainer process terminated with code {trainer_proc.returncode}")
                    raise RuntimeError(f"Trainer process terminated with code {trainer_proc.returncode}")

                try:
                    resp = requests.get(f"http://127.0.0.1:{server_port}/health", timeout=1)
                    if resp.status_code == 200:
                        server_ready = True
                        print("[Main] Server is ready!")
                        break
                except requests.exceptions.RequestException:
                    pass

                if i % 10 == 0:
                    print(f"[Main] Still waiting for server... ({i} seconds)")
                await asyncio.sleep(1)

            if not server_ready:
                raise TimeoutError("Server did not become ready within 5 minutes")

            # Continuously generate completions
            print("[Main] Starting continuous generation loop...")
            generations = []
            start_time = time.time()
            generation_interval = 0.5  # Generate every 0.5 seconds (more frequent)
            max_duration = 120  # Run for 120 seconds max (covers 15s + 3 broadcasts with 5s delays)

            def check_pattern_detected(generations):
                """Check if we have detected the full pattern (4 phases)."""
                if len(generations) < 4:
                    return False

                # Track when the text changes to identify phase boundaries
                phases = []
                current_text = None
                current_phase = []

                for ts, text in generations:
                    if text != current_text:
                        if current_phase:
                            phases.append((current_text, current_phase))
                        current_text = text
                        current_phase = [(ts, text)]
                    else:
                        current_phase.append((ts, text))

                # Add the last phase
                if current_phase:
                    phases.append((current_text, current_phase))

                # Check if we have at least 4 phases
                if len(phases) < 4:
                    return False

                # Verify the pattern: phase1 != phase2, phase3 == phase1, phase4 == phase2
                phase1_text = phases[0][0]
                phase2_text = phases[1][0]
                phase3_text = phases[2][0]
                phase4_text = phases[3][0]

                if phase1_text == phase2_text:
                    return False  # Phase 1 and 2 should be different
                if phase3_text != phase1_text:
                    return False  # Phase 3 should match Phase 1
                if phase4_text != phase2_text:
                    return False  # Phase 4 should match Phase 2

                return True

            while time.time() - start_time < max_duration:
                # Check if trainer is still running
                trainer_poll = trainer_proc.poll()
                if trainer_poll is not None:
                    print(f"[Main] Trainer exited with code {trainer_poll}")
                    break

                try:
                    # Generate via HTTP API
                    payload = {
                        "model": model_name,
                        "prompt": simple_prompt,
                        "max_tokens": generation_config["max_tokens"],
                        "temperature": 0.0,  # Deterministic
                        "top_p": 1.0,  # Must match engine params
                        "seed": 42,
                    }

                    resp = requests.post(
                        f"http://127.0.0.1:{server_port}/v1/completions",
                        json=payload,
                        timeout=30,
                    )

                    if resp.status_code == 200:
                        result = resp.json()
                        generated_text = result["choices"][0]["text"]
                        timestamp = time.time() - start_time
                        generations.append((timestamp, generated_text))
                        print(f"[Main] [{timestamp:.1f}s] Generated: '{generated_text}'")

                        # Check if pattern is detected - stop early if confirmed
                        if check_pattern_detected(generations):
                            print(f"[Main] Pattern detected! Stopping generation early at {timestamp:.1f}s")
                            break
                    else:
                        print(f"[Main] Generation failed with status {resp.status_code}")

                except requests.exceptions.RequestException as e:
                    print(f"[Main] Request failed: {e}")

                await asyncio.sleep(generation_interval)

            # Wait a bit more for trainer to finish
            print("[Main] Waiting for trainer to finish...")
            for _ in range(30):
                if trainer_proc.poll() is not None:
                    break
                await asyncio.sleep(1)

            # Analyze generation sequence
            print("\n" + "="*60)
            print("GENERATION SEQUENCE ANALYSIS")
            print("="*60)
            print(f"Total generations: {len(generations)}")

            # Print all generations
            for i, (ts, text) in enumerate(generations):
                print(f"[{ts:5.1f}s] Gen {i+1}: '{text[:80]}...'")

            # Identify unique generation texts and their phases
            # Expected pattern: original → perturbed → original → perturbed
            if len(generations) < 4:
                raise AssertionError(f"Not enough generations to verify pattern (need at least 4, got {len(generations)})")

            # Track when the text changes to identify phase boundaries
            phases = []
            current_text = None
            current_phase = []

            for ts, text in generations:
                if text != current_text:
                    if current_phase:
                        phases.append((current_text, current_phase))
                    current_text = text
                    current_phase = [(ts, text)]
                else:
                    current_phase.append((ts, text))

            # Add the last phase
            if current_phase:
                phases.append((current_text, current_phase))

            print("\n" + "="*60)
            print(f"Detected {len(phases)} phases:")
            for i, (text, items) in enumerate(phases):
                print(f"Phase {i+1}: {len(items)} generations - '{text[:60]}...'")
            print("="*60)

            # Verify the pattern
            assert len(phases) >= 4, f"Expected at least 4 phases (original → perturbed → original → perturbed), got {len(phases)}"

            phase1_text, phase1_items = phases[0]
            phase2_text, phase2_items = phases[1]
            phase3_text, phase3_items = phases[2]
            phase4_text, phase4_items = phases[3]

            # Verify phase 1 (original) != phase 2 (perturbed)
            assert phase1_text != phase2_text, "Phase 1 (original) and Phase 2 (perturbed) should be different"

            # Verify phase 3 (original) == phase 1 (original)
            assert phase3_text == phase1_text, f"Phase 3 should match Phase 1 (original weights restored)"

            # Verify phase 4 (perturbed) == phase 2 (perturbed)
            assert phase4_text == phase2_text, f"Phase 4 should match Phase 2 (perturbed weights reapplied)"

            print("\n✓ Pattern verified:")
            print(f"  Phase 1 (original):   {len(phase1_items)} generations")
            print(f"  Phase 2 (perturbed):  {len(phase2_items)} generations")
            print(f"  Phase 3 (original):   {len(phase3_items)} generations (matches Phase 1 ✓)")
            print(f"  Phase 4 (perturbed):  {len(phase4_items)} generations (matches Phase 2 ✓)")
            print("\n✓ Server weight update pattern test PASSED")

        finally:
            # Cleanup - always kill process tree even if main process exited
            # (child processes like vLLM workers might still be running)
            print("[Main] Cleaning up processes...")
            if server_proc:
                print(f"[Main] Killing server process tree (PID {server_proc.pid})...")
                kill_process_tree(server_proc.pid)
            if trainer_proc:
                print(f"[Main] Killing trainer process tree (PID {trainer_proc.pid})...")
                kill_process_tree(trainer_proc.pid)


# class TestConcurrentOperations:
#     """Test concurrent generation and weight updates."""

#     @pytest.mark.asyncio
#     @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
#     async def test_multiple_generations_before_update(
#         self,
#         vllm_engine_factory,
#         sample_prompts,
#         generation_config,
#     ):
#         """Test that multiple generation requests work correctly."""
#         from vllm import SamplingParams

#         async with vllm_engine_factory() as manager:
#             sampling_params = SamplingParams(
#                 temperature=generation_config["temperature"],
#                 top_p=generation_config["top_p"],
#                 max_tokens=generation_config["max_tokens"],
#                 seed=generation_config["seed"],
#             )

#             # Launch multiple generation requests
#             tasks = []
#             for i, prompt in enumerate(sample_prompts):
#                 async def generate_one(prompt, idx):
#                     request_id = f"concurrent_{idx}"
#                     async for output in manager.engine.generate(
#                         prompt,
#                         sampling_params=sampling_params,
#                         request_id=request_id,
#                     ):
#                         final = output
#                     return final.outputs[0].text

#                 tasks.append(generate_one(prompt, i))

#             # Run all generations concurrently
#             results = await asyncio.gather(*tasks)

#             assert len(results) == len(sample_prompts)
#             for i, result in enumerate(results):
#                 print(f"Result {i}: {result[:50]}...")
#                 assert len(result) > 0
