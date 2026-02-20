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

# Import shared utilities
from .server_weight_update_utils import (
    wait_for_server_ready,
    wait_for_all_servers_ready,
    run_generation_loop,
    run_generation_loop_multi,
    analyze_and_verify_pattern,
    start_vllm_server,
    start_trainer_process,
)

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


# ---------------------------------------------------------------------------
# Module-level helpers shared by all test variants
# ---------------------------------------------------------------------------

def _compare_actor_results(sync_dir: Path, num_actors: int):
    """Assert that all actors produced identical generation results.

    Each actor writes ``sync_dir/results_actor_{i}.json`` with keys
    res_or_1, res_mod_1, res_or_2, res_mod_2.
    """
    import json

    results = [
        json.loads((sync_dir / f"results_actor_{i}.json").read_text())
        for i in range(num_actors)
    ]
    for key in results[0]:
        texts = [r[key] for r in results]
        assert len(set(texts)) == 1, (
            f"Actors disagree on '{key}': {texts}"
        )


async def _run_back_and_forth_engine_test(
    model_name,
    simple_prompt,
    generation_config,
    init_method,
    distributed_trainer_helper,
    vllm_engine_helper,
    sync_dir,
    vllm_configs,
    trainer_gpu,
    world_size,
    timeout=1800,
):
    """Run back-and-forth engine test with one or more vLLM actor processes.

    Args:
        vllm_configs: List of dicts, each with keys:
            - cuda_devices: str, e.g. "0" or "0,1"
            - actor_llm_idx: int
            - tensor_parallel_size: int
        trainer_gpu: str, e.g. "1" or "2"
        world_size: total NCCL world size (all vLLM workers + trainer)
    """
    from .sync_helper import create_sync_dir

    num_actors = len(vllm_configs)
    all_procs = []

    # Start all vLLM actor subprocesses
    for cfg in vllm_configs:
        vllm_env = os.environ.copy()
        vllm_env["CUDA_VISIBLE_DEVICES"] = cfg["cuda_devices"]
        vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        vllm_env["PIPELINERL_DEBUG"] = "1"

        actor_idx = cfg["actor_llm_idx"]
        tp = cfg.get("tensor_parallel_size", 1)
        print(f"[Main] Starting vLLM actor {actor_idx} (GPU(s) {cfg['cuda_devices']}, TP={tp})")

        vllm_proc = subprocess.Popen(
            [
                sys.executable,
                str(vllm_engine_helper),
                "back_and_forth",
                "--model-name", model_name,
                "--init-method", init_method,
                "--actor-llm-idx", str(actor_idx),
                "--world-size", str(world_size),
                "--tensor-parallel-size", str(tp),
                "--prompt", simple_prompt,
                "--max-tokens", str(generation_config["max_tokens"]),
                "--sync-dir", str(sync_dir),
            ],
            env=vllm_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        all_procs.append((vllm_proc, f"vLLM Actor {actor_idx}"))

    await asyncio.sleep(1)

    # Start trainer subprocess
    trainer_env = os.environ.copy()
    trainer_env["CUDA_VISIBLE_DEVICES"] = trainer_gpu
    trainer_env["PIPELINERL_DEBUG"] = "1"

    print(f"[Main] Starting trainer (GPU {trainer_gpu}, {num_actors} actor(s), world_size={world_size})")
    trainer_proc = subprocess.Popen(
        [
            sys.executable,
            str(distributed_trainer_helper),
            "back_and_forth",
            "--init-method", init_method,
            "--model-name", model_name,
            "--sync-dir", str(sync_dir),
            "--num-actors", str(num_actors),
            "--world-size", str(world_size),
        ],
        env=trainer_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    all_procs.append((trainer_proc, "Trainer"))

    await wait_for_processes(all_procs, timeout=timeout)

    # Verify all actors produced the same results
    _compare_actor_results(sync_dir, num_actors)
    print(f"\n✓ Back-and-forth test PASSED ({num_actors} actor(s), world_size={world_size})")


async def _run_server_weight_update_test(
    model_name,
    simple_prompt,
    generation_config,
    init_method,
    distributed_trainer_helper,
    vllm_server_configs,
    trainer_gpu,
    world_size,
    timeout=2400,
):
    """Run server weight-update pattern test with one or more vLLM servers.

    Args:
        vllm_server_configs: List of dicts, each with keys:
            - port: int
            - gpu_ids: str
            - actor_llm_idx: int
            - tensor_parallel_size: int
        trainer_gpu: str, e.g. "1" or "2"
        world_size: total NCCL world size
    """
    server_procs = []
    server_urls = []

    for cfg in vllm_server_configs:
        port = cfg["port"]
        url = f"http://127.0.0.1:{port}"
        server_urls.append(url)

        server_proc, _, _ = start_vllm_server(
            model_name=model_name,
            server_port=port,
            distributed_init_method=init_method,
            stream_process_output_fn=stream_process_output,
            extra_args=None,
            gpu_ids=cfg.get("gpu_ids", "0"),
            actor_llm_idx=cfg.get("actor_llm_idx", 0),
            world_size=world_size,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        )
        server_procs.append(server_proc)

    await asyncio.sleep(1)

    trainer_proc, _, _ = start_trainer_process(
        trainer_helper_path=distributed_trainer_helper,
        distributed_init_method=init_method,
        model_name=model_name,
        server_urls=server_urls,
        stream_process_output_fn=stream_process_output,
        extra_args=None,
        gpu_id=trainer_gpu,
        world_size=world_size,
    )

    try:
        await wait_for_all_servers_ready(server_urls, server_procs, trainer_proc)

        if len(server_urls) == 1:
            generations = await run_generation_loop(
                server_url=server_urls[0],
                model_name=model_name,
                simple_prompt=simple_prompt,
                generation_config=generation_config,
                trainer_proc=trainer_proc,
            )
        else:
            generations = await run_generation_loop_multi(
                server_urls=server_urls,
                model_name=model_name,
                simple_prompt=simple_prompt,
                generation_config=generation_config,
                trainer_proc=trainer_proc,
            )

        # Wait for trainer to finish
        print("[Main] Waiting for trainer to finish...")
        for _ in range(30):
            if trainer_proc.poll() is not None:
                break
            await asyncio.sleep(1)

        analyze_and_verify_pattern(generations)
        print(f"\n✓ Server weight update pattern test PASSED ({len(server_urls)} server(s))")

    finally:
        print("[Main] Cleaning up processes...")
        for proc in server_procs:
            if proc:
                kill_process_tree(proc.pid)
        if trainer_proc:
            kill_process_tree(trainer_proc.pid)


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

        Validates that we can update weights multiple times and the results
        are deterministic and reproducible.
        """
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting back-and-forth test (TP=1, 1 actor, 2 GPUs)")
        print("="*60)

        sync_dir = create_sync_dir(shared_test_dir)
        await _run_back_and_forth_engine_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=shared_distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_engine_helper=vllm_engine_helper,
            sync_dir=sync_dir,
            vllm_configs=[{"cuda_devices": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1}],
            trainer_gpu="1",
            world_size=2,
            timeout=1800,
        )

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

        Validates the real-world scenario where a vLLM HTTP server receives
        weight updates from a trainer while serving requests.
        """
        print("\n" + "="*60)
        print("Starting server weight update pattern test (TP=1, 1 actor, 2 GPUs)")
        print("="*60)

        await _run_server_weight_update_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_server_configs=[{"port": 8000, "gpu_ids": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1}],
            trainer_gpu="1",
            world_size=2,
            timeout=2400,
        )


class TestWeightUpdateTP2:
    """Test weight updates with tensor-parallel (TP=2) vLLM — needs 3 GPUs."""

    @pytest.mark.timeout(2000)
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 3, reason="Requires at least 3 GPUs")
    async def test_weight_update_back_and_forth_tp2(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        temp_dir,
    ):
        """Back-and-forth test with TP=2: one vLLM instance on GPUs 0+1, trainer on GPU 2."""
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting back-and-forth test (TP=2, 1 actor, 3 GPUs)")
        print("="*60)

        sync_dir = create_sync_dir(temp_dir)
        await _run_back_and_forth_engine_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_engine_helper=vllm_engine_helper,
            sync_dir=sync_dir,
            vllm_configs=[{"cuda_devices": "0,1", "actor_llm_idx": 0, "tensor_parallel_size": 2}],
            trainer_gpu="2",
            world_size=3,
            timeout=1800,
        )

    @pytest.mark.timeout(2400)
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 3, reason="Requires at least 3 GPUs")
    async def test_server_weight_update_pattern_tp2(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        temp_dir,
    ):
        """Server weight update test with TP=2: one server on GPUs 0+1, trainer on GPU 2."""
        print("\n" + "="*60)
        print("Starting server weight update pattern test (TP=2, 1 actor, 3 GPUs)")
        print("="*60)

        await _run_server_weight_update_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_server_configs=[{"port": 8001, "gpu_ids": "0,1", "actor_llm_idx": 0, "tensor_parallel_size": 2}],
            trainer_gpu="2",
            world_size=3,
            timeout=2400,
        )


class TestWeightUpdateMultiActor:
    """Test weight updates with multiple independent vLLM actors."""

    @pytest.mark.timeout(2000)
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 3, reason="Requires at least 3 GPUs")
    async def test_weight_update_back_and_forth_2actors(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        temp_dir,
    ):
        """Back-and-forth test with 2 actors: vLLM on GPU 0 and GPU 1, trainer on GPU 2."""
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting back-and-forth test (TP=1, 2 actors, 3 GPUs)")
        print("="*60)

        sync_dir = create_sync_dir(temp_dir)
        await _run_back_and_forth_engine_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_engine_helper=vllm_engine_helper,
            sync_dir=sync_dir,
            vllm_configs=[
                {"cuda_devices": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1},
                {"cuda_devices": "1", "actor_llm_idx": 1, "tensor_parallel_size": 1},
            ],
            trainer_gpu="2",
            world_size=3,
            timeout=1800,
        )

    @pytest.mark.timeout(2000)
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
    async def test_weight_update_back_and_forth_3actors(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        vllm_engine_helper,
        temp_dir,
    ):
        """Back-and-forth test with 3 actors: vLLM on GPUs 0/1/2, trainer on GPU 3."""
        from .sync_helper import create_sync_dir

        print("\n" + "="*60)
        print("Starting back-and-forth test (TP=1, 3 actors, 4 GPUs)")
        print("="*60)

        sync_dir = create_sync_dir(temp_dir)
        await _run_back_and_forth_engine_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_engine_helper=vllm_engine_helper,
            sync_dir=sync_dir,
            vllm_configs=[
                {"cuda_devices": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1},
                {"cuda_devices": "1", "actor_llm_idx": 1, "tensor_parallel_size": 1},
                {"cuda_devices": "2", "actor_llm_idx": 2, "tensor_parallel_size": 1},
            ],
            trainer_gpu="3",
            world_size=4,
            timeout=1800,
        )

    @pytest.mark.timeout(2400)
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 3, reason="Requires at least 3 GPUs")
    async def test_server_weight_update_pattern_2actors(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        temp_dir,
    ):
        """Server weight update test with 2 actors: servers on GPUs 0 and 1, trainer on GPU 2."""
        print("\n" + "="*60)
        print("Starting server weight update pattern test (TP=1, 2 actors, 3 GPUs)")
        print("="*60)

        await _run_server_weight_update_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_server_configs=[
                {"port": 8000, "gpu_ids": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1},
                {"port": 8001, "gpu_ids": "1", "actor_llm_idx": 1, "tensor_parallel_size": 1},
            ],
            trainer_gpu="2",
            world_size=3,
            timeout=2400,
        )

    @pytest.mark.timeout(2400)
    @pytest.mark.asyncio
    @pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
    async def test_server_weight_update_pattern_3actors(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        distributed_trainer_helper,
        temp_dir,
    ):
        """Server weight update test with 3 actors: servers on GPUs 0/1/2, trainer on GPU 3."""
        print("\n" + "="*60)
        print("Starting server weight update pattern test (TP=1, 3 actors, 4 GPUs)")
        print("="*60)

        await _run_server_weight_update_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            distributed_trainer_helper=distributed_trainer_helper,
            vllm_server_configs=[
                {"port": 8000, "gpu_ids": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1},
                {"port": 8001, "gpu_ids": "1", "actor_llm_idx": 1, "tensor_parallel_size": 1},
                {"port": 8002, "gpu_ids": "2", "actor_llm_idx": 2, "tensor_parallel_size": 1},
            ],
            trainer_gpu="3",
            world_size=4,
            timeout=2400,
        )


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
