"""Integration tests for vllm1 with Fast-LLM weight broadcast protocol."""

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
    run_generation_loop,
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
            for line in iter(stream.readline, ""):
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


@pytest.fixture
def fast_llm_trainer_helper():
    """Path to Fast-LLM trainer helper script."""
    return Path(__file__).parent / "fast_llm_trainer_helper.py"


@pytest.fixture
def redis_server():
    """Start a Redis server for testing and stop it after the test.

    Returns:
        Tuple of (host, port) for the Redis server
    """
    import shutil
    import socket

    # Check if redis-server is available
    redis_server_bin = shutil.which("redis-server")
    if not redis_server_bin:
        pytest.skip("redis-server not found in PATH")

    # Find an available port
    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    redis_port = find_free_port()
    redis_host = "localhost"

    print(f"[Redis] Starting Redis server on {redis_host}:{redis_port}")

    # Start Redis server with minimal config
    redis_proc = subprocess.Popen(
        [
            redis_server_bin,
            "--port", str(redis_port),
            "--bind", redis_host,
            "--save", "",  # Disable persistence
            "--appendonly", "no",  # Disable AOF
            "--protected-mode", "no",  # Allow connections without password
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Start streaming Redis output
    redis_stdout_thread, redis_stderr_thread = stream_process_output(redis_proc, "Redis")

    # Wait for Redis to be ready
    import redis
    r = redis.Redis(host=redis_host, port=redis_port)
    for i in range(30):
        try:
            r.ping()
            print(f"[Redis] Server ready on {redis_host}:{redis_port}")
            break
        except redis.ConnectionError:
            if redis_proc.poll() is not None:
                raise RuntimeError(f"Redis server failed to start (exit code {redis_proc.returncode})")
            time.sleep(0.1)
    else:
        redis_proc.kill()
        raise TimeoutError("Redis server did not start within 3 seconds")

    try:
        yield (redis_host, redis_port)
    finally:
        # Cleanup
        print(f"[Redis] Stopping Redis server (PID {redis_proc.pid})")
        redis_proc.terminate()
        try:
            redis_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("[Redis] Redis did not stop gracefully, killing...")
            redis_proc.kill()
            redis_proc.wait()
        print("[Redis] Redis server stopped")


class TestFastLLMServerIntegration:
    """Test Fast-LLM weight broadcast with vLLM HTTP server."""

    @pytest.mark.timeout(2400)  # 40 minutes for server test
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs"
    )
    async def test_server_fast_llm_broadcast_pattern(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        fast_llm_trainer_helper,
        redis_server,
        temp_dir,
    ):
        """Server integration test: verify Fast-LLM weight broadcast pattern with HTTP API.

        This test validates the Fast-LLM protocol where:
        1. Redis server is automatically started
        2. vLLM server is running and serving HTTP requests
        3. Trainer broadcasts weight updates via Redis stream + broadcast_object_list
        4. Server responses change based on weight updates

        Flow:
        - Start Redis server (via fixture)
        - Start vLLM HTTP server with --weight-update-mode=fast-llm
        - Continuously generate via HTTP API (deterministic)
        - Trainer: wait 15s → broadcast perturbed → wait 5s → broadcast original → wait 5s → broadcast perturbed
        - Verify generation pattern: original → perturbed → original → perturbed
        - Stop Redis server (via fixture cleanup)
        """
        print("\n" + "=" * 60)
        print("Starting Fast-LLM server weight update pattern test")
        print("=" * 60)

        # Get Redis connection info from fixture
        redis_host, redis_port = redis_server
        print(f"[Main] Using Redis server at {redis_host}:{redis_port}")

        server_port = 8000
        server_url = f"http://127.0.0.1:{server_port}"

        # Start vLLM server with Fast-LLM mode
        server_proc, _, _ = start_vllm_server(
            model_name=model_name,
            server_port=server_port,
            distributed_init_method=distributed_init_method,
            stream_process_output_fn=stream_process_output,
            extra_args=[
                "--weight-update-mode", "fast-llm",
                "--redis-host", redis_host,
                "--redis-port", str(redis_port),
            ],
        )

        # Give server a moment to start
        await asyncio.sleep(1)

        # Start trainer process
        trainer_proc, _, _ = start_trainer_process(
            trainer_helper_path=fast_llm_trainer_helper,
            distributed_init_method=distributed_init_method,
            model_name=model_name,
            server_url=server_url,
            stream_process_output_fn=stream_process_output,
            extra_args=[
                "--redis-host", redis_host,
                "--redis-port", str(redis_port),
            ],
        )

        try:
            # Wait for server to be ready
            await wait_for_server_ready(server_url, server_proc, trainer_proc)

            # Run generation loop
            generations = await run_generation_loop(
                server_url=server_url,
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

            # Analyze and verify pattern
            analyze_and_verify_pattern(generations)
            print("\n✓ Fast-LLM server weight update pattern test PASSED")

        finally:
            # Cleanup - always kill process tree even if main process exited
            # (child processes like vLLM workers might still be running)
            print("[Main] Cleaning up processes...")
            if server_proc:
                print(
                    f"[Main] Killing server process tree (PID {server_proc.pid})..."
                )
                kill_process_tree(server_proc.pid)
            if trainer_proc:
                print(
                    f"[Main] Killing trainer process tree (PID {trainer_proc.pid})..."
                )
                kill_process_tree(trainer_proc.pid)
