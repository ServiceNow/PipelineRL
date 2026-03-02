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
    wait_for_all_servers_ready,
    run_generation_loop,
    run_generation_loop_multi,
    analyze_and_verify_pattern,
    analyze_and_verify_pattern_multi,
    analyze_and_verify_transitions,
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


# ---------------------------------------------------------------------------
# Module-level helper shared by all Fast-LLM test variants
# ---------------------------------------------------------------------------

async def _run_fast_llm_server_test(
    model_name,
    simple_prompt,
    generation_config,
    init_method,
    fast_llm_trainer_helper,
    redis_host,
    redis_port,
    vllm_server_configs,
    trainer_gpu,
    world_size,
    timeout=2400,
):
    """Run Fast-LLM server weight-update pattern test with one or more vLLM servers.

    Args:
        vllm_server_configs: List of dicts, each with keys:
            - port: int
            - gpu_ids: str
            - actor_llm_idx: int
            - tensor_parallel_size: int
        trainer_gpu: str, e.g. "1" or "2"
        world_size: total NCCL world size (trainer + all vLLM workers)
        redis_host: Redis host address
        redis_port: Redis port number
    """
    server_procs = []
    server_urls = []

    fast_llm_server_args = [
        "--weight-update-mode", "fast-llm",
        "--redis-host", redis_host,
        "--redis-port", str(redis_port),
    ]

    for cfg in vllm_server_configs:
        port = cfg["port"]
        url = f"http://127.0.0.1:{port}"
        server_urls.append(url)

        server_proc, _, _ = start_vllm_server(
            model_name=model_name,
            server_port=port,
            distributed_init_method=init_method,
            stream_process_output_fn=stream_process_output,
            extra_args=fast_llm_server_args,
            gpu_ids=cfg.get("gpu_ids", "0"),
            actor_llm_idx=cfg.get("actor_llm_idx", 0),
            world_size=world_size,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
        )
        server_procs.append(server_proc)

    await asyncio.sleep(1)

    trainer_proc, _, _ = start_trainer_process(
        trainer_helper_path=fast_llm_trainer_helper,
        distributed_init_method=init_method,
        model_name=model_name,
        server_urls=server_urls,
        stream_process_output_fn=stream_process_output,
        extra_args=[
            "--redis-host", redis_host,
            "--redis-port", str(redis_port),
        ],
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
            per_server_generations = await run_generation_loop_multi(
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

        if len(server_urls) == 1:
            analyze_and_verify_pattern(generations)
        else:
            analyze_and_verify_pattern_multi(per_server_generations)
        print(f"\n✓ Fast-LLM server weight update pattern test PASSED ({len(server_urls)} server(s))")

    finally:
        print("[Main] Cleaning up processes...")
        for proc in server_procs:
            if proc:
                kill_process_tree(proc.pid)
        if trainer_proc:
            kill_process_tree(trainer_proc.pid)


class TestFastLLMServerIntegration:
    """Test Fast-LLM weight broadcast with vLLM HTTP server — 2 GPUs (baseline)."""

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

        Validates the Fast-LLM protocol where:
        - Redis server signals weight updates
        - vLLM server receives weights via broadcast_object_list + broadcast
        - Server responses change as expected (original → perturbed → original → perturbed)

        Topology: 1 vLLM server on GPU 0, trainer on GPU 1 (world_size=2).
        """
        print("\n" + "=" * 60)
        print("Starting Fast-LLM server weight update pattern test (TP=1, 1 actor, 2 GPUs)")
        print("=" * 60)

        redis_host, redis_port = redis_server

        await _run_fast_llm_server_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            fast_llm_trainer_helper=fast_llm_trainer_helper,
            redis_host=redis_host,
            redis_port=redis_port,
            vllm_server_configs=[{"port": 8000, "gpu_ids": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1}],
            trainer_gpu="1",
            world_size=2,
            timeout=2400,
        )

    @pytest.mark.timeout(2400)
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs"
    )
    async def test_fast_llm_server_catch_transitions(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        fast_llm_trainer_helper,
        redis_server,
        temp_dir,
    ):
        """Diagnostic test: catch garbage generations during Fast-LLM weight broadcasts.

        The trainer runs a slow initial cycle (perturbed → original, 5 s each)
        to firmly establish text_A and text_B, then fires N rapid back-to-back
        broadcast cycles (perturbed → original) with no inter-broadcast delay.
        The generation loop runs with generation_interval=0.0 to maximise the
        chance of hitting a mid-broadcast state.

        Assertions:
          1. The A→B→A→B pattern is still detected (broadcasts actually worked).
          2. At least one transition/garbage phase was captured.

        Topology: 1 vLLM server on GPU 0, trainer on GPU 1 (world_size=2).
        """
        print("\n" + "=" * 60)
        print("Starting Fast-LLM transition-capture test (TP=1, 1 actor, 2 GPUs)")
        print("=" * 60)

        redis_host, redis_port = redis_server
        server_url = "http://127.0.0.1:8000"

        server_proc, _, _ = start_vllm_server(
            model_name=model_name,
            server_port=8000,
            distributed_init_method=distributed_init_method,
            stream_process_output_fn=stream_process_output,
            extra_args=[
                "--weight-update-mode", "fast-llm",
                "--redis-host", redis_host,
                "--redis-port", str(redis_port),
            ],
            gpu_ids="0",
            actor_llm_idx=0,
            world_size=2,
            tensor_parallel_size=1,
        )

        await asyncio.sleep(1)

        trainer_proc, _, _ = start_trainer_process(
            trainer_helper_path=fast_llm_trainer_helper,
            distributed_init_method=distributed_init_method,
            model_name=model_name,
            server_urls=[server_url],
            stream_process_output_fn=stream_process_output,
            extra_args=[
                "--redis-host", redis_host,
                "--redis-port", str(redis_port),
                "--n-cycles", "6",
            ],
            gpu_id="1",
            world_size=2,
        )

        try:
            await wait_for_server_ready(server_url, server_proc, trainer_proc)

            generations = await run_generation_loop(
                server_url=server_url,
                model_name=model_name,
                simple_prompt=simple_prompt,
                generation_config=generation_config,
                trainer_proc=trainer_proc,
                generation_interval=0.0,
            )

            print("[Main] Waiting for trainer to finish...")
            for _ in range(30):
                if trainer_proc.poll() is not None:
                    break
                await asyncio.sleep(1)

            analyze_and_verify_transitions(generations, n_cycles=6)
            print("\n✓ Fast-LLM transition-capture test PASSED")

        finally:
            print("[Main] Cleaning up processes...")
            if server_proc:
                kill_process_tree(server_proc.pid)
            if trainer_proc:
                kill_process_tree(trainer_proc.pid)


class TestFastLLMServerTP2:
    """Test Fast-LLM weight broadcast with tensor-parallel (TP=2) — needs 3 GPUs."""

    @pytest.mark.timeout(2400)
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        torch.cuda.device_count() < 3, reason="Requires at least 3 GPUs"
    )
    async def test_server_fast_llm_broadcast_pattern_tp2(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        fast_llm_trainer_helper,
        redis_server,
        temp_dir,
    ):
        """Fast-LLM server test with TP=2: one server on GPUs 0+1, trainer on GPU 2.

        Verifies that tensor-parallel vLLM correctly receives Fast-LLM weight
        updates when multiple GPU workers share the same NCCL process group.
        """
        print("\n" + "=" * 60)
        print("Starting Fast-LLM server weight update pattern test (TP=2, 1 actor, 3 GPUs)")
        print("=" * 60)

        redis_host, redis_port = redis_server

        await _run_fast_llm_server_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            fast_llm_trainer_helper=fast_llm_trainer_helper,
            redis_host=redis_host,
            redis_port=redis_port,
            vllm_server_configs=[{"port": 8001, "gpu_ids": "0,1", "actor_llm_idx": 0, "tensor_parallel_size": 2}],
            trainer_gpu="2",
            world_size=3,
            timeout=2400,
        )


class TestFastLLMServerMultiActor:
    """Test Fast-LLM weight broadcast with multiple independent vLLM actors."""

    @pytest.mark.timeout(2400)
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        torch.cuda.device_count() < 3, reason="Requires at least 3 GPUs"
    )
    async def test_server_fast_llm_broadcast_pattern_2actors(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        fast_llm_trainer_helper,
        redis_server,
        temp_dir,
    ):
        """Fast-LLM server test with 2 actors: servers on GPUs 0 and 1, trainer on GPU 2.

        Verifies that two separate vLLM servers simultaneously receive the same
        Fast-LLM weight broadcast and produce identical generation results.
        """
        print("\n" + "=" * 60)
        print("Starting Fast-LLM server weight update pattern test (TP=1, 2 actors, 3 GPUs)")
        print("=" * 60)

        redis_host, redis_port = redis_server

        await _run_fast_llm_server_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            fast_llm_trainer_helper=fast_llm_trainer_helper,
            redis_host=redis_host,
            redis_port=redis_port,
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
    @pytest.mark.skipif(
        torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs"
    )
    async def test_server_fast_llm_broadcast_pattern_3actors(
        self,
        model_name,
        simple_prompt,
        generation_config,
        distributed_init_method,
        fast_llm_trainer_helper,
        redis_server,
        temp_dir,
    ):
        """Fast-LLM server test with 3 actors: servers on GPUs 0/1/2, trainer on GPU 3.

        Verifies that three separate vLLM servers simultaneously receive the same
        Fast-LLM weight broadcast and produce identical generation results.
        """
        print("\n" + "=" * 60)
        print("Starting Fast-LLM server weight update pattern test (TP=1, 3 actors, 4 GPUs)")
        print("=" * 60)

        redis_host, redis_port = redis_server

        await _run_fast_llm_server_test(
            model_name=model_name,
            simple_prompt=simple_prompt,
            generation_config=generation_config,
            init_method=distributed_init_method,
            fast_llm_trainer_helper=fast_llm_trainer_helper,
            redis_host=redis_host,
            redis_port=redis_port,
            vllm_server_configs=[
                {"port": 8000, "gpu_ids": "0", "actor_llm_idx": 0, "tensor_parallel_size": 1},
                {"port": 8001, "gpu_ids": "1", "actor_llm_idx": 1, "tensor_parallel_size": 1},
                {"port": 8002, "gpu_ids": "2", "actor_llm_idx": 2, "tensor_parallel_size": 1},
            ],
            trainer_gpu="3",
            world_size=4,
            timeout=2400,
        )
