"""Shared utilities for server weight update integration tests."""

import asyncio
import requests
import time
from pathlib import Path
import subprocess
import sys
import os


async def wait_for_server_ready(server_url: str, server_proc, trainer_proc, timeout_seconds: int = 300):
    """Wait for server to be ready by polling health endpoint.

    Args:
        server_url: Base URL of server (e.g., "http://127.0.0.1:8000")
        server_proc: Server subprocess
        trainer_proc: Trainer subprocess
        timeout_seconds: Maximum time to wait

    Returns:
        True if server is ready

    Raises:
        RuntimeError: If server or trainer process terminates
        TimeoutError: If server doesn't become ready within timeout
    """
    print("[Main] Waiting for server to be ready...")
    for i in range(timeout_seconds):
        # Check if server process crashed
        if server_proc.poll() is not None:
            print(f"[Main] Server process terminated with code {server_proc.returncode}")
            raise RuntimeError(f"Server process terminated with code {server_proc.returncode}")

        # Check if trainer process crashed
        if trainer_proc.poll() is not None:
            print(f"[Main] Trainer process terminated with code {trainer_proc.returncode}")
            raise RuntimeError(f"Trainer process terminated with code {trainer_proc.returncode}")

        try:
            resp = requests.get(f"{server_url}/health", timeout=1)
            if resp.status_code == 200:
                print("[Main] Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass

        if i % 10 == 0:
            print(f"[Main] Still waiting for server... ({i} seconds)")
        await asyncio.sleep(1)

    raise TimeoutError(f"Server did not become ready within {timeout_seconds} seconds")


def check_pattern_detected(generations):
    """Check if we have detected the full pattern (4 phases).

    Args:
        generations: List of (timestamp, text) tuples

    Returns:
        True if pattern is detected (4 phases with correct relationships)
    """
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


async def run_generation_loop(
    server_url: str,
    model_name: str,
    simple_prompt: str,
    generation_config: dict,
    trainer_proc,
    max_duration: int = 120,
    generation_interval: float = 0.5,
):
    """Run continuous generation loop until pattern is detected or timeout.

    Args:
        server_url: Base URL of server
        model_name: Model name for API request
        simple_prompt: Prompt to generate from
        generation_config: Config dict with max_tokens, etc.
        trainer_proc: Trainer subprocess to monitor
        max_duration: Maximum duration in seconds
        generation_interval: Time between generations

    Returns:
        List of (timestamp, generated_text) tuples
    """
    print("[Main] Starting continuous generation loop...")
    generations = []
    start_time = time.time()

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
                "top_p": 1.0,
                "seed": 42,
            }

            resp = requests.post(
                f"{server_url}/v1/completions",
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

    return generations


def analyze_and_verify_pattern(generations):
    """Analyze generation sequence and verify the expected pattern.

    Args:
        generations: List of (timestamp, text) tuples

    Raises:
        AssertionError: If pattern is not as expected
    """
    print("\n" + "=" * 60)
    print("GENERATION SEQUENCE ANALYSIS")
    print("=" * 60)
    print(f"Total generations: {len(generations)}")

    # Print all generations
    for i, (ts, text) in enumerate(generations):
        print(f"[{ts:5.1f}s] Gen {i+1}: '{text[:80]}...'")

    # Identify unique generation texts and their phases
    if len(generations) < 4:
        raise AssertionError(
            f"Not enough generations to verify pattern (need at least 4, got {len(generations)})"
        )

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

    print("\n" + "=" * 60)
    print(f"Detected {len(phases)} phases:")
    for i, (text, items) in enumerate(phases):
        print(f"Phase {i+1}: {len(items)} generations - '{text[:60]}...'")
    print("=" * 60)

    # Verify the pattern
    assert (
        len(phases) >= 4
    ), f"Expected at least 4 phases (original → perturbed → original → perturbed), got {len(phases)}"

    phase1_text, phase1_items = phases[0]
    phase2_text, phase2_items = phases[1]
    phase3_text, phase3_items = phases[2]
    phase4_text, phase4_items = phases[3]

    # Verify phase 1 (original) != phase 2 (perturbed)
    assert (
        phase1_text != phase2_text
    ), "Phase 1 (original) and Phase 2 (perturbed) should be different"

    # Verify phase 3 (original) == phase 1 (original)
    assert (
        phase3_text == phase1_text
    ), f"Phase 3 should match Phase 1 (original weights restored)"

    # Verify phase 4 (perturbed) == phase 2 (perturbed)
    assert (
        phase4_text == phase2_text
    ), f"Phase 4 should match Phase 2 (perturbed weights reapplied)"

    print("\n✓ Pattern verified:")
    print(f"  Phase 1 (original):   {len(phase1_items)} generations")
    print(f"  Phase 2 (perturbed):  {len(phase2_items)} generations")
    print(f"  Phase 3 (original):   {len(phase3_items)} generations (matches Phase 1 ✓)")
    print(f"  Phase 4 (perturbed):  {len(phase4_items)} generations (matches Phase 2 ✓)")


def start_vllm_server(
    model_name: str,
    server_port: int,
    distributed_init_method: str,
    stream_process_output_fn,
    extra_args: list = None,
):
    """Start vLLM HTTP server subprocess.

    Args:
        model_name: Model to load
        server_port: Port to bind to
        distributed_init_method: Distributed initialization method
        stream_process_output_fn: Function to stream process output
        extra_args: Additional CLI arguments (e.g., ["--weight-update-mode", "fast-llm"])

    Returns:
        Tuple of (server_proc, stdout_thread, stderr_thread)
    """
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = "0"
    vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    print(f"[Main] Starting vLLM HTTP server on port {server_port} (GPU 0)")
    vllm_entry_point = Path(__file__).parent.parent / "pipelinerl" / "entrypoints" / "run_vllm1.py"

    cmd = [
        sys.executable,
        str(vllm_entry_point),
        "--model", model_name,
        "--port", str(server_port),
        "--host", "127.0.0.1",
        "--actor-llm-idx", "0",
        "--weight-update-group-init-method", distributed_init_method,
        "--weight-update-group-world-size", "2",
    ]

    if extra_args:
        cmd.extend(extra_args)

    server_proc = subprocess.Popen(
        cmd,
        env=vllm_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("[Main] Starting server output streaming...")
    stdout_thread, stderr_thread = stream_process_output_fn(server_proc, "vLLM Server")

    return server_proc, stdout_thread, stderr_thread


def start_trainer_process(
    trainer_helper_path: Path,
    distributed_init_method: str,
    model_name: str,
    server_url: str,
    stream_process_output_fn,
    extra_args: list = None,
):
    """Start trainer subprocess.

    Args:
        trainer_helper_path: Path to trainer helper script
        distributed_init_method: Distributed initialization method
        model_name: Model name
        server_url: Server URL for health check
        stream_process_output_fn: Function to stream process output
        extra_args: Additional CLI arguments (e.g., ["--redis-host", "localhost"])

    Returns:
        Tuple of (trainer_proc, stdout_thread, stderr_thread)
    """
    trainer_env = os.environ.copy()
    trainer_env["CUDA_VISIBLE_DEVICES"] = "1"

    print("[Main] Starting trainer process (GPU 1) for process group rendezvous")

    cmd = [
        sys.executable,
        str(trainer_helper_path),
    ]

    # Check which trainer helper is being used by the script name
    if "fast_llm" in str(trainer_helper_path):
        # fast_llm_trainer_helper.py uses argparse with --init-method, --model, etc.
        cmd.extend([
            "--init-method", distributed_init_method,
            "--model", model_name,
            "--server-url", server_url,
        ])
    else:
        # distributed_trainer_helper.py uses positional args
        cmd.extend([
            "timed_broadcast_server_test",
            "--init-method", distributed_init_method,
            "--model-name", model_name,
            "--server-url", server_url,
        ])

    if extra_args:
        cmd.extend(extra_args)

    trainer_proc = subprocess.Popen(
        cmd,
        env=trainer_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("[Main] Starting trainer output streaming...")
    stdout_thread, stderr_thread = stream_process_output_fn(trainer_proc, "Trainer")

    return trainer_proc, stdout_thread, stderr_thread
