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


def _build_phases(generations):
    """Collapse a generation list into (text, items) phase tuples."""
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
    if current_phase:
        phases.append((current_text, current_phase))
    return phases


def _identify_stable_texts(phases, min_stable_gens=5):
    """Return (text_a, text_b) identified from the first two stable phases.

    Iterates phases in order, skipping any with fewer than ``min_stable_gens``
    generations (transition artifacts).  The first stable phase gives text_A;
    the first stable phase with a different text gives text_B.

    Returns (text_a, text_b) or (None, None) if two distinct stable texts
    cannot be found.
    """
    text_a = None
    text_b = None
    for text, items in phases:
        if len(items) < min_stable_gens:
            continue
        if text_a is None:
            text_a = text
        elif text != text_a:
            text_b = text
            break
    if text_a is None or text_b is None:
        return None, None
    return text_a, text_b


def _find_abab_pattern(phases, min_stable_gens=5):
    """Search for the A→B→A→B pattern.

    text_A and text_B are identified from the first two *stable* phases —
    phases with at least ``min_stable_gens`` generations.  Short transition
    phases (1–few gens) produced while an NCCL broadcast is in-flight are
    automatically skipped during identification.

    The test is designed so that the server always starts with a long run of
    text_A (original weights, typically hundreds of gens) followed by a long
    run of text_B (first perturbed broadcast, tens of gens), making them
    unambiguous even with transition artifacts in between.

    After identifying text_A and text_B the full A→B→A→B subsequence is
    located in the phase list (transition phases between the four anchors are
    silently skipped).

    Returns (phase_a, phase_b, phase_a2, phase_b2) or None.
    """
    if len(phases) < 4:
        return None

    text_a, text_b = _identify_stable_texts(phases, min_stable_gens)

    if text_a is None or text_b is None:
        return None

    texts = [t for t, _ in phases]

    # Find ABAB as a subsequence in the phase list
    first_a = next((i for i, t in enumerate(texts) if t == text_a), None)
    if first_a is None:
        return None

    first_b = next((i for i in range(first_a + 1, len(phases)) if texts[i] == text_b), None)
    if first_b is None:
        return None

    second_a = next((i for i in range(first_b + 1, len(phases)) if texts[i] == text_a), None)
    if second_a is None:
        return None

    second_b = next((i for i in range(second_a + 1, len(phases)) if texts[i] == text_b), None)
    if second_b is None:
        return None

    return phases[first_a], phases[first_b], phases[second_a], phases[second_b]


def check_pattern_detected(generations):
    """Check whether the full A→B→A→B pattern is present in the generation history.

    This is a **post-hoc analysis helper** (e.g. for assertions after the
    generation loop ends).  It is intentionally *not* used as an early-stop
    signal inside the generation loops.

    Why not early-stop? Any transition artifact text T that happens to appear
    with several consecutive identical generations (possible when NCCL broadcasts
    are slow) is indistinguishable from the real perturbed text B at generation
    time.  False positives would cut the loop short before the final stable B
    phase accumulates.  The generation loops instead rely on the trainer process
    exiting (``trainer_proc.poll() is not None``) as their sole reliable
    termination signal — the trainer exits within milliseconds of completing its
    last broadcast, so no significant extra generation happens.

    Args:
        generations: List of (timestamp, text) tuples

    Returns:
        True if the A→B→A→B pattern is present
    """
    if len(generations) < 4:
        return False
    phases = _build_phases(generations)
    return _find_abab_pattern(phases) is not None


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
            else:
                print(f"[Main] Generation failed with status {resp.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"[Main] Request failed: {e}")

        await asyncio.sleep(generation_interval)

    return generations


def analyze_and_verify_pattern(generations):
    """Analyze generation sequence and verify the expected A→B→A→B pattern.

    Tolerates transition-artifact phases (e.g. a single generation produced
    while an NCCL broadcast was in-flight) by searching for the pattern as
    a subsequence rather than requiring it at exactly positions [0,1,2,3].

    Args:
        generations: List of (timestamp, text) tuples

    Returns:
        Tuple of (text_a, text_b) — the original and perturbed texts.

    Raises:
        AssertionError: If pattern is not as expected
    """
    print("\n" + "=" * 60)
    print("GENERATION SEQUENCE ANALYSIS")
    print("=" * 60)
    print(f"Total generations: {len(generations)}")

    for i, (ts, text) in enumerate(generations):
        print(f"[{ts:5.1f}s] Gen {i+1}: '{text[:80]}...'")

    assert len(generations) >= 4, (
        f"Not enough generations to verify pattern (need at least 4, got {len(generations)})"
    )

    phases = _build_phases(generations)

    _GRAY = "\033[90m"
    _RESET = "\033[0m"
    stable_a, stable_b = _identify_stable_texts(phases)
    stable_texts = {t for t in (stable_a, stable_b) if t is not None}
    print("\n" + "=" * 60)
    print(f"Detected {len(phases)} phase(s):")
    for i, (text, items) in enumerate(phases):
        line = f"Phase {i+1}: {len(items)} generation(s) - '{text[:60]}...'"
        if text not in stable_texts:
            print(f"{_GRAY}{line}  ← transition{_RESET}")
        else:
            print(line)
    print("=" * 60)

    result = _find_abab_pattern(phases)
    assert result is not None, (
        f"Could not find A→B→A→B pattern in {len(phases)} phase(s). "
        f"Phases: {[(text[:40], len(items)) for text, items in phases]}"
    )

    (phase_a_text, phase_a_items), (phase_b_text, phase_b_items), \
    (phase_a2_text, phase_a2_items), (phase_b2_text, phase_b2_items) = result

    # These hold by construction from _find_abab_pattern, but assert for clarity
    assert phase_a_text != phase_b_text, "Phase A and Phase B should be different"
    assert phase_a2_text == phase_a_text, "Second A should match first A (original weights restored)"
    assert phase_b2_text == phase_b_text, "Second B should match first B (perturbed weights reapplied)"

    skipped = len(phases) - 4
    skip_note = f" ({skipped} transition phase(s) skipped)" if skipped else ""
    print(f"\n✓ Pattern verified{skip_note}:")
    print(f"  Phase A  (original):   {len(phase_a_items)} generation(s)")
    print(f"  Phase B  (perturbed):  {len(phase_b_items)} generation(s)")
    print(f"  Phase A2 (original):   {len(phase_a2_items)} generation(s)  ← matches A ✓")
    print(f"  Phase B2 (perturbed):  {len(phase_b2_items)} generation(s)  ← matches B ✓")

    return phase_a_text, phase_b_text


def analyze_and_verify_pattern_multi(per_server_generations):
    """Verify A→B→A→B pattern independently per server, then check consistency.

    Each server's generation history is checked independently (since weight
    updates are not coordinated with requests, servers can transiently disagree).
    After all pass, we assert that every server converged on the same text A
    and text B.

    Args:
        per_server_generations: List of per-server generation lists, each a
            list of (timestamp, text) tuples (as returned by
            run_generation_loop_multi).

    Raises:
        AssertionError: If any server fails its pattern check or servers
            disagree on text A / text B.
    """
    patterns = []
    for i, generations in enumerate(per_server_generations):
        print(f"\n{'=' * 60}")
        print(f"Actor {i} pattern analysis")
        text_a, text_b = analyze_and_verify_pattern(generations)
        patterns.append((text_a, text_b))

    unique_a = set(t_a for t_a, _ in patterns)
    unique_b = set(t_b for _, t_b in patterns)
    assert len(unique_a) == 1, (
        f"Servers disagree on text A (original weights): "
        f"{[t_a[:40] for t_a, _ in patterns]}"
    )
    assert len(unique_b) == 1, (
        f"Servers disagree on text B (perturbed weights): "
        f"{[t_b[:40] for _, t_b in patterns]}"
    )
    print(f"\n✓ All {len(patterns)} actor(s) agree on text A and text B")


def extract_transition_phases(generations, text_a, text_b):
    """Return phases that are neither text_a nor text_b.

    These are mid-broadcast 'garbage' generations produced while an NCCL
    weight update was in flight and the model had partially updated weights.

    Args:
        generations: List of (timestamp, text) tuples
        text_a: The original-weights text (established first)
        text_b: The perturbed-weights text

    Returns:
        List of (text, items) phase tuples where text is neither text_a nor text_b
    """
    phases = _build_phases(generations)
    return [(text, items) for text, items in phases if text != text_a and text != text_b]


def analyze_and_verify_transitions(generations, n_cycles):
    """Verify A→B→A→B pattern and assert that transition generations were caught.

    The ``rapid_broadcast_cycles`` trainer command performs:
      - 1 startup A phase (server starts on original weights)
      - 1 slow perturbed broadcast  → text_B
      - 1 slow original broadcast   → text_A
      - n_cycles rapid pairs        → text_B, text_A each cycle
      - 1 final slow perturbed      → text_B

    This gives exactly ``4 + 2 * n_cycles`` stable phases.  Seeing fewer
    means a broadcast was missed entirely (timing/sync bug).

    Args:
        generations: List of (timestamp, text) tuples
        n_cycles: Number of rapid broadcast pairs (passed as ``--n-cycles``
            to the trainer helper).

    Raises:
        AssertionError: If ABAB pattern is not found, stable phase count is
            wrong, or no transition generations were caught.
    """
    text_a, text_b = analyze_and_verify_pattern(generations)

    phases = _build_phases(generations)
    stable_phases = [(text, items) for text, items in phases if text == text_a or text == text_b]
    expected_stable = 4 + 2 * n_cycles
    assert len(stable_phases) == expected_stable, (
        f"Expected {expected_stable} stable phases (4 + 2×{n_cycles} cycles) "
        f"but found {len(stable_phases)}. "
        f"A broadcast may have been missed or merged. "
        f"Stable phase counts: {[len(items) for _, items in stable_phases]}"
    )
    print(f"\n✓ Stable phase count correct: {len(stable_phases)} (expected {expected_stable})")

    transition_phases = extract_transition_phases(generations, text_a, text_b)

    print("\n" + "=" * 60)
    print(f"TRANSITION / GARBAGE GENERATIONS: {len(transition_phases)} phase(s)")
    print("=" * 60)
    if transition_phases:
        for i, (text, items) in enumerate(transition_phases):
            ts_start = items[0][0]
            ts_end = items[-1][0]
            print(f"  [{i + 1}] {len(items)} gen(s) @ {ts_start:.2f}s–{ts_end:.2f}s: '{text[:120]}'")
    else:
        print("  (none)")

    assert len(transition_phases) > 0, (
        "No transition generations were caught. "
        "Try increasing --n-cycles or verify generation_interval=0.0 is set."
    )
    print(f"\n✓ Caught {len(transition_phases)} transition phase(s)")


def start_vllm_server(
    model_name: str,
    server_port: int,
    distributed_init_method: str,
    stream_process_output_fn,
    extra_args: list = None,
    gpu_ids: str = "0",
    actor_llm_idx: int = 0,
    world_size: int = 2,
    tensor_parallel_size: int = 1,
):
    """Start vLLM HTTP server subprocess.

    Args:
        model_name: Model to load
        server_port: Port to bind to
        distributed_init_method: Distributed initialization method
        stream_process_output_fn: Function to stream process output
        extra_args: Additional CLI arguments (e.g., ["--weight-update-mode", "fast-llm"])
        gpu_ids: CUDA_VISIBLE_DEVICES value (e.g., "0" or "0,1")
        actor_llm_idx: Actor index for this vLLM instance
        world_size: Total distributed world size
        tensor_parallel_size: Number of GPUs for tensor parallelism

    Returns:
        Tuple of (server_proc, stdout_thread, stderr_thread)
    """
    vllm_env = os.environ.copy()
    vllm_env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    vllm_env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    print(f"[Main] Starting vLLM HTTP server on port {server_port} (GPU(s) {gpu_ids}, actor_idx={actor_llm_idx}, TP={tensor_parallel_size})")
    vllm_entry_point = Path(__file__).parent.parent / "pipelinerl" / "entrypoints" / "run_vllm1.py"

    cmd = [
        sys.executable,
        str(vllm_entry_point),
        "--model", model_name,
        "--port", str(server_port),
        "--host", "127.0.0.1",
        "--actor-llm-idx", str(actor_llm_idx),
        "--weight-update-group-init-method", distributed_init_method,
        "--weight-update-group-world-size", str(world_size),
        "--tensor-parallel-size", str(tensor_parallel_size),
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
    stdout_thread, stderr_thread = stream_process_output_fn(server_proc, f"vLLM Server (actor {actor_llm_idx})")

    return server_proc, stdout_thread, stderr_thread


async def wait_for_all_servers_ready(
    server_urls: list,
    server_procs: list,
    trainer_proc,
    timeout_seconds: int = 300,
):
    """Wait for all servers to be ready by polling their health endpoints.

    Args:
        server_urls: List of server base URLs
        server_procs: List of server subprocesses (same order as server_urls)
        trainer_proc: Trainer subprocess
        timeout_seconds: Maximum time to wait per server

    Returns:
        True if all servers are ready

    Raises:
        RuntimeError: If any process terminates unexpectedly
        TimeoutError: If any server doesn't become ready within timeout
    """
    for url, proc in zip(server_urls, server_procs):
        await wait_for_server_ready(url, proc, trainer_proc, timeout_seconds)
    return True


async def run_generation_loop_multi(
    server_urls: list,
    model_name: str,
    simple_prompt: str,
    generation_config: dict,
    trainer_proc,
    max_duration: int = 120,
    generation_interval: float = 0.5,
):
    """Run continuous generation loop querying all servers each round.

    Each server is tracked independently because weight updates and requests
    are not coordinated — different actors can temporarily return different
    results while a broadcast is in flight.  Pattern checking is therefore
    done per-server after the loop (see analyze_and_verify_pattern_multi).

    Args:
        server_urls: List of server base URLs
        model_name: Model name for API request
        simple_prompt: Prompt to generate from
        generation_config: Config dict with max_tokens, etc.
        trainer_proc: Trainer subprocess to monitor
        max_duration: Maximum duration in seconds
        generation_interval: Time between generation rounds

    Returns:
        List of per-server generation lists, each a list of
        (timestamp, generated_text) tuples (same order as server_urls).
    """
    print(f"[Main] Starting continuous generation loop across {len(server_urls)} server(s)...")
    per_server = [[] for _ in server_urls]
    start_time = time.time()

    payload = {
        "model": model_name,
        "prompt": simple_prompt,
        "max_tokens": generation_config["max_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
    }

    while time.time() - start_time < max_duration:
        # Check if trainer is still running
        trainer_poll = trainer_proc.poll()
        if trainer_poll is not None:
            print(f"[Main] Trainer exited with code {trainer_poll}")
            break

        for i, url in enumerate(server_urls):
            try:
                resp = requests.post(
                    f"{url}/v1/completions",
                    json=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["text"]
                    timestamp = time.time() - start_time
                    per_server[i].append((timestamp, text))
                    print(f"[Main] [{timestamp:.1f}s] Actor {i}: '{text}'")
                else:
                    print(f"[Main] Generation from actor {i} ({url}) failed with status {resp.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"[Main] Request to actor {i} ({url}) failed: {e}")

        await asyncio.sleep(generation_interval)

    return per_server


def start_trainer_process(
    trainer_helper_path: Path,
    distributed_init_method: str,
    model_name: str,
    server_urls: list,
    stream_process_output_fn,
    extra_args: list = None,
    gpu_id: str = "1",
    world_size: int = 2,
    command: str = "timed_broadcast_server_test",
):
    """Start trainer subprocess.

    Args:
        trainer_helper_path: Path to trainer helper script
        distributed_init_method: Distributed initialization method
        model_name: Model name
        server_urls: List of server URLs (one per actor)
        stream_process_output_fn: Function to stream process output
        extra_args: Additional CLI arguments (e.g., ["--n-cycles", "6"])
        gpu_id: CUDA_VISIBLE_DEVICES value for the trainer GPU
        world_size: Total distributed world size
        command: Positional command for distributed_trainer_helper.py
            (ignored for fast_llm_trainer_helper.py which uses --init-method style)

    Returns:
        Tuple of (trainer_proc, stdout_thread, stderr_thread)
    """
    trainer_env = os.environ.copy()
    trainer_env["CUDA_VISIBLE_DEVICES"] = gpu_id

    print(f"[Main] Starting trainer process (GPU {gpu_id}) for process group rendezvous")

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
            "--world-size", str(world_size),
            "--server-urls",
        ] + list(server_urls))
    else:
        # distributed_trainer_helper.py uses positional command + flags
        cmd.extend([
            command,
            "--init-method", distributed_init_method,
            "--model-name", model_name,
            "--world-size", str(world_size),
            "--server-urls",
        ] + list(server_urls))

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
