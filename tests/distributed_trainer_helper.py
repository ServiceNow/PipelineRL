#!/usr/bin/env python3
"""Helper script for distributed trainer process.

This script is run as a separate process with CUDA_VISIBLE_DEVICES set,
allowing proper GPU isolation for distributed tests.
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trainer_test_utils import (
    _resolve_model_path,
    _load_state_dict,
    _create_perturbed_state_dict,
    _init_actor_process_group,
    _broadcast_tensors,
    _wait_for_servers_ready,
)

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [TRAINER-%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _wait_all_actors(sync_path, name: str, num_actors: int, timeout: float = 120):
    """Wait for all actors to signal a named sync point.

    Each actor signals ``{name}_actor_{i}`` for i in range(num_actors).
    """
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint

    for i in range(num_actors):
        SyncPoint(sync_path, f"{name}_actor_{i}").wait(timeout=timeout)


def _broadcast_via_server(
    state_dict: dict,
    server_urls: list,
    version: int,
    process_group,
    label: str = "",
):
    """Broadcast weights to one or more running vLLM servers via HTTP POST + NCCL.

    One POST thread is started per server URL (all in parallel) before the
    NCCL broadcast so that all servers are ready to receive simultaneously.
    """
    import threading
    import time
    import requests
    from weight_update_utils import create_weight_update_request_from_state_dict

    label_str = f" {label}" if label else ""
    print(f"[Trainer] Broadcasting {len(state_dict)}{label_str} parameters to {len(server_urls)} server(s)")

    request = create_weight_update_request_from_state_dict(state_dict, version=version)

    errors = []
    threads = []

    for url in server_urls:
        err = {"error": None}
        errors.append(err)

        def _post(server_url=url, post_result=err):
            try:
                print(f"[Trainer] POSTing weight update request to {server_url}...")
                resp = requests.post(
                    f"{server_url}/receive_weight_update",
                    json=request.model_dump(),
                    timeout=600,
                )
                if resp.status_code != 200:
                    post_result["error"] = (
                        f"POST to {server_url} failed with status {resp.status_code}: {resp.text}"
                    )
                else:
                    print(f"[Trainer] Server {server_url} acknowledged weight update")
            except Exception as e:
                post_result["error"] = f"POST to {server_url} failed: {e}"

        t = threading.Thread(target=_post, daemon=False)
        threads.append(t)
        t.start()

    time.sleep(0.5)  # Give all servers a moment to start receiving

    _broadcast_tensors(state_dict, process_group)

    for t in threads:
        t.join(timeout=60)

    failed = [e["error"] for e in errors if e["error"]]
    if failed:
        raise RuntimeError(f"Weight update POST(s) failed: {failed}")

    print(f"[Trainer] Broadcast{label_str} complete")


# ---------------------------------------------------------------------------
# Public command functions
# ---------------------------------------------------------------------------

def init_process_group(init_method: str, rank: int, world_size: int):
    """Initialize a distributed process group and wait."""
    import torch.distributed as dist
    import time

    process_group = _init_actor_process_group(init_method, rank, world_size)
    print(f"[Trainer rank={rank}] Process group initialized successfully")

    # Wait for coordination
    time.sleep(3)

    print(f"[Trainer rank={rank}] Destroying process group")
    dist.destroy_process_group(process_group)
    print(f"[Trainer rank={rank}] Process group destroyed")


def save_model_to_dir(state_dict: dict, output_dir: str, model_name: str):
    """Save state_dict to a directory as safetensors with config.

    Args:
        state_dict: Model state dict to save
        output_dir: Directory to save model
        model_name: Original model name to copy config from
    """
    from pathlib import Path
    from safetensors.torch import save_file
    import shutil

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    safetensors_path = output_path / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"[Trainer] Saved model weights to {safetensors_path}")

    # Copy config.json from original model
    original_path = _resolve_model_path(model_name)

    config_src = original_path / "config.json"
    config_dst = output_path / "config.json"
    shutil.copy(config_src, config_dst)
    print(f"[Trainer] Copied config.json to {config_dst}")

    # Copy tokenizer files
    for filename in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
    ]:
        src = original_path / filename
        if src.exists():
            dst = output_path / filename
            shutil.copy(src, dst)
            print(f"[Trainer] Copied {filename}")

    return str(output_path)


def broadcast_weights(
    init_method: str, model_name: str, perturb: bool = False, sync_dir: str = None
):
    """Load model and broadcast weights to vLLM worker."""
    import torch
    import torch.distributed as dist
    from pathlib import Path

    # Setup sync points if provided
    if sync_dir:
        sys.path.insert(0, str(Path(__file__).parent))
        from sync_helper import SyncPoint, write_weight_update_request

        sync_path = Path(sync_dir)
        baseline_done = SyncPoint(sync_path, "baseline_done")
        ready_to_receive = SyncPoint(sync_path, "ready_to_receive")
        request_ready = SyncPoint(sync_path, "request_ready")
        receiving_started = SyncPoint(sync_path, "receiving_started")
        broadcast_done = SyncPoint(sync_path, "broadcast_done")

    # IMPORTANT: Initialize process group FIRST (before any waiting)
    process_group = _init_actor_process_group(init_method, rank=0, world_size=2)

    # Now wait for vLLM to finish baseline and be ready to receive
    if sync_dir:
        print("[Trainer] Waiting for vLLM to finish baseline generation...")
        baseline_done.wait(timeout=60)
        print("[Trainer] Baseline done")

        print("[Trainer] Waiting for vLLM to be ready to receive weights...")
        ready_to_receive.wait(timeout=60)
        print("[Trainer] vLLM ready, starting weight broadcast")

    print(f"[Trainer] Loading tensors from safetensors for {model_name}")
    state_dict, _ = _load_state_dict(model_name)

    params_to_broadcast = state_dict
    print(f"[Trainer] Will broadcast {len(params_to_broadcast)} parameters")

    # Create and send WeightUpdateRequest to vLLM
    if sync_dir:
        from weight_update_utils import create_weight_update_request_from_state_dict

        print("[Trainer] Creating WeightUpdateRequest...")
        request = create_weight_update_request_from_state_dict(
            params_to_broadcast, version=1
        )
        write_weight_update_request(sync_path, request)
        request_ready.signal()
        print(
            f"[Trainer] Sent WeightUpdateRequest with {len(request.parameters_info)} parameters"
        )

        # Wait for vLLM to start receiving before we broadcast
        print("[Trainer] Waiting for vLLM to start receiving...")
        receiving_started.wait(timeout=60)
        print("[Trainer] vLLM is receiving, starting broadcast")

    print(f"[Trainer] Broadcasting {len(params_to_broadcast)} parameters")

    # Optionally perturb weights - add noise to ALL tensors
    if perturb:
        params_to_broadcast = _create_perturbed_state_dict(params_to_broadcast)

    # Broadcast each weight with detailed logging
    logger.info(f"Starting broadcast of {len(params_to_broadcast)} parameters")
    for i, (name, tensor) in enumerate(params_to_broadcast.items()):
        logger.debug(f"[{i+1}/{len(state_dict)}] Preparing to broadcast: {name}")
        logger.debug(
            f"  - shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}"
        )
        if tensor.device.type != "cuda":
            logger.debug(f"  - Moving {name} to CUDA")
            tensor = tensor.cuda(0)
            logger.debug(f"  - {name} now on device: {tensor.device}")
        logger.debug(f"  - Calling dist.broadcast for {name}...")
        dist.broadcast(tensor, src=0, group=process_group)
        logger.debug(f"  - Broadcast complete for {name}")
        if (i + 1) % 10 == 0:
            logger.info(f"Broadcasted {i+1}/{len(params_to_broadcast)} parameters")

    print(f"[Trainer] All {len(params_to_broadcast)} parameters broadcasted")

    # Signal broadcast complete BEFORE destroying process group
    if sync_dir:
        broadcast_done.signal()
        print("[Trainer] Signaled broadcast complete")

    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed")


def broadcast_cross_validation(
    init_method: str, model_name: str, sync_dir: str, temp_dir: str
):
    """Cross-validation test: broadcast perturbed, then original weights.

    Also saves perturbed model to disk for vLLM to load.
    """
    import torch.distributed as dist
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, write_weight_update_request
    from weight_update_utils import create_weight_update_request_from_state_dict

    sync_path = Path(sync_dir)
    baseline_done = SyncPoint(sync_path, "baseline_done")
    perturbed_model_saved = SyncPoint(sync_path, "perturbed_model_saved")
    ready_to_receive_perturbed = SyncPoint(sync_path, "ready_to_receive_perturbed")
    perturbed_broadcast_done = SyncPoint(sync_path, "perturbed_broadcast_done")
    mod1_done = SyncPoint(sync_path, "mod1_done")
    first_engine_destroyed = SyncPoint(sync_path, "first_engine_destroyed")
    engine_recreated = SyncPoint(sync_path, "engine_recreated")
    ready_to_receive_original = SyncPoint(sync_path, "ready_to_receive_original")
    original_broadcast_done = SyncPoint(sync_path, "original_broadcast_done")

    process_group = _init_actor_process_group(init_method, rank=0, world_size=2)

    print("[Trainer] Waiting for vLLM baseline generation...")
    baseline_done.wait(timeout=120)

    print(f"[Trainer] Loading original model {model_name}")
    original_state_dict, model_path = _load_state_dict(model_name)

    perturbed_state_dict = _create_perturbed_state_dict(original_state_dict)

    # Save perturbed model to disk
    perturbed_model_dir = Path(temp_dir) / "perturbed_model"
    print(f"[Trainer] Saving perturbed model to {perturbed_model_dir}")
    saved_path = save_model_to_dir(
        perturbed_state_dict, str(perturbed_model_dir), str(model_path)
    )

    path_file = sync_path / "perturbed_model_path.txt"
    path_file.write_text(saved_path)
    perturbed_model_saved.signal()
    print(f"[Trainer] Signaled perturbed model saved at: {saved_path}")

    # Broadcast perturbed weights
    print("[Trainer] Waiting for vLLM to be ready for perturbed broadcast...")
    ready_to_receive_perturbed.wait(timeout=120)

    print(f"[Trainer] Broadcasting {len(perturbed_state_dict)} perturbed parameters")
    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=1)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(perturbed_state_dict, process_group)

    perturbed_broadcast_done.signal()
    print("[Trainer] Perturbed weights broadcast complete")

    print("[Trainer] Waiting for vLLM to finish res_mod_1...")
    mod1_done.wait(timeout=120)

    print("[Trainer] Destroying process group for first broadcast")
    dist.destroy_process_group(process_group)

    print("[Trainer] Waiting for vLLM to destroy first engine...")
    first_engine_destroyed.wait(timeout=120)

    print("[Trainer] Recreating process group for second broadcast")
    process_group = _init_actor_process_group(init_method, rank=0, world_size=2)
    print("[Trainer] Process group recreated, waiting at rendezvous...")

    print("[Trainer] Waiting for vLLM to recreate engine...")
    engine_recreated.wait(timeout=300)  # 5 minutes - engine creation can be slow
    print("[Trainer] vLLM engine recreated, both in new process group")

    # Broadcast original weights
    print("[Trainer] Waiting for vLLM to be ready for original broadcast...")
    ready_to_receive_original.wait(timeout=120)

    print(f"[Trainer] Broadcasting {len(original_state_dict)} original parameters")
    request = create_weight_update_request_from_state_dict(original_state_dict, version=2)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(original_state_dict, process_group)

    original_broadcast_done.signal()
    print("[Trainer] Original weights broadcast complete")

    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed")


def broadcast_back_and_forth(
    init_method: str,
    model_name: str,
    sync_dir: str,
    num_actors: int = 1,
    world_size: int = 2,
):
    """Back-and-forth test: broadcast perturbed → original → perturbed again.

    Tests that we can switch between weight sets multiple times.
    Supports multiple actors: waits for all actors to signal readiness before
    each broadcast, then sends a single shared completion signal.
    """
    import torch.distributed as dist
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, write_weight_update_request
    from weight_update_utils import create_weight_update_request_from_state_dict

    sync_path = Path(sync_dir)
    perturbed1_done = SyncPoint(sync_path, "perturbed1_done")
    original_done = SyncPoint(sync_path, "original_done")
    perturbed2_done = SyncPoint(sync_path, "perturbed2_done")

    process_group = _init_actor_process_group(init_method, rank=0, world_size=world_size)

    print(f"[Trainer] Waiting for {num_actors} actor(s) to finish baseline generation...")
    _wait_all_actors(sync_path, "baseline_done", num_actors, timeout=120)

    print(f"[Trainer] Loading model {model_name}")
    original_state_dict, model_path = _load_state_dict(model_name)

    perturbed_state_dict = _create_perturbed_state_dict(original_state_dict)

    # Save perturbed weights for reuse in server tests
    perturbed_weights_dir = Path(sync_dir) / "perturbed_weights"
    print(f"[Trainer] Saving perturbed weights to {perturbed_weights_dir}")
    saved_path = save_model_to_dir(
        perturbed_state_dict, str(perturbed_weights_dir), str(model_path)
    )
    print(f"[Trainer] Perturbed weights saved to {saved_path}")

    # Broadcast 1: Perturbed weights
    print(f"[Trainer] Waiting for {num_actors} actor(s) to be ready for first perturbed broadcast...")
    _wait_all_actors(sync_path, "ready_for_perturbed1", num_actors, timeout=120)

    print(f"[Trainer] Broadcasting perturbed weights (1st time) to {num_actors} actor(s)")
    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=1)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(perturbed_state_dict, process_group)

    perturbed1_done.signal()
    print("[Trainer] First perturbed broadcast complete")

    # Broadcast 2: Original weights
    print(f"[Trainer] Waiting for {num_actors} actor(s) to be ready for original broadcast...")
    _wait_all_actors(sync_path, "ready_for_original", num_actors, timeout=120)

    print(f"[Trainer] Broadcasting original weights to {num_actors} actor(s)")
    request = create_weight_update_request_from_state_dict(original_state_dict, version=2)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(original_state_dict, process_group)

    original_done.signal()
    print("[Trainer] Original broadcast complete")

    # Broadcast 3: Perturbed weights again (same as first)
    print(f"[Trainer] Waiting for {num_actors} actor(s) to be ready for second perturbed broadcast...")
    _wait_all_actors(sync_path, "ready_for_perturbed2", num_actors, timeout=120)

    print(f"[Trainer] Broadcasting perturbed weights (2nd time) to {num_actors} actor(s)")
    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=3)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(perturbed_state_dict, process_group)

    perturbed2_done.signal()
    print("[Trainer] Second perturbed broadcast complete")

    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed")


def timed_broadcast_server_test(
    init_method: str,
    model_name: str,
    server_urls: list,
    world_size: int = 2,
):
    """Timed broadcast for server tests: perturbed → original → perturbed with delays.

    This simulates a real-world scenario where weight updates happen while
    the server is running and serving requests.

    Pattern: original (server default) → perturbed → original → perturbed

    Args:
        init_method: Distributed init method
        model_name: Model name to load
        server_urls: List of base URLs of vLLM servers (e.g., ["http://127.0.0.1:8000"])
        world_size: Total world size (trainer rank 0 + all vLLM workers)
    """
    import torch.distributed as dist
    import time
    import requests

    process_group = _init_actor_process_group(init_method, rank=0, world_size=world_size)

    _wait_for_servers_ready(server_urls, extra_wait_secs=10)

    print(f"[Trainer] Loading original weights from {model_name}")
    original_state_dict, _ = _load_state_dict(model_name)

    perturbed_state_dict = _create_perturbed_state_dict(original_state_dict)

    # Broadcast 1: Perturbed weights
    _broadcast_via_server(perturbed_state_dict, server_urls, version=1, process_group=process_group, label="perturbed")

    print("[Trainer] Waiting 5 seconds before broadcasting original weights...")
    time.sleep(5)

    # Broadcast 2: Original weights
    _broadcast_via_server(original_state_dict, server_urls, version=2, process_group=process_group, label="original")

    print("[Trainer] Waiting 5 seconds before broadcasting perturbed weights again...")
    time.sleep(5)

    # Broadcast 3: Perturbed weights again (same as first)
    _broadcast_via_server(perturbed_state_dict, server_urls, version=3, process_group=process_group, label="perturbed (2nd time)")

    # Wait to allow generation with the last broadcast before tearing down
    print("[Trainer] Waiting 5 seconds for generation with final weights...")
    time.sleep(5)

    # Signal training is finished so vLLM servers destroy their side of the process group
    for url in server_urls:
        print(f"[Trainer] Sending training_finished signal to {url}...")
        requests.post(f"{url}/training_finished", timeout=10)

    # Cleanup — destroy_process_group now resolves because vLLM servers respond to /training_finished
    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed, exiting")


def rapid_broadcast_cycles(
    init_method: str,
    model_name: str,
    server_urls: list,
    world_size: int = 2,
    n_cycles: int = 6,
):
    """Hybrid broadcast designed to catch transition/garbage generations.

    Structure:
      1. Slow broadcast: perturbed  (5 s wait after) — establishes text_B
      2. Slow broadcast: original   (5 s wait after) — re-establishes text_A
      3. n_cycles rapid pairs: perturbed → original  (1 s between each)
      4. Slow broadcast: perturbed  (5 s wait after) — end on text_B so the
         overall A→B→A→B pattern remains detectable

    The slow initial cycles give the generation loop enough stable time to
    identify text_A and text_B by frequency.  The rapid cycles create many
    short broadcast windows where mid-broadcast (garbage) generations are
    likely to be caught by a zero-interval generation loop.
    """
    import torch.distributed as dist
    import time
    import requests

    process_group = _init_actor_process_group(init_method, rank=0, world_size=world_size)

    _wait_for_servers_ready(server_urls, extra_wait_secs=10)

    print(f"[Trainer] Loading weights from {model_name}")
    original_state_dict, _ = _load_state_dict(model_name)
    perturbed_state_dict = _create_perturbed_state_dict(original_state_dict)

    version = 1

    # --- Slow cycle: establish text_B and text_A clearly ---
    print("[Trainer] Slow broadcast 1: perturbed (establishing text_B)...")
    _broadcast_via_server(perturbed_state_dict, server_urls, version=version, process_group=process_group, label="perturbed (slow)")
    version += 1
    time.sleep(5)

    print("[Trainer] Slow broadcast 2: original (re-establishing text_A)...")
    _broadcast_via_server(original_state_dict, server_urls, version=version, process_group=process_group, label="original (slow)")
    version += 1
    time.sleep(5)

    # --- Rapid cycles: 1 s between broadcasts ---
    for i in range(n_cycles):
        print(f"[Trainer] Rapid cycle {i + 1}/{n_cycles}: perturbed...")
        _broadcast_via_server(perturbed_state_dict, server_urls, version=version, process_group=process_group, label=f"perturbed (rapid {i + 1})")
        version += 1
        time.sleep(1)

        print(f"[Trainer] Rapid cycle {i + 1}/{n_cycles}: original...")
        _broadcast_via_server(original_state_dict, server_urls, version=version, process_group=process_group, label=f"original (rapid {i + 1})")
        version += 1
        time.sleep(1)

    # --- Final slow broadcast: end on perturbed so ABAB pattern holds ---
    print("[Trainer] Final slow broadcast: perturbed (ending on text_B)...")
    _broadcast_via_server(perturbed_state_dict, server_urls, version=version, process_group=process_group, label="perturbed (final)")
    time.sleep(5)

    for url in server_urls:
        print(f"[Trainer] Sending training_finished signal to {url}...")
        requests.post(f"{url}/training_finished", timeout=10)

    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed, exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed trainer helper")
    parser.add_argument("command", choices=["init", "broadcast", "cross_validation", "back_and_forth", "timed_broadcast_server_test", "rapid_broadcast_cycles"])
    parser.add_argument("--init-method", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--sync-dir", type=str, help="Directory for sync files")
    parser.add_argument(
        "--temp-dir", type=str, help="Temporary directory for saving models"
    )
    parser.add_argument(
        "--server-urls", nargs="+", help="Base URL(s) of vLLM server(s) (e.g., http://127.0.0.1:8000)"
    )
    parser.add_argument("--num-actors", type=int, default=1, help="Number of vLLM actor processes")
    parser.add_argument("--n-cycles", type=int, default=6, help="Number of rapid broadcast cycles (rapid_broadcast_cycles command)")

    args = parser.parse_args()

    try:
        if args.command == "init":
            init_process_group(args.init_method, args.rank, args.world_size)
        elif args.command == "broadcast":
            if not args.model_name:
                print("Error: --model-name required for broadcast command")
                sys.exit(1)
            broadcast_weights(
                args.init_method, args.model_name, args.perturb, args.sync_dir
            )
        elif args.command == "cross_validation":
            if not args.model_name or not args.sync_dir or not args.temp_dir:
                print(
                    "Error: --model-name, --sync-dir, and --temp-dir required for cross_validation"
                )
                sys.exit(1)
            broadcast_cross_validation(
                args.init_method, args.model_name, args.sync_dir, args.temp_dir
            )
        elif args.command == "back_and_forth":
            if not args.model_name or not args.sync_dir:
                print("Error: --model-name and --sync-dir required for back_and_forth")
                sys.exit(1)
            broadcast_back_and_forth(
                args.init_method,
                args.model_name,
                args.sync_dir,
                num_actors=args.num_actors,
                world_size=args.world_size,
            )
        elif args.command == "timed_broadcast_server_test":
            if not args.model_name or not args.server_urls:
                print("Error: --model-name and --server-urls required for timed_broadcast_server_test")
                sys.exit(1)
            timed_broadcast_server_test(
                args.init_method,
                args.model_name,
                args.server_urls,
                world_size=args.world_size,
            )
        elif args.command == "rapid_broadcast_cycles":
            if not args.model_name or not args.server_urls:
                print("Error: --model-name and --server-urls required for rapid_broadcast_cycles")
                sys.exit(1)
            rapid_broadcast_cycles(
                args.init_method,
                args.model_name,
                args.server_urls,
                world_size=args.world_size,
                n_cycles=args.n_cycles,
            )
    except Exception as e:
        print(f"[Trainer] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
