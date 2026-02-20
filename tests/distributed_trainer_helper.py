#!/usr/bin/env python3
"""Helper script for distributed trainer process.

This script is run as a separate process with CUDA_VISIBLE_DEVICES set,
allowing proper GPU isolation for distributed tests.
"""

import sys
import argparse
import logging

# Setup debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [TRAINER-%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_model_path(model_name: str):
    """Resolve model name to a local Path, downloading from HuggingFace if needed."""
    from pathlib import Path
    from huggingface_hub import snapshot_download

    model_path = Path(model_name)
    if not model_path.exists():
        print(f"[Trainer] Downloading model from HuggingFace Hub: {model_name}")
        model_path = Path(snapshot_download(model_name))
    return model_path


def _load_state_dict(model_name: str, device: str = "cuda:0") -> tuple:
    """Load model state dict from safetensors files.

    Returns:
        (state_dict, model_path)
    """
    import json
    from safetensors.torch import load_file

    model_path = _resolve_model_path(model_name)
    index_file = model_path / "model.safetensors.index.json"

    if index_file.exists():
        print(f"[Trainer] Found index file, loading sharded model")
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        file_to_params = {}
        for param_name, filename in weight_map.items():
            file_to_params.setdefault(filename, []).append(param_name)

        state_dict = {}
        for filename, param_names in file_to_params.items():
            file_path = model_path / filename
            print(f"[Trainer] Loading {len(param_names)} parameters from {filename}")
            tensors = load_file(str(file_path), device=device)
            for param_name in param_names:
                state_dict[param_name] = tensors[param_name]
    else:
        safetensors_file = model_path / "model.safetensors"
        print(f"[Trainer] Loading from single file: {safetensors_file}")
        state_dict = load_file(str(safetensors_file), device=device)

    print(f"[Trainer] Loaded {len(state_dict)} parameters from safetensors")
    return state_dict, model_path


def _init_actor_process_group(init_method: str, rank: int = 0, world_size: int = 2):
    """Initialize the actor NCCL process group and return it."""
    import pipelinerl.torch_utils

    print(f"[Trainer] Initializing process group as rank {rank}")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    print("[Trainer] Process group initialized")
    return process_group


def _create_perturbed_state_dict(
    state_dict: dict, seed: int = 42, noise_scale: float = 0.001
) -> dict:
    """Return a new state dict with Gaussian noise added to all tensors."""
    import torch

    print(f"[Trainer] Creating perturbed weights (all tensors) with seed={seed}...")
    torch.manual_seed(seed)
    perturbed = {}
    for name, tensor in state_dict.items():
        perturbed_tensor = tensor.clone()
        perturbed_tensor.add_(torch.randn_like(perturbed_tensor) * noise_scale)
        perturbed[name] = perturbed_tensor
    print(
        f"[Trainer] Perturbed all {len(perturbed)} tensors with noise={noise_scale}, seed={seed}"
    )
    return perturbed


def _broadcast_tensors(state_dict: dict, process_group, log_interval: int = 50):
    """Broadcast every tensor in state_dict via NCCL (src=0)."""
    import torch.distributed as dist

    total = len(state_dict)
    for i, (name, tensor) in enumerate(state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % log_interval == 0:
            print(f"[Trainer] Broadcasted {i+1}/{total} parameters")
    print(f"[Trainer] All {total} parameters broadcasted")


def _broadcast_via_server(
    state_dict: dict,
    server_url: str,
    version: int,
    process_group,
    label: str = "",
):
    """Broadcast weights to a running vLLM server via HTTP POST + NCCL.

    The POST blocks on the server side until NCCL broadcast completes, so we
    run it in a background thread while we drive the broadcast ourselves.
    """
    import threading
    import time
    import requests
    from weight_update_utils import create_weight_update_request_from_state_dict

    label_str = f" {label}" if label else ""
    print(f"[Trainer] Broadcasting {len(state_dict)}{label_str} parameters")

    request = create_weight_update_request_from_state_dict(state_dict, version=version)

    post_result = {"error": None}

    def _post():
        try:
            print("[Trainer] POSTing weight update request to server...")
            resp = requests.post(
                f"{server_url}/receive_weight_update",
                json=request.model_dump(),
                timeout=600,
            )
            if resp.status_code != 200:
                post_result["error"] = (
                    f"POST failed with status {resp.status_code}: {resp.text}"
                )
            else:
                print("[Trainer] Server acknowledged weight update")
        except Exception as e:
            post_result["error"] = f"POST failed: {e}"

    post_thread = threading.Thread(target=_post, daemon=False)
    post_thread.start()
    time.sleep(0.5)  # Give server a moment to start receiving

    _broadcast_tensors(state_dict, process_group)

    post_thread.join(timeout=60)
    if post_result["error"]:
        raise RuntimeError(f"Weight update POST failed: {post_result['error']}")

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


def broadcast_back_and_forth(init_method: str, model_name: str, sync_dir: str):
    """Back-and-forth test: broadcast perturbed → original → perturbed again.

    Tests that we can switch between weight sets multiple times.
    """
    import torch.distributed as dist
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, write_weight_update_request
    from weight_update_utils import create_weight_update_request_from_state_dict

    sync_path = Path(sync_dir)
    baseline_done = SyncPoint(sync_path, "baseline_done")
    ready_for_perturbed1 = SyncPoint(sync_path, "ready_for_perturbed1")
    perturbed1_done = SyncPoint(sync_path, "perturbed1_done")
    ready_for_original = SyncPoint(sync_path, "ready_for_original")
    original_done = SyncPoint(sync_path, "original_done")
    ready_for_perturbed2 = SyncPoint(sync_path, "ready_for_perturbed2")
    perturbed2_done = SyncPoint(sync_path, "perturbed2_done")

    process_group = _init_actor_process_group(init_method, rank=0, world_size=2)

    print("[Trainer] Waiting for vLLM baseline generation...")
    baseline_done.wait(timeout=120)

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
    print("[Trainer] Waiting for vLLM to be ready for first perturbed broadcast...")
    ready_for_perturbed1.wait(timeout=120)

    print(f"[Trainer] Broadcasting perturbed weights (1st time)")
    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=1)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(perturbed_state_dict, process_group)

    perturbed1_done.signal()
    print("[Trainer] First perturbed broadcast complete")

    # Broadcast 2: Original weights
    print("[Trainer] Waiting for vLLM to be ready for original broadcast...")
    ready_for_original.wait(timeout=120)

    print(f"[Trainer] Broadcasting original weights")
    request = create_weight_update_request_from_state_dict(original_state_dict, version=2)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(original_state_dict, process_group)

    original_done.signal()
    print("[Trainer] Original broadcast complete")

    # Broadcast 3: Perturbed weights again (same as first)
    print("[Trainer] Waiting for vLLM to be ready for second perturbed broadcast...")
    ready_for_perturbed2.wait(timeout=120)

    print(f"[Trainer] Broadcasting perturbed weights (2nd time)")
    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=3)
    write_weight_update_request(sync_path, request)
    _broadcast_tensors(perturbed_state_dict, process_group)

    perturbed2_done.signal()
    print("[Trainer] Second perturbed broadcast complete")

    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed")


def timed_broadcast_server_test(
    init_method: str, model_name: str, server_url: str
):
    """Timed broadcast for server tests: perturbed → original → perturbed with delays.

    This simulates a real-world scenario where weight updates happen while
    the server is running and serving requests.

    Pattern: original (server default) → perturbed → original → perturbed

    Args:
        init_method: Distributed init method
        model_name: Model name to load
        server_url: Base URL of vLLM server (e.g., "http://127.0.0.1:8000")
    """
    import torch.distributed as dist
    from pathlib import Path
    import time
    import requests

    sys.path.insert(0, str(Path(__file__).parent))

    process_group = _init_actor_process_group(init_method, rank=0, world_size=2)

    # Wait for server to be ready by polling health endpoint
    print("[Trainer] Waiting for server to be ready...")
    server_ready = False
    for i in range(120):  # Try for up to 2 minutes
        try:
            resp = requests.get(f"{server_url}/health", timeout=1)
            if resp.status_code == 200:
                server_ready = True
                print(f"[Trainer] Server is ready (took {i} seconds)")
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)

    if not server_ready:
        raise TimeoutError("Server did not become ready within 2 minutes")

    # Wait additional 10 seconds for server to fully initialize
    print("[Trainer] Waiting additional 10 seconds for server to fully initialize...")
    time.sleep(10)

    print(f"[Trainer] Loading original weights from {model_name}")
    original_state_dict, _ = _load_state_dict(model_name)

    perturbed_state_dict = _create_perturbed_state_dict(original_state_dict)

    # Broadcast 1: Perturbed weights
    _broadcast_via_server(perturbed_state_dict, server_url, version=1, process_group=process_group, label="perturbed")

    print("[Trainer] Waiting 5 seconds before broadcasting original weights...")
    time.sleep(5)

    # Broadcast 2: Original weights
    _broadcast_via_server(original_state_dict, server_url, version=2, process_group=process_group, label="original")

    print("[Trainer] Waiting 5 seconds before broadcasting perturbed weights again...")
    time.sleep(5)

    # Broadcast 3: Perturbed weights again (same as first)
    _broadcast_via_server(perturbed_state_dict, server_url, version=3, process_group=process_group, label="perturbed (2nd time)")

    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed, exiting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed trainer helper")
    parser.add_argument("command", choices=["init", "broadcast", "cross_validation", "back_and_forth", "timed_broadcast_server_test"])
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
        "--server-url", type=str, help="Base URL of vLLM server (e.g., http://127.0.0.1:8000)"
    )

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
            broadcast_back_and_forth(args.init_method, args.model_name, args.sync_dir)
        elif args.command == "timed_broadcast_server_test":
            if not args.model_name or not args.server_url:
                print("Error: --model-name and --server-url required for timed_broadcast_server_test")
                sys.exit(1)
            timed_broadcast_server_test(
                args.init_method, args.model_name, args.server_url
            )
    except Exception as e:
        print(f"[Trainer] Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
