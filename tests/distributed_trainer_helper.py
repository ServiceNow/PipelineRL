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


def init_process_group(init_method: str, rank: int, world_size: int):
    """Initialize a distributed process group and wait."""
    import torch.distributed as dist
    import time
    import pipelinerl.torch_utils

    print(f"[Trainer rank={rank}] Initializing process group")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
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
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save weights as safetensors
    safetensors_path = output_path / "model.safetensors"
    save_file(state_dict, str(safetensors_path))
    print(f"[Trainer] Saved model weights to {safetensors_path}")

    # Copy config.json from original model
    original_path = Path(model_name)
    if not original_path.exists():
        # Download if needed
        from huggingface_hub import snapshot_download

        original_path = Path(snapshot_download(model_name))

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
    from transformers import AutoModelForCausalLM
    from pathlib import Path
    import pipelinerl.torch_utils

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
    # Use the same init_extra_process_group as vLLM to create the SAME process group
    print("[Trainer] Initializing process group as rank 0")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=0,
        world_size=2,
    )
    print("[Trainer] Process group initialized")

    # Now wait for vLLM to finish baseline and be ready to receive
    if sync_dir:
        print("[Trainer] Waiting for vLLM to finish baseline generation...")
        baseline_done.wait(timeout=60)
        print("[Trainer] Baseline done")

        print("[Trainer] Waiting for vLLM to be ready to receive weights...")
        ready_to_receive.wait(timeout=60)
        print("[Trainer] vLLM ready, starting weight broadcast")

    # Load tensors directly from safetensors files (not the full model)
    print(f"[Trainer] Loading tensors from safetensors for {model_name}")
    from pathlib import Path
    import json
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    # Handle both local paths and HuggingFace model IDs
    model_path = Path(model_name)
    if not model_path.exists():
        # Download from HuggingFace Hub
        print(f"[Trainer] Downloading model from HuggingFace Hub: {model_name}")
        model_path = Path(snapshot_download(model_name))

    index_file = model_path / "model.safetensors.index.json"

    # Load state_dict from safetensors files
    if index_file.exists():
        # Sharded model - use index to load from multiple files
        print(f"[Trainer] Found index file, loading sharded model")
        with open(index_file) as f:
            index = json.load(f)

        weight_map = index["weight_map"]  # {param_name: filename}

        # Group parameters by file to load each file only once
        file_to_params = {}
        for param_name, filename in weight_map.items():
            if filename not in file_to_params:
                file_to_params[filename] = []
            file_to_params[filename].append(param_name)

        # Load all tensors
        state_dict = {}
        for filename, param_names in file_to_params.items():
            file_path = model_path / filename
            print(f"[Trainer] Loading {len(param_names)} parameters from {filename}")
            tensors = load_file(str(file_path), device="cuda:0")
            for param_name in param_names:
                state_dict[param_name] = tensors[param_name]
    else:
        # Single file model
        safetensors_file = model_path / "model.safetensors"
        print(f"[Trainer] Loading from single file: {safetensors_file}")
        state_dict = load_file(str(safetensors_file), device="cuda:0")

    print(f"[Trainer] Loaded {len(state_dict)} parameters from safetensors")

    # Fast-LLM broadcasts weights as they are in safetensors files
    # No filtering - vLLM handles its own implementation details
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
        logger.info("Perturbing ALL weights with seed=42")
        torch.manual_seed(42)
        for name, tensor in params_to_broadcast.items():
            if tensor.device.type != "cuda":
                tensor = tensor.cuda(0)
            noise = torch.randn_like(tensor) * 0.001  # Smaller noise to avoid breaking model
            tensor.add_(noise)
        print(f"[Trainer] Perturbed all {len(params_to_broadcast)} tensors with noise=0.001, seed=42")

    # Broadcast each weight with detailed logging
    logger.info(f"Starting broadcast of {len(params_to_broadcast)} parameters")
    for i, (name, tensor) in enumerate(params_to_broadcast.items()):
        logger.debug(f"[{i+1}/{len(state_dict)}] Preparing to broadcast: {name}")
        logger.debug(
            f"  - shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}"
        )

        # Move to GPU if needed
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
    # This ensures vLLM sees the signal before trainer exits
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
    import torch
    import torch.distributed as dist
    from pathlib import Path
    import json
    import pipelinerl.torch_utils
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, write_weight_update_request

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

    # Initialize process group
    print("[Trainer] Initializing process group as rank 0")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=0,
        world_size=2,
    )
    print("[Trainer] Process group initialized")

    # Wait for baseline
    print("[Trainer] Waiting for vLLM baseline generation...")
    baseline_done.wait(timeout=120)

    # Load original model
    print(f"[Trainer] Loading original model {model_name}")
    model_path = Path(model_name)
    if not model_path.exists():
        print(f"[Trainer] Downloading model from HuggingFace Hub: {model_name}")
        model_path = Path(snapshot_download(model_name))

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        print(f"[Trainer] Loading sharded model")
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        file_to_params = {}
        for param_name, filename in weight_map.items():
            if filename not in file_to_params:
                file_to_params[filename] = []
            file_to_params[filename].append(param_name)

        original_state_dict = {}
        for filename, param_names in file_to_params.items():
            file_path = model_path / filename
            tensors = load_file(str(file_path), device="cuda:0")
            for param_name in param_names:
                original_state_dict[param_name] = tensors[param_name]
    else:
        safetensors_file = model_path / "model.safetensors"
        original_state_dict = load_file(str(safetensors_file), device="cuda:0")

    print(f"[Trainer] Loaded {len(original_state_dict)} original parameters")

    # Create perturbed version - add noise to ALL tensors
    print("[Trainer] Creating perturbed weights (all tensors) with seed=42...")
    torch.manual_seed(42)
    perturbed_state_dict = {}
    for name, tensor in original_state_dict.items():
        perturbed_tensor = tensor.clone()
        # Add smaller noise to avoid completely breaking the model
        noise = torch.randn_like(perturbed_tensor) * 0.001  # Reduced from 0.01
        perturbed_tensor.add_(noise)
        perturbed_state_dict[name] = perturbed_tensor
    print(f"[Trainer] Perturbed all {len(perturbed_state_dict)} tensors with noise=0.001, seed=42")

    # Save perturbed model to disk
    perturbed_model_dir = Path(temp_dir) / "perturbed_model"
    print(f"[Trainer] Saving perturbed model to {perturbed_model_dir}")
    saved_path = save_model_to_dir(
        perturbed_state_dict, str(perturbed_model_dir), str(model_path)
    )

    # Write perturbed model path to sync file
    path_file = sync_path / "perturbed_model_path.txt"
    path_file.write_text(saved_path)
    perturbed_model_saved.signal()
    print(f"[Trainer] Signaled perturbed model saved at: {saved_path}")

    # Wait for vLLM to be ready to receive perturbed weights
    print("[Trainer] Waiting for vLLM to be ready for perturbed broadcast...")
    ready_to_receive_perturbed.wait(timeout=120)

    # Broadcast perturbed weights
    print(f"[Trainer] Broadcasting {len(perturbed_state_dict)} perturbed parameters")
    from weight_update_utils import create_weight_update_request_from_state_dict

    request = create_weight_update_request_from_state_dict(
        perturbed_state_dict, version=1
    )
    write_weight_update_request(sync_path, request)

    for i, (name, tensor) in enumerate(perturbed_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(
                f"[Trainer] Broadcasted {i+1}/{len(perturbed_state_dict)} perturbed parameters"
            )

    perturbed_broadcast_done.signal()
    print("[Trainer] Perturbed weights broadcast complete")

    # Wait for vLLM to finish generating res_mod_1
    print("[Trainer] Waiting for vLLM to finish res_mod_1...")
    mod1_done.wait(timeout=120)

    # Destroy our process group immediately after we're done using it
    # No need to wait for vLLM - destroy_process_group() is a local operation
    print("[Trainer] Destroying process group for first broadcast")
    dist.destroy_process_group(process_group)

    # Wait for vLLM to destroy its first engine before creating new groups
    print("[Trainer] Waiting for vLLM to destroy first engine...")
    first_engine_destroyed.wait(timeout=120)

    # Recreate our process group BEFORE vLLM creates its engine
    # (vLLM will rendezvous with us when it creates engine 2)
    print("[Trainer] Recreating process group for second broadcast")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=0,
        world_size=2,
    )
    print("[Trainer] Process group recreated, waiting at rendezvous...")

    # Wait for vLLM to recreate engine (confirms rendezvous completed)
    print("[Trainer] Waiting for vLLM to recreate engine...")
    engine_recreated.wait(timeout=300)  # 5 minutes - engine creation can be slow
    print("[Trainer] vLLM engine recreated, both in new process group")

    # Wait for vLLM to be ready for original weights
    print("[Trainer] Waiting for vLLM to be ready for original broadcast...")
    ready_to_receive_original.wait(timeout=120)

    # Broadcast original weights
    print(f"[Trainer] Broadcasting {len(original_state_dict)} original parameters")
    from weight_update_utils import create_weight_update_request_from_state_dict

    request = create_weight_update_request_from_state_dict(
        original_state_dict, version=2
    )
    write_weight_update_request(sync_path, request)

    for i, (name, tensor) in enumerate(original_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(
                f"[Trainer] Broadcasted {i+1}/{len(original_state_dict)} original parameters"
            )

    original_broadcast_done.signal()
    print("[Trainer] Original weights broadcast complete")

    # Cleanup
    dist.destroy_process_group(process_group)
    print("[Trainer] Process group destroyed")


def broadcast_back_and_forth(init_method: str, model_name: str, sync_dir: str):
    """Back-and-forth test: broadcast perturbed → original → perturbed again.

    Tests that we can switch between weight sets multiple times.
    """
    import torch
    import torch.distributed as dist
    from pathlib import Path
    import json
    import pipelinerl.torch_utils
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, write_weight_update_request

    sync_path = Path(sync_dir)
    baseline_done = SyncPoint(sync_path, "baseline_done")
    ready_for_perturbed1 = SyncPoint(sync_path, "ready_for_perturbed1")
    perturbed1_done = SyncPoint(sync_path, "perturbed1_done")
    ready_for_original = SyncPoint(sync_path, "ready_for_original")
    original_done = SyncPoint(sync_path, "original_done")
    ready_for_perturbed2 = SyncPoint(sync_path, "ready_for_perturbed2")
    perturbed2_done = SyncPoint(sync_path, "perturbed2_done")

    # Initialize process group
    print("[Trainer] Initializing process group as rank 0")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=0,
        world_size=2,
    )
    print("[Trainer] Process group initialized")

    # Wait for baseline
    print("[Trainer] Waiting for vLLM baseline generation...")
    baseline_done.wait(timeout=120)

    # Load original model
    print(f"[Trainer] Loading model {model_name}")
    model_path = Path(model_name)
    if not model_path.exists():
        print(f"[Trainer] Downloading model from HuggingFace Hub: {model_name}")
        model_path = Path(snapshot_download(model_name))

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        print(f"[Trainer] Loading sharded model")
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        file_to_params = {}
        for param_name, filename in weight_map.items():
            if filename not in file_to_params:
                file_to_params[filename] = []
            file_to_params[filename].append(param_name)

        original_state_dict = {}
        for filename, param_names in file_to_params.items():
            file_path = model_path / filename
            tensors = load_file(str(file_path), device="cuda:0")
            for param_name in param_names:
                original_state_dict[param_name] = tensors[param_name]
    else:
        safetensors_file = model_path / "model.safetensors"
        original_state_dict = load_file(str(safetensors_file), device="cuda:0")

    print(f"[Trainer] Loaded {len(original_state_dict)} original parameters")

    # Create perturbed version
    print("[Trainer] Creating perturbed weights with seed=42...")
    torch.manual_seed(42)
    perturbed_state_dict = {}
    for name, tensor in original_state_dict.items():
        perturbed_tensor = tensor.clone()
        noise = torch.randn_like(perturbed_tensor) * 0.001
        perturbed_tensor.add_(noise)
        perturbed_state_dict[name] = perturbed_tensor
    print(f"[Trainer] Perturbed all {len(perturbed_state_dict)} tensors with noise=0.001, seed=42")

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
    from weight_update_utils import create_weight_update_request_from_state_dict
    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=1)
    write_weight_update_request(sync_path, request)

    for i, (name, tensor) in enumerate(perturbed_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(f"[Trainer] Broadcasted {i+1}/{len(perturbed_state_dict)} parameters")

    perturbed1_done.signal()
    print("[Trainer] First perturbed broadcast complete")

    # Broadcast 2: Original weights
    print("[Trainer] Waiting for vLLM to be ready for original broadcast...")
    ready_for_original.wait(timeout=120)

    print(f"[Trainer] Broadcasting original weights")
    from weight_update_utils import create_weight_update_request_from_state_dict

    request = create_weight_update_request_from_state_dict(original_state_dict, version=2)
    write_weight_update_request(sync_path, request)

    for i, (name, tensor) in enumerate(original_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(f"[Trainer] Broadcasted {i+1}/{len(original_state_dict)} parameters")

    original_done.signal()
    print("[Trainer] Original broadcast complete")

    # Broadcast 3: Perturbed weights again (same as first)
    print("[Trainer] Waiting for vLLM to be ready for second perturbed broadcast...")
    ready_for_perturbed2.wait(timeout=120)

    print(f"[Trainer] Broadcasting perturbed weights (2nd time)")
    from weight_update_utils import create_weight_update_request_from_state_dict

    request = create_weight_update_request_from_state_dict(perturbed_state_dict, version=3)
    write_weight_update_request(sync_path, request)

    for i, (name, tensor) in enumerate(perturbed_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(f"[Trainer] Broadcasted {i+1}/{len(perturbed_state_dict)} parameters")

    perturbed2_done.signal()
    print("[Trainer] Second perturbed broadcast complete")

    # Cleanup
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
    import torch
    import torch.distributed as dist
    from pathlib import Path
    import json
    import pipelinerl.torch_utils
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    import time
    import requests
    import threading

    sys.path.insert(0, str(Path(__file__).parent))
    from weight_update_utils import create_weight_update_request_from_state_dict

    # Initialize process group
    print("[Trainer] Initializing process group as rank 0")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name="actor",
        backend="nccl",
        init_method=init_method,
        rank=0,
        world_size=2,
    )
    print("[Trainer] Process group initialized")

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

    # Load original weights
    print(f"[Trainer] Loading original weights from {model_name}")
    model_path = Path(model_name)
    if not model_path.exists():
        print(f"[Trainer] Downloading model from HuggingFace Hub: {model_name}")
        model_path = Path(snapshot_download(model_name))

    index_file = model_path / "model.safetensors.index.json"
    if index_file.exists():
        print(f"[Trainer] Loading sharded original model")
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        file_to_params = {}
        for param_name, filename in weight_map.items():
            if filename not in file_to_params:
                file_to_params[filename] = []
            file_to_params[filename].append(param_name)

        original_state_dict = {}
        for filename, param_names in file_to_params.items():
            file_path = model_path / filename
            tensors = load_file(str(file_path), device="cuda:0")
            for param_name in param_names:
                original_state_dict[param_name] = tensors[param_name]
    else:
        safetensors_file = model_path / "model.safetensors"
        original_state_dict = load_file(str(safetensors_file), device="cuda:0")

    print(f"[Trainer] Loaded {len(original_state_dict)} original parameters")

    # Create perturbed weights
    print("[Trainer] Creating perturbed weights with seed=42...")
    torch.manual_seed(42)
    perturbed_state_dict = {}
    for name, tensor in original_state_dict.items():
        perturbed_tensor = tensor.clone()
        noise = torch.randn_like(perturbed_tensor) * 0.001
        perturbed_tensor.add_(noise)
        perturbed_state_dict[name] = perturbed_tensor
    print(f"[Trainer] Perturbed all {len(perturbed_state_dict)} tensors with noise=0.001, seed=42")

    # Broadcast 1: Perturbed weights
    print(f"[Trainer] Broadcasting {len(perturbed_state_dict)} perturbed parameters")

    request = create_weight_update_request_from_state_dict(
        perturbed_state_dict, version=1
    )

    # POST request to server in background thread (it will block until broadcast completes)
    post_result = {"error": None}
    def post_weight_update():
        try:
            print("[Trainer] POSTing weight update request to server...")
            resp = requests.post(
                f"{server_url}/receive_weight_update",
                json=request.model_dump(),
                timeout=600,  # 10 minutes
            )
            if resp.status_code != 200:
                post_result["error"] = f"POST failed with status {resp.status_code}: {resp.text}"
            else:
                print("[Trainer] Server acknowledged weight update")
        except Exception as e:
            post_result["error"] = f"POST failed: {e}"

    post_thread = threading.Thread(target=post_weight_update, daemon=False)
    post_thread.start()

    # Give server a moment to start receiving
    time.sleep(0.5)

    # Now broadcast via NCCL
    for i, (name, tensor) in enumerate(perturbed_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(
                f"[Trainer] Broadcasted {i+1}/{len(perturbed_state_dict)} perturbed parameters"
            )

    # Wait for POST to complete
    post_thread.join(timeout=60)
    if post_result["error"]:
        raise RuntimeError(f"Weight update POST failed: {post_result['error']}")

    print("[Trainer] Perturbed weights broadcast complete")

    # Wait 5 seconds
    print("[Trainer] Waiting 5 seconds before broadcasting original weights...")
    time.sleep(5)

    # Broadcast 2: Original weights
    print(f"[Trainer] Broadcasting {len(original_state_dict)} original parameters")
    request = create_weight_update_request_from_state_dict(
        original_state_dict, version=2
    )

    # POST request to server in background thread
    post_result = {"error": None}
    def post_weight_update():
        try:
            print("[Trainer] POSTing weight update request to server...")
            resp = requests.post(
                f"{server_url}/receive_weight_update",
                json=request.model_dump(),
                timeout=600,
            )
            if resp.status_code != 200:
                post_result["error"] = f"POST failed with status {resp.status_code}: {resp.text}"
            else:
                print("[Trainer] Server acknowledged weight update")
        except Exception as e:
            post_result["error"] = f"POST failed: {e}"

    post_thread = threading.Thread(target=post_weight_update, daemon=False)
    post_thread.start()
    time.sleep(0.5)

    for i, (name, tensor) in enumerate(original_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(
                f"[Trainer] Broadcasted {i+1}/{len(original_state_dict)} original parameters"
            )

    post_thread.join(timeout=60)
    if post_result["error"]:
        raise RuntimeError(f"Weight update POST failed: {post_result['error']}")

    print("[Trainer] Original weights broadcast complete")

    # Wait 5 seconds
    print("[Trainer] Waiting 5 seconds before broadcasting perturbed weights again...")
    time.sleep(5)

    # Broadcast 3: Perturbed weights again (same as first)
    print(f"[Trainer] Broadcasting {len(perturbed_state_dict)} perturbed parameters (2nd time)")
    request = create_weight_update_request_from_state_dict(
        perturbed_state_dict, version=3
    )

    # POST request to server in background thread
    post_result = {"error": None}
    def post_weight_update():
        try:
            print("[Trainer] POSTing weight update request to server...")
            resp = requests.post(
                f"{server_url}/receive_weight_update",
                json=request.model_dump(),
                timeout=600,
            )
            if resp.status_code != 200:
                post_result["error"] = f"POST failed with status {resp.status_code}: {resp.text}"
            else:
                print("[Trainer] Server acknowledged weight update")
        except Exception as e:
            post_result["error"] = f"POST failed: {e}"

    post_thread = threading.Thread(target=post_weight_update, daemon=False)
    post_thread.start()
    time.sleep(0.5)

    for i, (name, tensor) in enumerate(perturbed_state_dict.items()):
        if tensor.device.type != "cuda":
            tensor = tensor.cuda(0)
        dist.broadcast(tensor, src=0, group=process_group)
        if (i + 1) % 50 == 0:
            print(
                f"[Trainer] Broadcasted {i+1}/{len(perturbed_state_dict)} perturbed parameters"
            )

    post_thread.join(timeout=60)
    if post_result["error"]:
        raise RuntimeError(f"Weight update POST failed: {post_result['error']}")

    print("[Trainer] Perturbed weights broadcast complete (2nd time)")

    # Cleanup
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
