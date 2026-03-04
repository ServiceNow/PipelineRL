"""Shared utilities for trainer helper scripts (both HTTP and fast-llm variants)."""


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


def _init_actor_process_group(init_method: str, rank: int = 0, world_size: int = 2, group_name: str = "actor"):
    """Initialize the actor NCCL process group and return it."""
    import pipelinerl.torch_utils

    print(f"[Trainer] Initializing process group as rank {rank} (group_name={group_name!r})")
    process_group = pipelinerl.torch_utils.init_extra_process_group(
        group_name=group_name,
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    print("[Trainer] Process group initialized")
    return process_group


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


def _wait_for_servers_ready(server_urls: list, extra_wait_secs: int = 10):
    """Poll /health on each server until all respond 200, then sleep extra_wait_secs."""
    import time
    import requests

    for server_url in server_urls:
        print(f"[Trainer] Waiting for server {server_url} to be ready...")
        server_ready = False
        for i in range(120):  # up to 2 minutes
            try:
                resp = requests.get(f"{server_url}/health", timeout=1)
                if resp.status_code == 200:
                    server_ready = True
                    print(f"[Trainer] Server {server_url} is ready (took {i} seconds)")
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        if not server_ready:
            raise TimeoutError(f"Server {server_url} did not become ready within 2 minutes")

    if extra_wait_secs > 0:
        print(
            f"[Trainer] Waiting additional {extra_wait_secs} seconds for server(s) to fully initialize..."
        )
        time.sleep(extra_wait_secs)
