"""Helper functions for Fast-LLM weight broadcast testing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from trainer_test_utils import (
    _load_state_dict,
    _create_perturbed_state_dict,
    _init_actor_process_group,
    _wait_for_servers_ready,
)


def timed_broadcast_fast_llm(
    init_method: str,
    model_name: str,
    server_urls: list,
    redis_host: str = "localhost",
    redis_port: int = 6379,
    world_size: int = 2,
):
    """Timed broadcast using Fast-LLM protocol: perturbed → original → perturbed with delays.

    This simulates Fast-LLM's weight broadcast protocol where weight updates are signaled
    via Redis stream and broadcast using broadcast_object_list + broadcast.

    Pattern: original (server default) → perturbed → original → perturbed

    Args:
        init_method: Distributed init method
        model_name: Model name to load
        server_urls: Base URLs of vLLM server(s) (for health check only)
        redis_host: Redis host address
        redis_port: Redis port number
        world_size: Total NCCL world size (trainer rank 0 + all vLLM workers)
    """
    import torch
    import torch.distributed as dist
    import time
    import redis
    import orjson

    # Initialize process group
    process_group = _init_actor_process_group(init_method, rank=0, world_size=world_size)

    # Connect to Redis
    print(f"[Trainer] Connecting to Redis at {redis_host}:{redis_port}")
    r = redis.Redis(host=redis_host, port=redis_port)
    stream_key = "fast_llm_events"
    payload_key = "event"
    print(f"[Trainer] Connected to Redis, will write to stream '{stream_key}'")

    _wait_for_servers_ready(server_urls, extra_wait_secs=15)

    # Load weights
    print(f"[Trainer] Loading original weights from {model_name}")
    original_state_dict, _ = _load_state_dict(model_name)
    perturbed_state_dict = _create_perturbed_state_dict(original_state_dict)

    # Helper function to broadcast weights using Fast-LLM protocol
    def broadcast_weights_fast_llm(state_dict, step):
        """Broadcast weights using Fast-LLM protocol.

        Protocol:
        1. Send Redis event: {type: "weights_ready", step: N}
        2. For each parameter:
           - broadcast_object_list([(shard_name, layer_name, shape, dtype)])
           - broadcast(tensor)
        3. Send end signal: broadcast_object_list([None])
        """
        # Send Redis stream event
        event = {"type": "weights_ready", "step": step}
        r.xadd(stream_key, {payload_key: orjson.dumps(event)})
        print(f"[Trainer] Sent Redis event to '{stream_key}': {event}")

        # Broadcast each parameter
        for i, (name, tensor) in enumerate(state_dict.items()):
            if tensor.device.type != "cuda":
                tensor = tensor.cuda(0)

            shard_name = ""
            layer_name = name

            # Broadcast metadata
            meta = [(shard_name, layer_name, list(tensor.shape), str(tensor.dtype))]
            dist.broadcast_object_list(meta, src=0, group=process_group)

            # Broadcast tensor
            dist.broadcast(tensor, src=0, group=process_group)

            if (i + 1) % 50 == 0:
                print(f"[Trainer] Broadcasted {i+1}/{len(state_dict)} parameters")

        # Send end signal
        dist.broadcast_object_list([None], src=0, group=process_group)
        print(f"[Trainer] Sent end signal, broadcast complete")

    # Broadcast 1: Perturbed weights
    print(f"[Trainer] Broadcasting {len(perturbed_state_dict)} perturbed parameters")
    broadcast_weights_fast_llm(perturbed_state_dict, step=1)
    print("[Trainer] Perturbed weights broadcast complete")

    print("[Trainer] Waiting 5 seconds before broadcasting original weights...")
    time.sleep(5)

    # Broadcast 2: Original weights
    print(f"[Trainer] Broadcasting {len(original_state_dict)} original parameters")
    broadcast_weights_fast_llm(original_state_dict, step=2)
    print("[Trainer] Original weights broadcast complete")

    print("[Trainer] Waiting 5 seconds before broadcasting perturbed weights again...")
    time.sleep(5)

    # Broadcast 3: Perturbed weights again (same as first)
    print(f"[Trainer] Broadcasting {len(perturbed_state_dict)} perturbed parameters (2nd time)")
    broadcast_weights_fast_llm(perturbed_state_dict, step=3)
    print("[Trainer] Perturbed weights broadcast complete (2nd time)")

    # Wait to allow generation with the last broadcast before tearing down
    print("[Trainer] Waiting 5 seconds for generation with final weights...")
    time.sleep(5)

    # Signal training is finished so vLLM workers destroy their side of the process group
    print("[Trainer] Sending training_finished signal...")
    r.xadd(stream_key, {payload_key: orjson.dumps({"type": "training_finished"})})

    # Cleanup — destroy_process_group now resolves because vLLM workers respond to training_finished
    r.close()
    dist.destroy_process_group(process_group)
    print("[Trainer] Redis connection closed, process group destroyed, exiting")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast-LLM trainer helper")
    parser.add_argument("--init-method", required=True, help="Distributed init method")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--server-urls", nargs="+", required=True, help="Server URL(s)")
    parser.add_argument("--world-size", type=int, default=2, help="Total distributed world size")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")

    args = parser.parse_args()

    timed_broadcast_fast_llm(
        init_method=args.init_method,
        model_name=args.model,
        server_urls=args.server_urls,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        world_size=args.world_size,
    )
