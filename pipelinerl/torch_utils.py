from datetime import timedelta
import os
from urllib.parse import urlparse
from typing import Any, Optional, Union
from torch.distributed import TCPStore
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_extra_process_group(
    backend: Union[str, Backend] = None,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
    world_size: int = -1,
    rank: int = -1,
    store: Optional[Store] = None,
    group_name: str = None,
    pg_options: Optional[Any] = None,
):
    prev_use_agent_store = os.environ.get("TORCHELASTIC_USE_AGENT_STORE")
    if prev_use_agent_store:
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "0"
    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None and init_method and init_method.startswith("tcp://"):
        if world_size <= 0 or rank < 0:
            raise ValueError("world_size and rank must be set when using tcp:// init_method")
        parsed = urlparse(init_method)
        if parsed.hostname is None or parsed.port is None:
            raise ValueError(f"Invalid tcp init_method: {init_method}")
        store = TCPStore(
            host_name=parsed.hostname,
            port=parsed.port,
            world_size=world_size,
            is_master=rank == 0,
            timeout=timeout,
        )
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        backend_options=pg_options,
        timeout=timeout,
    )

    if prev_use_agent_store is None:
        os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
    else:
        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = prev_use_agent_store

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg
