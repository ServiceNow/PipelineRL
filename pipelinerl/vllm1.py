import asyncio
import inspect
import logging
import os
import signal
import time
import torch
import uvloop
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.utils.system_utils import set_ulimit
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    create_server_socket,
    build_app,
    init_app_state,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm._version import version
from vllm.usage.usage_lib import UsageContext
from vllm.config import ModelConfig
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


from pipelinerl.finetune_loop import WeightUpdateRequest, ParameterInfo
from pipelinerl.vllm_quantization import string_to_dtype  # reuse mapping
from typing import Any, Protocol, runtime_checkable, Dict, Optional
from fastapi import BackgroundTasks
import pipelinerl.torch_utils
from pipelinerl.torch_utils import stateless_init_process_group
import pipelinerl.vllm_quantization  # Register bf16_last_layer_fp32 quantization config
from vllm.distributed import cleanup_dist_env_and_memory
from contextlib import asynccontextmanager

try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
except ModuleNotFoundError:
    from vllm.tool_parsers import ToolParserManager

logger = logging.getLogger(__name__)
# configure this logger individually, in order to avoid messing
# with the default vllm logger configuration
# Check environment variable to enable DEBUG logging (for tests)
import os

log_level = logging.DEBUG if os.getenv("PIPELINERL_DEBUG") else logging.INFO
logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter(
    "[%(asctime)s] [VLLM-%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
handler.setFormatter(formatter)
logger.addHandler(handler)
# Prevent propagation to vLLM's loggers to avoid double logging
logger.propagate = False


@runtime_checkable
class LikeWorker(Protocol):
    rank: int
    local_rank: int
    device: torch.device
    model_runner: GPUModelRunner
    pg_rank: int
    model_update_group: Any
    model_config: ModelConfig


class WorkerExtension:
    def is_extension_loaded(self: LikeWorker) -> int:
        """Simple method to verify the extension is loaded on workers.

        Returns:
            PID of the worker process
        """
        import os

        return os.getpid()

    def init_actor_update_group(
        self: LikeWorker,
        actor_idx: int,
        actor_ngpus: int,
        weight_update_group_init_method: str,
        weight_update_group_world_size: int,
        weight_update_mode: str = "http",
    ):
        self.pg_rank = 1 + actor_idx * actor_ngpus + self.rank
        # log all you know
        prefix = "[INIT_ACTOR_UPDATE_GROUP]: "
        logger.info(
            prefix
            + f"Actor index: {actor_idx}, actor ngpus: {actor_ngpus}, rank: {self.rank}, pg_rank: {self.pg_rank}"
        )
        logger.info(
            prefix
            + f"Weight update group init method: {weight_update_group_init_method}, world size: {weight_update_group_world_size}, mode: {weight_update_mode}"
        )

        batch_invariant_env = os.getenv("VLLM_BATCH_INVARIANT", "0")
        try:
            batch_invariant_enabled = int(batch_invariant_env) != 0
        except ValueError:
            batch_invariant_enabled = False

        if batch_invariant_enabled:
            # vLLM batch_invariant mode sets restrictive NCCL env vars (single channel,
            # tree algo, simple proto, P2P disabled) that the trainer does not share.
            # Clear them so the weight-update NCCL comm matches trainer defaults.
            # Safe at tp=1 because no intra-engine NCCL comm has been created yet.
            for _k in (
                "NCCL_LAUNCH_MODE", "NCCL_COLLNET_ENABLE", "NCCL_NVLS_ENABLE",
                "NCCL_P2P_NET_DISABLE", "NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS",
                "NCCL_PROTO", "NCCL_ALGO", "NCCL_NTHREADS", "NCCL_SOCKET_NTHREADS",
            ):
                os.environ.pop(_k, None)

        if weight_update_mode == 'http':
            # HTTP mode uses vLLM's StatelessProcessGroup to match the trainer,
            # which in pipelinerl/finetune_loop.py uses torch_utils.stateless_init_process_group.
            self.model_update_group = stateless_init_process_group(
                init_method=weight_update_group_init_method,
                rank=self.pg_rank,
                world_size=weight_update_group_world_size,
                device=self.device,
            )
        else:
            from fast_llm.engine.distributed.config import DistributedBackend
            from fast_llm.engine.distributed.distributed import ProcessGroupPool

            self.model_update_group = ProcessGroupPool(
                rank=self.pg_rank,
                world_size=weight_update_group_world_size,
                local_world_size=1,
                init_method=weight_update_group_init_method,
                backend=DistributedBackend.nccl,
            ).get_process_group(range(weight_update_group_world_size), self.pg_rank)
        self._process_group_destroyed = False
        logger.info(prefix + "Actor update process group initialized")

    def destroy_actor_update_group(self: LikeWorker):
        self._process_group_destroyed = True
        if isinstance(self.model_update_group, torch.distributed.ProcessGroup):
            torch.distributed.destroy_process_group(self.model_update_group)
        elif hasattr(self.model_update_group, "shutdown"):
            self.model_update_group.shutdown()
        # StatelessProcessGroup has no shutdown method; rely on GC.

    def is_actor_update_group_destroyed(self: LikeWorker) -> bool:
        return getattr(self, "_process_group_destroyed", False)

    def receive_weight_update(self: LikeWorker, request_json: str):
        request = WeightUpdateRequest.model_validate_json(request_json)
        torch.cuda.synchronize(self.device)
        logger.info(
            f"Start receiving weight update: {len(request.parameters_info)} parameters"
        )
        expected_dtypes = (torch.bfloat16, torch.float32, torch.float16)

        for i, info in enumerate(request.parameters_info):
            logger.debug(
                f"[{i+1}/{len(request.parameters_info)}] Preparing to receive: {info.name}"
            )
            logger.debug(f"  - shape: {info.shape}, dtype: {info.dtype}")

            target_dtype = string_to_dtype(info.dtype)
            if target_dtype not in expected_dtypes:
                logger.warning(f"Unexpected dtype for {info.name}: {info.dtype}")

            logger.debug(f"  - Creating buffer for {info.name}")
            buffer = torch.empty(
                tuple(info.shape), dtype=target_dtype, device=self.device
            )
            logger.debug(
                f"  - Buffer created: shape={buffer.shape}, dtype={buffer.dtype}, device={buffer.device}"
            )

            logger.debug(f"  - Calling broadcast for {info.name}...")
            # StatelessProcessGroup exposes .broadcast(); torch.distributed.ProcessGroup
            # (fast-llm path) uses the functional torch.distributed.broadcast.
            if isinstance(self.model_update_group, torch.distributed.ProcessGroup):
                torch.distributed.broadcast(buffer, src=0, group=self.model_update_group)
            else:
                self.model_update_group.broadcast(buffer, src=0, stream=torch.cuda.current_stream())
            logger.debug(f"  - Broadcast received for {info.name}")

            logger.debug(f"  - Loading weights for {info.name}...")
            try:
                loaded_params = self.model_runner.model.load_weights(weights=[(info.name, buffer)])  # type: ignore
                if len(loaded_params) == 0:
                    # Parameter doesn't exist in vLLM model - this is an error
                    logger.error(f"  - ERROR: {info.name} not found in vLLM model")
                    raise ValueError(
                        f"Parameter {info.name} not found in vLLM model state dict"
                    )
                elif len(loaded_params) == 1:
                    logger.debug(f"  - Weights loaded for {info.name}")
                else:
                    logger.error(
                        f"  - ERROR: load_weights returned {len(loaded_params)} params for {info.name}"
                    )
                    raise ValueError(
                        f"Unexpected number of parameters loaded for {info.name}"
                    )
            except Exception as e:
                logger.error(f"  - ERROR loading weights for {info.name}: {e}")
                raise

            if (i + 1) % 10 == 0:
                logger.info(f"Received {i+1}/{len(request.parameters_info)} parameters")

        pipelinerl.vllm_quantization.invalidate_fp32_cache()
        logger.info("Weight update received - all parameters processed")

    def receive_weight_update_fast_llm(self: LikeWorker):
        """Receive weight update via Fast-LLM broadcast protocol.

        Called via collective_rpc_async from the main-process monitoring thread,
        so it runs in each worker's main thread — serialized with inference,
        identical concurrency model to receive_weight_update (HTTP path).

        Protocol:
        1. Loop: receive metadata via broadcast_object_list
        2. Receive tensor via broadcast
        3. Call model.load_weights() for each parameter
        4. Exit when metadata is [None] (end signal)
        """
        torch.cuda.synchronize(self.device)
        logger.info(f"[Worker rank={self.rank}] Start receiving Fast-LLM weight update")

        expected_dtypes = (torch.bfloat16, torch.float32, torch.float16)
        param_count = 0

        from fast_llm.core.distributed import broadcast as _broadcast, broadcast_object as _broadcast_object

        while True:
            # Receive metadata
            logger.debug(f"[Worker rank={self.rank}] Waiting for metadata broadcast...")
            meta = _broadcast_object(None, self.model_update_group, src=0)
            logger.debug(f"[Worker rank={self.rank}] Received metadata: {meta}")

            # Check for end signal
            if meta is None:
                logger.info(
                    f"[Worker rank={self.rank}] Received end signal, finished receiving {param_count} parameters"
                )
                break

            # Parse metadata: (shard_name, layer_name, shape, dtype)
            # shard_name is a category label ("weights", "grads", etc.), not part of the HF param name
            shard_name, layer_name, shape, dtype = meta
            param_name = layer_name

            # Convert dtype to torch dtype
            target_dtype = string_to_dtype(str(dtype))

            # Allocate buffer and receive tensor (must happen for every broadcast to stay in sync)
            buffer = torch.empty(tuple(shape), dtype=target_dtype, device=self.device)
            _broadcast(buffer, 0, self.model_update_group)

            # Only load weight shards (skip grads, optimizer state, etc.)
            if shard_name != "weights":
                continue

            param_count += 1
            logger.debug(
                f"[{param_count}] Receiving: {param_name}, shape={shape}, dtype={dtype}"
            )

            if target_dtype not in expected_dtypes:
                logger.warning(f"Unexpected dtype for {param_name}: {dtype}")

            logger.debug(f"[{param_count}] Received tensor for {param_name}")

            # Load weights
            try:
                loaded_params = self.model_runner.model.load_weights(
                    weights=[(param_name, buffer)]
                )
                if len(loaded_params) == 0:
                    logger.error(f"ERROR: {param_name} not found in vLLM model")
                    raise ValueError(
                        f"Parameter {param_name} not found in vLLM model state dict"
                    )
                elif len(loaded_params) == 1:
                    logger.debug(f"[{param_count}] Loaded {param_name}")
                else:
                    logger.error(
                        f"ERROR: load_weights returned {len(loaded_params)} params for {param_name}"
                    )
                    raise ValueError(
                        f"Unexpected number of parameters loaded for {param_name}"
                    )
            except Exception as e:
                logger.error(f"ERROR loading {param_name}: {e!r}", exc_info=True)
                raise

            if param_count % 10 == 0:
                logger.info(f"[Worker rank={self.rank}] Received {param_count} parameters")

        pipelinerl.vllm_quantization.invalidate_fp32_cache()
        logger.info(
            f"[Worker rank={self.rank}] Fast-LLM weight update complete - {param_count} parameters processed"
        )

    def close_communicator(self):
        """Closes the communicator when weight synchronization is no longer needed."""
        if hasattr(self, "model_update_group") and self.model_update_group is not None:
            del self.model_update_group
            self.model_update_group = None
            logger.info("Weight update communicator closed")


async def _pause_generation(engine: AsyncLLM) -> None:
    """Pause generation without draining in-flight requests.

    Adapts to the installed vLLM version at runtime: newer builds expose
    pause_generation(mode=) while older ones use wait_for_inflight_requests=.
    """
    if 'mode' in inspect.signature(engine.pause_generation).parameters:
        await engine.pause_generation(mode="keep", clear_cache=False)
    else:
        await engine.pause_generation(wait_for_inflight_requests=False, clear_cache=False)


class EngineManager:
    def __init__(self, args, engine: AsyncLLM, engine_config: Any):
        self.args = args
        self.engine = engine
        self.engine_config = engine_config
        self.update_lock = asyncio.Lock()

    async def is_extension_loaded(self):
        return await self.engine.engine_core.collective_rpc_async(
            "is_extension_loaded",
            args=(),
        )

    async def init_actor_update_group(self):
        await self.engine.engine_core.collective_rpc_async(
            "init_actor_update_group",
            args=(
                self.args.actor_llm_idx,
                torch.cuda.device_count(),
                self.args.weight_update_group_init_method,
                self.args.weight_update_group_world_size,
                getattr(self.args, "weight_update_mode", "http"),
            ),
        )

    async def destroy_actor_update_group(self):
        await self.engine.engine_core.collective_rpc_async(
            "destroy_actor_update_group",
            args=(),
        )

    async def is_actor_update_group_destroyed(self) -> bool:
        results = await self.engine.engine_core.collective_rpc_async(
            "is_actor_update_group_destroyed",
            args=(),
        )
        return all(results)

    async def receive_weight_update(self, request: WeightUpdateRequest):
        async with self.update_lock:
            version = getattr(request, "version", "unknown")
            pause_started_at = time.perf_counter()
            logger.info(f"Pausing generation for weight update version={version}")
            await _pause_generation(self.engine)
            logger.info(
                f"Generation paused for weight update version={version} "
                f"in {time.perf_counter() - pause_started_at:.3f}s"
            )
            try:
                update_started_at = time.perf_counter()
                logger.info(f"Starting weight update version={version}")
                await self.engine.engine_core.collective_rpc_async(
                    "receive_weight_update", args=(request.model_dump_json(),)
                )
                logger.info(
                    f"Weight update processed version={version} "
                    f"in {time.perf_counter() - update_started_at:.3f}s"
                )
            finally:
                resume_started_at = time.perf_counter()
                logger.info(f"Resuming generation after weight update version={version}")
                await self.engine.resume_generation()
                logger.info(
                    f"Generation resumed after weight update version={version} "
                    f"in {time.perf_counter() - resume_started_at:.3f}s"
                )

    async def close_communicator(self):
        """Closes the communicator when weight synchronization is no longer needed."""
        await self.engine.engine_core.collective_rpc_async("close_communicator")

    async def init_fast_llm_receiver(self):
        """Store Redis connection info for the main-process monitoring thread."""
        self._redis_host = self.args.redis_host
        self._redis_port = self.args.redis_port
        logger.info(
            f"Fast-LLM receiver initialized (Redis {self._redis_host}:{self._redis_port})"
        )

    async def receive_weight_update_fast_llm(self):
        """Run a fast-llm broadcast weight update paused-for-the-duration.

        Pause/resume wraps the collective RPC symmetrically with the HTTP path
        so that in-flight generation cannot interleave with a mid-broadcast
        parameter swap (the source of logprob drift PR #137 closed).

        NOTE: this must NOT be used for the very first weights_ready event
        after process startup, because at that point the actor has not yet
        begun issuing rollouts (it's blocked in wait_for_model_version) and
        pause_generation will deadlock waiting for an in-flight-decode state
        that never arrives. The monitor thread gates this accordingly.
        """
        async with self.update_lock:
            pause_started_at = time.perf_counter()
            logger.info("Pausing generation for fast-llm weight update")
            await _pause_generation(self.engine)
            logger.info(
                f"Generation paused for fast-llm weight update "
                f"in {time.perf_counter() - pause_started_at:.3f}s"
            )
            try:
                update_started_at = time.perf_counter()
                await self.engine.engine_core.collective_rpc_async(
                    "receive_weight_update_fast_llm", args=()
                )
                logger.info(
                    f"Fast-llm weight update processed "
                    f"in {time.perf_counter() - update_started_at:.3f}s"
                )
            finally:
                resume_started_at = time.perf_counter()
                logger.info("Resuming generation after fast-llm weight update")
                await self.engine.resume_generation()
                logger.info(
                    f"Generation resumed after fast-llm weight update "
                    f"in {time.perf_counter() - resume_started_at:.3f}s"
                )

    async def start_fast_llm_monitoring(self):
        """Start a single Redis monitoring thread in the main process.

        When weights_ready arrives the thread calls
        collective_rpc_async("receive_weight_update_fast_llm") which runs in
        each worker's main thread — blocking inference during the update,
        identical concurrency to the HTTP path.  training_finished is handled
        the same way via destroy_actor_update_group().
        """
        import asyncio
        import threading

        self._fast_llm_stop_event = threading.Event()
        loop = asyncio.get_event_loop()

        def monitor_redis_stream():
            import redis
            import orjson
            import time

            r = redis.Redis(host=self._redis_host, port=self._redis_port)
            stream_key = "fast_llm_events"
            payload_key = b"event"
            last_id = "0-0"
            # First weights_ready event since this vLLM process started is the
            # initial broadcast (step can be 0 on fresh start or k>0 on resume).
            # Actor is still blocked in wait_for_model_version at this point, so
            # vLLM has zero in-flight requests — pause_generation would deadlock.
            # Take the raw RPC path for the first event; wrap with pause/resume
            # thereafter, matching PR #137's guard against mid-rollout weight swaps.
            first_weights_ready_seen = False

            logger.info("[FastLLM] Main-process Redis monitoring started")

            while not self._fast_llm_stop_event.is_set():
                try:
                    result = r.xread({stream_key: last_id}, count=1, block=1000)
                    if not result:
                        continue

                    for _stream_name, messages in result:
                        for msg_id, msg_data in messages:
                            last_id = msg_id

                            if payload_key not in msg_data:
                                logger.warning(
                                    f"[FastLLM] Event missing 'event' field: {msg_data}"
                                )
                                continue

                            try:
                                event = orjson.loads(msg_data[payload_key])
                            except Exception as e:
                                logger.error(f"[FastLLM] Failed to parse event: {e}")
                                continue

                            event_type = event.get("type")
                            step = event.get("step")

                            if event_type == "weights_ready":
                                if not first_weights_ready_seen:
                                    logger.info(
                                        f"[FastLLM] weights_ready step={step} (initial broadcast — no pause wrap)"
                                    )
                                    coro = self.engine.engine_core.collective_rpc_async(
                                        "receive_weight_update_fast_llm", args=()
                                    )
                                    first_weights_ready_seen = True
                                else:
                                    logger.info(
                                        f"[FastLLM] weights_ready step={step}, dispatching to workers"
                                    )
                                    coro = self.receive_weight_update_fast_llm()
                                try:
                                    future = asyncio.run_coroutine_threadsafe(coro, loop)
                                    future.result()
                                    logger.info(
                                        f"[FastLLM] Weight update complete: step={step}"
                                    )
                                except Exception as e:
                                    logger.error(
                                        f"[FastLLM] Error receiving weight update: {e}"
                                    )

                            elif event_type == "training_finished":
                                logger.info(
                                    "[FastLLM] training_finished received, destroying process group"
                                )
                                try:
                                    future = asyncio.run_coroutine_threadsafe(
                                        self.destroy_actor_update_group(), loop
                                    )
                                    future.result()
                                except Exception as e:
                                    logger.error(
                                        f"[FastLLM] Error destroying process group: {e}"
                                    )
                                self._fast_llm_stop_event.set()

                except Exception as e:
                    logger.error(f"[FastLLM] Error in Redis monitor: {e}")
                    if not self._fast_llm_stop_event.is_set():
                        time.sleep(1)

            logger.info("[FastLLM] Main-process Redis monitoring stopped")
            r.close()

        self._fast_llm_monitor_thread = threading.Thread(
            target=monitor_redis_stream,
            daemon=True,
            name="FastLLMMonitor",
        )
        self._fast_llm_monitor_thread.start()
        logger.info("[FastLLM] Main-process monitoring thread started")

    async def stop_fast_llm_monitoring(self):
        """Stop the main-process Fast-LLM monitoring thread."""
        if not hasattr(self, "_fast_llm_stop_event"):
            return
        if not self._fast_llm_stop_event.is_set():
            logger.warning("[FastLLM] training_finished was not received; forcing stop")
            self._fast_llm_stop_event.set()
        if hasattr(self, "_fast_llm_monitor_thread"):
            self._fast_llm_monitor_thread.join(timeout=5)
            logger.info("[FastLLM] Main-process monitoring thread stopped")

    @asynccontextmanager
    @staticmethod
    async def create_engine(
        args: Any,
        cleanup: bool = True,
    ):
        """Create vLLM AsyncLLM engine with automatic cleanup.

        This is an async context manager that ensures proper engine lifecycle
        management with automatic cleanup on exit.

        Usage:
            # Simple usage (tests)
            async with create_engine(args) as (engine, engine_config):
                # Use engine for generation
                async for output in engine.generate(...):
                    ...
            # Automatic cleanup happens here

            # Or unpack only what you need
            async with create_engine(args) as (engine, _):
                # Use engine, ignore config
                ...

            # Server usage (no cleanup)
            async with create_engine(args, cleanup=False) as (engine, engine_config):
                # Use both engine and config
                await init_app_state(engine, engine_config, ...)
                ...

        Args:
            args: Arguments object with vLLM engine configuration.
                Must be compatible with AsyncEngineArgs.from_cli_args().
                Required attributes: model
                Optional attributes: tensor_parallel_size, disable_log_stats,
                                    disable_log_requests, etc.
            cleanup: Whether to cleanup engine on exit (default: True).
                    Set to False for server usage where engine runs indefinitely.

        Yields:
            Tuple of (engine, engine_config):
                - engine: AsyncLLM engine instance
                - engine_config: VllmConfig for init_app_state
        """
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine_args.worker_extension_cls = "pipelinerl.vllm1.WorkerExtension"
        engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)

        logger.info(f"Creating vLLM engine with model={args.model}")
        engine = AsyncLLM.from_vllm_config(
            vllm_config=engine_config,
            usage_context=UsageContext.OPENAI_API_SERVER,
            disable_log_stats=engine_args.disable_log_stats,
            enable_log_requests=engine_args.enable_log_requests,
        )

        logger.info("vLLM engine created successfully")

        try:
            assert isinstance(engine.engine_core, AsyncMPClient)
            manager = EngineManager(args, engine, engine_config)
            weight_update_mode = getattr(args, "weight_update_mode", "http")
            if not args.disable_weight_updates:
                await manager.init_actor_update_group()

                # Initialize Fast-LLM mode if enabled
                if weight_update_mode == "fast-llm":
                    await manager.init_fast_llm_receiver()
                    await manager.start_fast_llm_monitoring()
                    logger.info("Fast-LLM weight update mode enabled")

            yield manager
        finally:
            if not args.disable_weight_updates:
                # Stop Fast-LLM monitoring if enabled
                if weight_update_mode == "fast-llm":
                    await manager.stop_fast_llm_monitoring()

                if not await manager.is_actor_update_group_destroyed():
                    logger.warning(
                        "training_finished was not called before shutdown; "
                        "NCCL process group was not destroyed — potential resource leak"
                    )
            if cleanup:
                logger.info("Cleaning up vLLM engine")
                # Clear manager reference to engine first
                manager.engine = None
                manager.engine_config = None
                # Delete engine and force immediate garbage collection
                del engine
                del manager
                import gc

                gc.collect()
                cleanup_dist_env_and_memory()


async def run_server(args, **uvicorn_kwargs) -> None:
    # COPIED FROM vllm/entrypoints/openai/api_server.py, vllm version 0.6.6.post1
    logger.info(f"vLLM API server version {version}")
    logger.info(f"args: {args}")

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    if hasattr(ToolParserManager, "list_registered"):
        valid_tool_parses = ToolParserManager.list_registered()
    else:
        valid_tool_parses = list(ToolParserManager.tool_parsers.keys())
    if args.enable_auto_tool_choice and args.tool_call_parser not in valid_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} (chose from {{ {','.join(valid_tool_parses)} }})"
        )

    # workaround to make sure that we bind the port before the engine is set up.
    # This avoids race conditions with ray.
    # see https://github.com/vllm-project/vllm/issues/8204
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)

    # workaround to avoid footguns where uvicorn drops requests with too
    # many concurrent requests active
    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    # Create engine (cleanup=False since server runs indefinitely)
    async with EngineManager.create_engine(args, cleanup=False) as manager:
        # Run HTTP server
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)
        # vLLM 0.18.1+ requires supported_tasks to build the app and app state;
        # older vllm (e.g. 0.14.x) has 1-arg build_app / 3-arg init_app_state.
        import inspect as _inspect
        _build_app_params = _inspect.signature(build_app).parameters
        if "supported_tasks" in _build_app_params and hasattr(manager.engine, "get_supported_tasks"):
            supported_tasks = await manager.engine.get_supported_tasks()
            logger.info(f"Supported tasks: {supported_tasks}")
            app = build_app(args, supported_tasks)
        else:
            supported_tasks = None
            app = build_app(args)

        # Register HTTP endpoint only if using HTTP mode
        if getattr(args, "weight_update_mode", "http") == "http":
            @app.post("/receive_weight_update")
            async def _receive_weight_update(request: WeightUpdateRequest):
                await manager.receive_weight_update(request)
                return {"status": "ok"}

            @app.post("/training_finished")
            async def _training_finished(background_tasks: BackgroundTasks):
                logger.info("Received /training_finished, scheduling NCCL process group teardown")
                background_tasks.add_task(manager.destroy_actor_update_group)
                return {"status": "ok"}

            logger.info("HTTP weight update endpoint registered")
        else:
            logger.info("Fast-LLM mode: using Redis stream (no HTTP endpoint registered)")

        if "supported_tasks" in _inspect.signature(init_app_state).parameters:
            await init_app_state(manager.engine, app.state, args, supported_tasks)
        else:
            await init_app_state(manager.engine, app.state, args)
        shutdown_task = await serve_http(
            app,
            sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            # increase timeout
            timeout_keep_alive=60,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

        # NB: Await server shutdown only after the backend context is exited
        await shutdown_task

        sock.close()

        # NOTE: weight-broadcast process group teardown must be coordinated with the trainer —
        # the trainer sends training_finished, then the engine manager destroys its side here.


def run_llm():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--disable-weight-updates",
        action="store_true",
        help="Whether to receive weight updates from the trainer",
    )
    parser.add_argument(
        "--actor-llm-idx",
        type=int,
    )
    parser.add_argument(
        "--weight-update-group-init-method",
        type=str,
    )
    parser.add_argument(
        "--weight-update-group-world-size",
        type=int,
    )
    parser.add_argument(
        "--weight-update-mode",
        type=str,
        choices=["http", "fast-llm"],
        default="http",
        help="Weight update protocol: 'http' (HTTP POST) or 'fast-llm' (Redis+broadcast)",
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default="localhost",
        help="Redis host for Fast-LLM mode",
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        default=6379,
        help="Redis port for Fast-LLM mode",
    )
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
