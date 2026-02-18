import logging
import signal
import torch
import uvloop
from vllm.utils.system_utils import set_ulimit
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import (
    make_arg_parser,
    validate_parsed_serve_args,
)
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    run_server,
    create_server_socket,
    build_app,
    init_app_state,
)
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.tool_parsers import ToolParserManager
from vllm._version import version
from vllm.usage.usage_lib import UsageContext
from vllm.config import ModelConfig
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


from pipelinerl.finetune_loop import WeightUpdateRequest, ParameterInfo
from pipelinerl.vllm_quantization import string_to_dtype  # reuse mapping
from typing import Any, Protocol, runtime_checkable, Dict, Optional
import pipelinerl.torch_utils
import pipelinerl.vllm_quantization  # Register bf16_last_layer_fp32 quantization config
from vllm.distributed import cleanup_dist_env_and_memory
from contextlib import asynccontextmanager

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
    process_group: Any
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
            + f"Weight update group init method: {weight_update_group_init_method}, world size: {weight_update_group_world_size}"
        )
        self.process_group = pipelinerl.torch_utils.init_extra_process_group(
            group_name="actor",
            backend="nccl",
            init_method=weight_update_group_init_method,
            rank=self.pg_rank,
            world_size=weight_update_group_world_size,
        )

    def destroy_actor_update_group(self: LikeWorker):
        torch.distributed.destroy_process_group(self.process_group)

    def receive_weight_update(self: LikeWorker, request: WeightUpdateRequest):
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
            torch.distributed.broadcast(buffer, src=0, group=self.process_group)
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

    def init_fast_llm_receiver(
        self: LikeWorker,
        redis_host: str,
        redis_port: int,
    ):
        """Initialize Fast-LLM weight receiver (called once at startup).

        This method:
        1. Stores Redis connection info
        2. Sets up threading infrastructure
        3. Does NOT start monitoring thread (that's managed by EngineManager)
        """
        import threading

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.fast_llm_stop_event = threading.Event()
        logger.info(
            f"[Worker rank={self.rank}] Fast-LLM receiver initialized with Redis {redis_host}:{redis_port}"
        )

    def start_fast_llm_monitoring(self: LikeWorker):
        """Start background thread to monitor Redis stream.

        This thread:
        1. Connects to Redis stream "fast_llm_events"
        2. Listens for {type: "weights_ready", step: N} events
        3. On event, triggers receive_weight_update_fast_llm()
        4. Runs until stop_event is set
        """
        import threading
        import time

        def monitor_redis_stream():
            import redis
            import orjson

            r = redis.Redis(host=self.redis_host, port=self.redis_port)
            stream_key = "fast_llm_events"
            payload_key = b"event"
            last_id = "0-0"

            logger.info(f"[Worker rank={self.rank}] Starting Redis stream monitoring")

            while not self.fast_llm_stop_event.is_set():
                try:
                    # Non-blocking read with 1s timeout
                    result = r.xread({stream_key: last_id}, count=1, block=1000)

                    if not result:
                        continue

                    for stream_name, messages in result:
                        for msg_id, msg_data in messages:
                            last_id = msg_id

                            if payload_key not in msg_data:
                                logger.warning(
                                    f"[Worker rank={self.rank}] Event missing 'event' field: {msg_data}"
                                )
                                continue

                            try:
                                event = orjson.loads(msg_data[payload_key])
                            except Exception as e:
                                logger.error(
                                    f"[Worker rank={self.rank}] Failed to parse event: {e}"
                                )
                                continue

                            event_type = event.get("type")
                            step = event.get("step")

                            if event_type == "weights_ready":
                                logger.info(
                                    f"[Worker rank={self.rank}] Received weights_ready event: step={step}"
                                )
                                # Call receive_weight_update_fast_llm directly (runs in this thread)
                                try:
                                    self.receive_weight_update_fast_llm()
                                except Exception as e:
                                    logger.error(
                                        f"[Worker rank={self.rank}] Error receiving Fast-LLM weight update: {e}"
                                    )
                            elif event_type == "training_finished":
                                logger.info(
                                    f"[Worker rank={self.rank}] Received training_finished event"
                                )

                except Exception as e:
                    logger.error(f"[Worker rank={self.rank}] Error in Redis monitor: {e}")
                    if not self.fast_llm_stop_event.is_set():
                        time.sleep(1)  # Avoid tight loop on error

            logger.info(f"[Worker rank={self.rank}] Redis monitoring stopped")
            r.close()

        import threading
        self.fast_llm_monitor_thread = threading.Thread(
            target=monitor_redis_stream,
            daemon=True,
            name=f"FastLLMMonitor-Rank{self.rank}",
        )
        self.fast_llm_monitor_thread.start()
        logger.info(f"[Worker rank={self.rank}] Fast-LLM monitoring thread started")

    def stop_fast_llm_monitoring(self: LikeWorker):
        """Stop the Fast-LLM monitoring thread."""
        if hasattr(self, "fast_llm_stop_event"):
            logger.info(f"[Worker rank={self.rank}] Stopping Fast-LLM monitoring")
            self.fast_llm_stop_event.set()
            if hasattr(self, "fast_llm_monitor_thread"):
                self.fast_llm_monitor_thread.join(timeout=5)
                logger.info(f"[Worker rank={self.rank}] Fast-LLM monitoring stopped")

    def receive_weight_update_fast_llm(self: LikeWorker):
        """Receive weight update via Fast-LLM broadcast protocol.

        This method:
        1. Loops receiving metadata via broadcast_object_list
        2. Receives tensor via broadcast
        3. Calls model.load_weights() for each parameter
        4. Exits when metadata is [None] (end signal)

        NOTE: This is called from the monitoring thread.
        """
        torch.cuda.synchronize(self.device)
        logger.info(f"[Worker rank={self.rank}] Start receiving Fast-LLM weight update")

        expected_dtypes = (torch.bfloat16, torch.float32, torch.float16)
        param_count = 0

        while True:
            # Receive metadata
            meta = [None]
            logger.debug(f"[Worker rank={self.rank}] Waiting for metadata broadcast...")
            torch.distributed.broadcast_object_list(
                meta, group=self.process_group, src=0
            )
            logger.debug(f"[Worker rank={self.rank}] Received metadata: {meta}")

            # Check for end signal
            if meta[0] is None:
                logger.info(
                    f"[Worker rank={self.rank}] Received end signal, finished receiving {param_count} parameters"
                )
                break

            # Parse metadata: (shard_name, layer_name, shape, dtype)
            shard_name, layer_name, shape, dtype = meta[0]
            param_name = f"{shard_name}.{layer_name}" if shard_name else layer_name
            param_count += 1

            logger.debug(
                f"[{param_count}] Receiving: {param_name}, shape={shape}, dtype={dtype}"
            )

            # Convert dtype to torch dtype
            target_dtype = string_to_dtype(str(dtype))
            if target_dtype not in expected_dtypes:
                logger.warning(f"Unexpected dtype for {param_name}: {dtype}")

            # Allocate buffer
            buffer = torch.empty(tuple(shape), dtype=target_dtype, device=self.device)

            # Receive tensor
            logger.debug(f"[{param_count}] Broadcasting tensor for {param_name}...")
            torch.distributed.broadcast(buffer, src=0, group=self.process_group)
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
                logger.error(f"ERROR loading {param_name}: {e}")
                raise

            if param_count % 10 == 0:
                logger.info(f"[Worker rank={self.rank}] Received {param_count} parameters")

        pipelinerl.vllm_quantization.invalidate_fp32_cache()
        logger.info(
            f"[Worker rank={self.rank}] Fast-LLM weight update complete - {param_count} parameters processed"
        )


class EngineManager:
    def __init__(self, args, engine: AsyncLLM, engine_config: Any):
        self.args = args
        self.engine = engine
        self.engine_config = engine_config

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
            ),
        )

    async def destroy_actor_update_group(self):
        await self.engine.engine_core.collective_rpc_async(
            "destroy_actor_update_group",
            args=(),
        )

    async def receive_weight_update(self, request: WeightUpdateRequest):
        await self.engine.engine_core.collective_rpc_async(
            "receive_weight_update", args=(request,)
        )
        logger.info("Weight update processed")

    async def init_fast_llm_receiver(self):
        """Initialize Fast-LLM receiver on all workers."""
        await self.engine.engine_core.collective_rpc_async(
            "init_fast_llm_receiver",
            args=(self.args.redis_host, self.args.redis_port),
        )
        logger.info("Fast-LLM receiver initialized on all workers")

    async def start_fast_llm_monitoring(self):
        """Start Fast-LLM monitoring threads on all workers."""
        await self.engine.engine_core.collective_rpc_async(
            "start_fast_llm_monitoring",
            args=(),
        )
        logger.info("Fast-LLM monitoring started on all workers")

    async def stop_fast_llm_monitoring(self):
        """Stop Fast-LLM monitoring threads on all workers."""
        await self.engine.engine_core.collective_rpc_async(
            "stop_fast_llm_monitoring",
            args=(),
        )
        logger.info("Fast-LLM monitoring stopped on all workers")

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
            if not args.disable_weight_updates:
                await manager.init_actor_update_group()

                # Initialize Fast-LLM mode if enabled
                if hasattr(args, 'weight_update_mode') and args.weight_update_mode == "fast-llm":
                    await manager.init_fast_llm_receiver()
                    await manager.start_fast_llm_monitoring()
                    logger.info("Fast-LLM weight update mode enabled")

            yield manager
        finally:
            if not args.disable_weight_updates:
                # Stop Fast-LLM monitoring if enabled
                if hasattr(args, 'weight_update_mode') and args.weight_update_mode == "fast-llm":
                    await manager.stop_fast_llm_monitoring()

                await manager.destroy_actor_update_group()
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

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} (chose from {{ {','.join(valide_tool_parses)} }})"
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
        app = build_app(args)

        # Register HTTP endpoint only if using HTTP mode
        if not hasattr(args, 'weight_update_mode') or args.weight_update_mode == "http":
            @app.post("/receive_weight_update")
            async def _receive_weight_update(request: WeightUpdateRequest):
                await manager.receive_weight_update(request)
                return {"status": "ok"}
            logger.info("HTTP weight update endpoint registered")
        else:
            logger.info("Fast-LLM mode: using Redis stream (no HTTP endpoint registered)")

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

        # TODO: proper cleanup
        # dist.destroy_process_group(actor_update_group)


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
