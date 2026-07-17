import asyncio
import logging
import os
import signal
import time
from contextlib import asynccontextmanager
from typing import Any, Protocol, runtime_checkable

import torch
import uvloop
from fastapi import BackgroundTasks
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

import pipelinerl.vllm_quantization  # Register bf16_last_layer_fp32 quantization config
from pipelinerl.finetune_loop import WeightUpdateRequest
from pipelinerl.state import read_fast_llm_events
from pipelinerl.torch_utils import stateless_init_process_group
from pipelinerl.vllm_quantization import string_to_dtype  # reuse mapping

try:
    from vllm.entrypoints.openai.tool_parsers import ToolParserManager
except ModuleNotFoundError:
    from vllm.tool_parsers import ToolParserManager

logger = logging.getLogger(__name__)
# Configure this logger with its own handler to avoid interfering with vLLM's logger configuration.
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [VLLM-%(levelname)s] %(message)s", datefmt="%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
# Prevent propagation to vLLM's loggers to avoid double logging
logger.propagate = False


# --- Per-token model version capture --------------------------------------------------
# The active weight version is a global, serialized quantity: it changes only inside a
# weight swap, while generation is paused (`_pause_generation`). We record the version
# active when the output processor commits each token, then ride it to the client inside
# the existing per-token `token` string of the chat logprobs
# (`token_id:<id>` -> `token_id:<id>:v<version>`), so no response-schema change is needed.
# Both seams run in the API-server process, alongside the version-tracking monitor thread.
#
# Any missing link (unpatched vLLM build, flat logprobs, a token absent from its own
# top-logprobs) simply omits the version; the consumer then falls back to the per-rollout
# version.
_current_model_version: dict[str, int | None] = {"value": None}
# Set once if a patched seam ever raises: the annotation hooks then no-op cheaply and the
# consumer falls back to the per-rollout version.
_version_tagging_disabled: dict[str, bool] = {"value": False}


def _set_current_model_version(version: int | None) -> None:
    _current_model_version["value"] = version


def _disable_version_tagging(context: str, error: Exception) -> None:
    if not _version_tagging_disabled["value"]:
        _version_tagging_disabled["value"] = True
        logger.warning(
            f"[FastLLM] Per-token model_version tagging disabled after error in {context}: {error!r}"
        )


def _install_model_version_patches() -> None:
    """Monkeypatch the vLLM v1 output path to tag generated tokens with the model version.

    Two seams, both in the API-server process:
      1. `LogprobsProcessor.update_from_output` — annotate each newly committed position's
         `Logprob` objects with `.version` = the version active at commit time.
      2. `OpenAIServingChat._create_chat_logprobs` — append `:v<version>` to each per-token
         `token` string, read back from the annotated `Logprob`.
    Idempotent, and defensive: a version mismatch that moves these seams disables per-token
    versions (consumer falls back to the per-rollout version) rather than crashing the server.
    """
    try:
        from vllm.v1.engine.logprobs import LogprobsProcessor

        try:
            # Newer vLLM keeps the chat serving class in a chat_completion package;
            # older builds define it in serving_chat.py.
            from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        except ImportError:
            from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    except ImportError as error:
        logger.warning(f"[FastLLM] Per-token model_version disabled (vLLM layout changed): {error!r}")
        return

    if getattr(LogprobsProcessor, "_pipelinerl_version_patched", False):
        return

    original_update_from_output = LogprobsProcessor.update_from_output

    def update_from_output(self, *args, **kwargs):
        previous_length = len(self.logprobs) if isinstance(self.logprobs, list) else None
        original_update_from_output(self, *args, **kwargs)
        if _version_tagging_disabled["value"]:
            return
        try:
            version = _current_model_version["value"]
            if version is None or previous_length is None or not isinstance(self.logprobs, list):
                return
            # Every logprob at a decode position shares that position's version; annotate all of
            # them so the serving layer reads the right value regardless of dict ordering.
            for position in self.logprobs[previous_length:]:
                if isinstance(position, dict):
                    for logprob in position.values():
                        logprob.version = version
        except Exception as error:
            # Best effort: never let version tagging break the output processor.
            _disable_version_tagging("output processor", error)

    original_create_chat_logprobs = OpenAIServingChat._create_chat_logprobs

    def _create_chat_logprobs(self, *args, **kwargs):
        result = original_create_chat_logprobs(self, *args, **kwargs)
        if _version_tagging_disabled["value"]:
            return result
        try:
            token_ids = args[0] if args else kwargs.get("token_ids")
            top_logprobs = args[1] if len(args) > 1 else kwargs.get("top_logprobs")
            content = getattr(result, "content", None)
            if content and token_ids is not None and top_logprobs is not None:
                for index, item in enumerate(content):
                    position = top_logprobs[index] if index < len(top_logprobs) else None
                    token_id = token_ids[index] if index < len(token_ids) else None
                    sampled = position.get(token_id) if position is not None and token_id is not None else None
                    version = getattr(sampled, "version", None)
                    # Only extend the `token_id:<id>` form; never mangle a decoded text token.
                    if version is not None and item.token.startswith("token_id:"):
                        item.token = f"{item.token}:v{version}"
        except Exception as error:
            # Best effort: never let version tagging break the response.
            _disable_version_tagging("chat logprobs", error)
        return result

    LogprobsProcessor.update_from_output = update_from_output
    OpenAIServingChat._create_chat_logprobs = _create_chat_logprobs
    LogprobsProcessor._pipelinerl_version_patched = True
    logger.info("[FastLLM] Per-token model_version patches installed")


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

        if weight_update_mode == "http":
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
        return self._process_group_destroyed

    def receive_weight_update(self: LikeWorker, request_json: str):
        request = WeightUpdateRequest.model_validate_json(request_json)
        torch.cuda.synchronize(self.device)
        logger.info(
            f"Start receiving weight update: {len(request.parameters_info)} parameters"
        )
        expected_dtypes = (torch.bfloat16, torch.float32, torch.float16)

        for i, info in enumerate(request.parameters_info):
            target_dtype = string_to_dtype(info.dtype)
            if target_dtype not in expected_dtypes:
                logger.warning(f"Unexpected dtype for {info.name}: {info.dtype}")

            buffer = torch.empty(
                tuple(info.shape), dtype=target_dtype, device=self.device
            )

            # StatelessProcessGroup exposes .broadcast(); torch.distributed.ProcessGroup
            # (fast-llm path) uses the functional torch.distributed.broadcast.
            if isinstance(self.model_update_group, torch.distributed.ProcessGroup):
                torch.distributed.broadcast(buffer, src=0, group=self.model_update_group)
            else:
                self.model_update_group.broadcast(buffer, src=0, stream=torch.cuda.current_stream())

            loaded_params = self.model_runner.model.load_weights(weights=[(info.name, buffer)])  # type: ignore
            if len(loaded_params) == 0:
                raise ValueError(
                    f"Parameter {info.name} not found in vLLM model state dict"
                )
            elif len(loaded_params) > 1:
                raise ValueError(
                    f"Unexpected number of parameters loaded for {info.name}"
                )

            if (i + 1) % 10 == 0:
                logger.info(f"Received {i + 1}/{len(request.parameters_info)} parameters")

        pipelinerl.vllm_quantization.invalidate_fp32_cache()
        logger.info("Weight update received - all parameters processed")

    def receive_weight_update_fast_llm(self: LikeWorker):
        """Receive weight update via Fast-LLM broadcast protocol.

        Called via collective_rpc_async from the main-process monitoring thread,
        so it runs in each worker's main thread — serialized with inference,
        identical concurrency model to receive_weight_update (HTTP path).

        Protocol:
        1. Loop: receive metadata via broadcast_object
        2. Receive tensor via broadcast
        3. Call model.load_weights() for each parameter
        4. Exit when metadata is None (end signal)
        """
        torch.cuda.synchronize(self.device)
        logger.info(f"[Worker rank={self.rank}] Start receiving Fast-LLM weight update")

        expected_dtypes = (torch.bfloat16, torch.float32, torch.float16)
        param_count = 0

        from fast_llm.core.distributed import broadcast as _broadcast, broadcast_object as _broadcast_object

        while True:
            meta = _broadcast_object(None, self.model_update_group, src=0)

            if meta is None:
                logger.info(
                    f"[Worker rank={self.rank}] Received end signal, finished receiving {param_count} parameters"
                )
                break

            # Parse metadata: (shard_name, param_name, shape, dtype)
            # shard_name is a category label ("weights", "grads", etc.), not part of the HF param name
            shard_name, param_name, shape, dtype = meta

            target_dtype = string_to_dtype(str(dtype))

            # Allocate buffer and receive tensor (must happen for every broadcast to stay in sync)
            buffer = torch.empty(tuple(shape), dtype=target_dtype, device=self.device)
            _broadcast(buffer, 0, self.model_update_group)

            # Only load weight shards (skip grads, optimizer state, etc.)
            if shard_name != "weights":
                continue

            param_count += 1
            if target_dtype not in expected_dtypes:
                logger.warning(f"Unexpected dtype for {param_name}: {dtype}")

            # Load weights
            loaded_params = self.model_runner.model.load_weights(
                weights=[(param_name, buffer)]
            )
            if len(loaded_params) == 0:
                raise ValueError(
                    f"Parameter {param_name} not found in vLLM model state dict"
                )
            elif len(loaded_params) > 1:
                raise ValueError(
                    f"Unexpected number of parameters loaded for {param_name}"
                )

            if param_count % 10 == 0:
                logger.info(f"[Worker rank={self.rank}] Received {param_count} parameters")

        pipelinerl.vllm_quantization.invalidate_fp32_cache()
        logger.info(
            f"[Worker rank={self.rank}] Fast-LLM weight update complete - {param_count} parameters processed"
        )


async def _pause_generation(engine: AsyncLLM) -> None:
    """Pause generation, keeping in-flight requests, for an in-place weight update."""
    await engine.pause_generation(mode="keep", clear_cache=False)


class EngineManager:
    def __init__(self, args, engine: AsyncLLM, engine_config: Any):
        self.args = args
        self.engine = engine
        self.engine_config = engine_config
        self.update_lock = asyncio.Lock()

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

    async def init_fast_llm_receiver(self):
        """Store Redis connection info for the main-process monitoring thread."""
        self._redis_host = self.args.redis_host
        self._redis_port = self.args.redis_port
        logger.info(
            f"Fast-LLM receiver initialized (Redis {self._redis_host}:{self._redis_port})"
        )

    async def receive_weight_update_fast_llm(self, version: int | None = None):
        """Run a fast-llm broadcast weight update paused-for-the-duration.

        Pause/resume wraps the collective RPC symmetrically with the HTTP path
        so that in-flight generation cannot interleave with a mid-broadcast
        parameter swap (the source of logprob drift PR #137 closed).

        `version` is recorded as the active model version once the new weights are
        loaded but before generation resumes, so tokens sampled after the swap are
        stamped with the new version and those before it keep the old one.

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
                # Weights are loaded; stamp subsequent tokens with the new version before resuming.
                _set_current_model_version(version)
                logger.info(
                    f"Fast-llm weight update processed version={version} "
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
        import threading

        self._fast_llm_stop_event = threading.Event()
        loop = asyncio.get_event_loop()

        def monitor_redis_stream():
            import redis

            redis_client = redis.Redis(host=self._redis_host, port=self._redis_port)
            # First weights_ready event since this vLLM process started is the
            # initial broadcast (step can be 0 on fresh start or k>0 on resume).
            # Actor is still blocked in wait_for_model_version at this point, so
            # vLLM has zero in-flight requests — pause_generation would deadlock.
            # Take the raw RPC path for the first event; wrap with pause/resume
            # thereafter, matching PR #137's guard against mid-rollout weight swaps.
            first_weights_ready_seen = False

            logger.info("[FastLLM] Main-process Redis monitoring started")

            try:
                for event_type, version, step, documents_seen in read_fast_llm_events(
                    redis_client, self._fast_llm_stop_event
                ):
                    if event_type == "weights_ready":
                        if not first_weights_ready_seen:
                            logger.info(
                                f"[FastLLM] weights_ready step={step} documents_seen={documents_seen} "
                                f"(initial broadcast — no pause wrap)"
                            )
                            coro = self.engine.engine_core.collective_rpc_async(
                                "receive_weight_update_fast_llm", args=()
                            )
                            first_weights_ready_seen = True
                            initial_broadcast = True
                        else:
                            logger.info(
                                f"[FastLLM] weights_ready step={step} documents_seen={documents_seen}, "
                                f"dispatching to workers"
                            )
                            coro = self.receive_weight_update_fast_llm(version)
                            initial_broadcast = False
                        try:
                            future = asyncio.run_coroutine_threadsafe(coro, loop)
                            future.result()
                            # The pause-wrapped path stamps the version internally (before
                            # resume); the initial raw path runs before the actor generates,
                            # so setting it here has no token to race with.
                            if initial_broadcast:
                                _set_current_model_version(version)
                            logger.info(f"[FastLLM] Weight update complete: step={step}")
                        except Exception as e:
                            logger.error(f"[FastLLM] Error receiving weight update: {e}")

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
                            logger.error(f"[FastLLM] Error destroying process group: {e}")
                        self._fast_llm_stop_event.set()
            finally:
                logger.info("[FastLLM] Main-process Redis monitoring stopped")
                redis_client.close()

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

    @staticmethod
    @asynccontextmanager
    async def create_engine(args: Any):
        """Create a vLLM AsyncLLM engine wrapped in an EngineManager.

        Async context manager yielding an ``EngineManager`` whose ``.engine`` and
        ``.engine_config`` hold the ``AsyncLLM`` and its ``VllmConfig``. The engine is
        left running on exit (server usage runs indefinitely).
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

        assert isinstance(engine.engine_core, AsyncMPClient)
        manager = EngineManager(args, engine, engine_config)
        weight_update_mode = getattr(args, "weight_update_mode", "http")
        try:
            if not args.disable_weight_updates:
                await manager.init_actor_update_group()

                if weight_update_mode == "fast-llm":
                    _install_model_version_patches()
                    await manager.init_fast_llm_receiver()
                    await manager.start_fast_llm_monitoring()
                    logger.info("Fast-LLM weight update mode enabled")

            yield manager
        finally:
            if not args.disable_weight_updates:
                if weight_update_mode == "fast-llm":
                    await manager.stop_fast_llm_monitoring()

                if not await manager.is_actor_update_group_destroyed():
                    logger.warning(
                        "training_finished was not called before shutdown; "
                        "NCCL process group was not destroyed — potential resource leak"
                    )


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

    async with EngineManager.create_engine(args) as manager:
        # Run HTTP server
        sock_addr = (args.host or "", args.port)
        sock = create_server_socket(sock_addr)
        # vLLM 0.18.1+ requires supported_tasks to build the app and app state;
        # older vllm (e.g. 0.14.x) has 1-arg build_app / 3-arg init_app_state.
        import inspect
        _build_app_params = inspect.signature(build_app).parameters
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

        if "supported_tasks" in inspect.signature(init_app_state).parameters:
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
