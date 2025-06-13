import logging
import signal
import torch
import uvloop
from vllm.utils import FlexibleArgumentParser, set_ulimit
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
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm._version import version
from vllm.usage.usage_lib import UsageContext
from vllm.config import ModelConfig
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import AsyncMPClient
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


from pipelinerl.run_finetune import WeightUpdateRequest
from typing import Any, Protocol, runtime_checkable
import pipelinerl.torch_utils

logger = logging.getLogger(__name__)
# configure this logger individually, in order to avoid messign
# with the default vllm logger configuration
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


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

    def receive_weight_update(self: LikeWorker, request: WeightUpdateRequest):
        torch.cuda.synchronize(self.device)
        logger.info("Start receiving weight update")
        for info in request.parameters_info:
            model_dtype = self.model_config.dtype
            assert info.dtype == str(model_dtype), (
                f"mismatch dtype: src {info.dtype}, dst {self.model_config.dtype}"
            )
            buffer = torch.empty(tuple(info.shape), dtype=model_dtype, device=self.device)
            torch.distributed.broadcast(buffer, src=0, group=self.process_group)
            loaded_params = self.model_runner.model.load_weights(weights=[(info.name, buffer)]) # type: ignore
            if len(loaded_params) != 1:
                raise ValueError(f"model {info.name} not found in model state dict")
        logger.info("Weight update received")


class WeightUpdateManager:
    def __init__(self, args, engine_client: AsyncMPClient):
        self.args = args
        self.engine_client = engine_client

    async def input_process_groups(self):
        await self.engine_client.collective_rpc_async(
            "init_actor_update_group",
            args=(
                self.args.actor_llm_idx,
                torch.cuda.device_count(),
                self.args.weight_update_group_init_method,
                self.args.weight_update_group_world_size,
            ),
        )

    async def receive_weight_update(self, request: WeightUpdateRequest):
        await self.engine_client.collective_rpc_async(
            "receive_weight_update", args=(request,)
        )
        logger.info("Weight update processed")


async def run_server(args, **uvicorn_kwargs) -> None:
    # COPIED FROM vllm/entrypoints/openai/api_server.py, vllm version 0.6.6.post1
    logger.info("vLLM API server version %s", version)
    logger.info("args: %s", args)

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

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_args.worker_extension_cls = "pipelinerl.run_llm.WorkerExtension"
    engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    engine = AsyncLLM.from_vllm_config(
        vllm_config=engine_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        disable_log_stats=engine_args.disable_log_stats,
        disable_log_requests=engine_args.disable_log_requests,
    )
    assert isinstance(engine.engine_core, AsyncMPClient)

    weight_update_manager = WeightUpdateManager(args, engine.engine_core)
    if not args.disable_weight_updates:
        await weight_update_manager.input_process_groups()

    # Run HTTP server
    sock_addr = (args.host or "", args.port)
    sock = create_server_socket(sock_addr)
    app = build_app(args)

    @app.post("/receive_weight_update")
    async def _receive_weight_update(request: WeightUpdateRequest):
        await weight_update_manager.receive_weight_update(request)
        return {"status": "ok"}

    model_config = await engine.get_model_config()
    await init_app_state(engine, model_config, app.state, args)
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
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    parser.add_argument(
        "--disable-weight-updates", action="store_true", help="Whether to receive weight updates from the trainer"
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
    args = parser.parse_args()
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))
