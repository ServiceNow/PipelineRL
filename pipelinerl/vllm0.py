import asyncio
import json
import logging
import os
import signal
from pydantic import TypeAdapter
import torch
import uvloop
from vllm import AsyncLLMEngine
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
from vllm.logger import init_logger
from vllm._version import version
from vllm.worker.worker import Worker
from vllm.executor.multiproc_worker_utils import ProcessWorkerWrapper
from vllm.executor.mp_distributed_executor import MultiprocessingDistributedExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.usage.usage_lib import UsageContext
from vllm.worker.multi_step_worker import MultiStepWorker
from vllm.worker.multi_step_model_runner import MultiStepModelRunner


import torch.distributed as dist
from pipelinerl.finetune_loop import TrainerMessage, WeightUpdateRequest
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


def make_worker_class(multi_step: bool):
    base_class = MultiStepWorker if multi_step else Worker

    class NewWorkerClass(base_class):
        def init_actor_update_group(
            self,
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

        def receive_weight_update(self, request: WeightUpdateRequest):
            torch.cuda.synchronize(self.device)
            for info in request.parameters_info:
                model_dtype = self.model_config.dtype
                assert info.dtype == str(model_dtype), (
                    f"mismatch dtype: src {info.dtype},\ dst {self.model_config.dtype}"
                )
                buffer = torch.empty(tuple(info.shape), dtype=model_dtype, device=self.device)
                torch.distributed.broadcast(buffer, src=0, group=self.process_group)
                if isinstance(self.model_runner, MultiStepModelRunner):
                    loaded_params = self.model_runner._base_model_runner.model.load_weights(
                        weights=[(info.name, buffer)]
                    )
                else:
                    loaded_params = self.model_runner.model.load_weights(weights=[(info.name, buffer)])
                if len(loaded_params) != 1:
                    raise ValueError(f"model {info.name} not found in model state dict")
            logger.info("Weight update received")

    return NewWorkerClass


AsyncRLWorker = make_worker_class(multi_step=False)
AsyncRLMultiStepWorker = make_worker_class(multi_step=True)

executor_lock = asyncio.Lock()


class AsyncRLExecutor(MultiprocessingDistributedExecutor):
    async def execute_model_async(self, execute_model_req: ExecuteModelRequest) -> list[SamplerOutput]:
        async with executor_lock:
            return await super().execute_model_async(execute_model_req)

    async def stop_remote_worker_execution_loop_async(self) -> None:
        async with executor_lock:
            await super().stop_remote_worker_execution_loop_async()

    async def stop_remote_worker_execution_loop_no_lock(self) -> None:
        await super().stop_remote_worker_execution_loop_async()


class WeightUpdateManager:
    def __init__(self, args, executor: AsyncRLExecutor):
        self.executor = executor
        self.driver_worker = getattr(executor, "driver_worker")
        self.multi_step = args.num_scheduler_steps > 1
        assert isinstance(self.driver_worker.worker, AsyncRLMultiStepWorker if self.multi_step else AsyncRLWorker)
        self.other_workers = getattr(executor, "workers")
        self.args = args

    def input_process_groups(self):
        # Make a render-vous with the trainer
        futures = []
        for i, worker in enumerate(self.other_workers):
            assert isinstance(worker, ProcessWorkerWrapper)
            futures.append(
                worker.execute_method(
                    "init_actor_update_group",
                    self.args.actor_llm_idx,
                    torch.cuda.device_count(),
                    self.args.weight_update_group_init_method,
                    self.args.weight_update_group_world_size,
                )
            )
        self.driver_worker.init_actor_update_group(
            self.args.actor_llm_idx,
            torch.cuda.device_count(),
            self.args.weight_update_group_init_method,
            self.args.weight_update_group_world_size,
        )
        for future in futures:
            future.get()

    async def receive_weight_update(self, message: WeightUpdateRequest):
        logger.info(f"Received weight update request")
        async with executor_lock:
            if isinstance(self.executor, AsyncRLExecutor):
                await self.executor.stop_remote_worker_execution_loop_no_lock()
            logger.info(f"Stopped remote worker")
            futures = []
            for worker in self.other_workers:
                futures.append(worker.execute_method("receive_weight_update", message))
            self.driver_worker.receive_weight_update(message)
            for future in futures:
                future.get()
            logger.info(f"All workers received weight updates")


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

    # Build the engine with the bespoke Executor and orker clases
    multi_step = args.num_scheduler_steps > 1
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_config = engine_args.create_engine_config(UsageContext.OPENAI_API_SERVER)
    engine_config.parallel_config.distributed_executor_backend = AsyncRLExecutor
    engine_config.parallel_config.worker_cls = (
        "pipelinerl.vllm0.AsyncRLMultiStepWorker" if multi_step else "pipelinerl.vllm0.AsyncRLWorker"
    )
    engine = AsyncLLMEngine.from_vllm_config(
        vllm_config=engine_config,
        usage_context=UsageContext.OPENAI_API_SERVER,
        stat_loggers=None,
        start_engine_loop=True,
        disable_log_stats=engine_args.disable_log_stats,
        disable_log_requests=engine_args.disable_log_requests,
    )

    assert isinstance(engine.engine.model_executor, AsyncRLExecutor)
    weight_update_manager = WeightUpdateManager(args, engine.engine.model_executor)
    if not args.disable_weight_updates:
        weight_update_manager.input_process_groups()

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