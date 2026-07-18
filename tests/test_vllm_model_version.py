import asyncio
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from pipelinerl.finetune_loop import WeightUpdateRequest
from pipelinerl.vllm1 import (
    MODEL_VERSION_END_HEADER,
    MODEL_VERSION_START_HEADER,
    WeightUpdateManager,
    install_model_version_headers,
)


class _Engine:
    def __init__(self) -> None:
        self.calls = []

    async def pause_generation(self, *, mode: str, clear_cache: bool) -> None:
        self.calls.append(("pause", mode, clear_cache))

    async def resume_generation(self) -> None:
        self.calls.append(("resume",))


class _EngineClient:
    def __init__(self, *, fail_update: bool = False) -> None:
        self.fail_update = fail_update
        self.calls = []

    async def collective_rpc_async(self, method: str, args=()) -> None:
        self.calls.append((method, args))
        if self.fail_update and method == "receive_weight_update":
            raise RuntimeError("update failed")


def _manager(*, fail_update: bool = False) -> tuple[WeightUpdateManager, _Engine, _EngineClient]:
    engine = _Engine()
    engine_client = _EngineClient(fail_update=fail_update)
    manager = WeightUpdateManager(SimpleNamespace(), engine, engine_client)
    return manager, engine, engine_client


def _update(version: int) -> WeightUpdateRequest:
    return WeightUpdateRequest(version=version, parameters_info=[])


def test_active_chat_request_records_crossing_update_headers():
    async def run_test() -> None:
        manager, engine, _ = _manager()
        app = FastAPI()
        generation_started = asyncio.Event()
        finish_generation = asyncio.Event()

        @app.post("/v1/chat/completions")
        async def chat_completion(request: Request):
            assert (await request.json())["model"] == "test"
            generation_started.set()
            await finish_generation.wait()
            return {"choices": []}

        install_model_version_headers(app, manager)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response_task = asyncio.create_task(
                client.post("/v1/chat/completions", json={"model": "test", "stream": False})
            )
            await asyncio.wait_for(generation_started.wait(), timeout=1)
            await manager.receive_weight_update(_update(7))
            finish_generation.set()
            response = await response_task

        assert response.headers[MODEL_VERSION_START_HEADER] == "0"
        assert response.headers[MODEL_VERSION_END_HEADER] == "7"
        assert engine.calls == [
            ("pause", "keep", False),
            ("resume",),
        ]

    asyncio.run(run_test())


def test_failed_update_keeps_served_version_and_successful_update_advances_it():
    async def run_test() -> None:
        manager, engine, _ = _manager(fail_update=True)
        assert manager.served_version == 0

        with pytest.raises(RuntimeError, match="update failed"):
            await manager.receive_weight_update(_update(4))

        assert manager.served_version == 0
        assert engine.calls[-1] == ("resume",)

        manager.engine_client.fail_update = False
        await manager.receive_weight_update(_update(9))
        assert manager.served_version == 9

    asyncio.run(run_test())


def test_version_header_middleware_does_not_serialize_concurrent_generations():
    async def run_test() -> None:
        manager, _, _ = _manager()
        app = FastAPI()
        both_started = asyncio.Event()
        finish_generations = asyncio.Event()
        active = 0

        @app.post("/v1/chat/completions")
        async def chat_completion(request: Request):
            nonlocal active
            assert (await request.json())["model"] == "test"
            active += 1
            if active == 2:
                both_started.set()
            await finish_generations.wait()
            return {"choices": []}

        install_model_version_headers(app, manager)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            tasks = [
                asyncio.create_task(
                    client.post("/v1/chat/completions", json={"model": "test", "stream": False})
                )
                for _ in range(2)
            ]
            await asyncio.wait_for(both_started.wait(), timeout=1)
            finish_generations.set()
            responses = await asyncio.gather(*tasks)

        assert active == 2
        assert [
            (
                response.headers[MODEL_VERSION_START_HEADER],
                response.headers[MODEL_VERSION_END_HEADER],
            )
            for response in responses
        ] == [("0", "0"), ("0", "0")]

    asyncio.run(run_test())


def test_streaming_chat_response_is_not_stamped():
    async def run_test() -> None:
        manager, _, _ = _manager()
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completion():
            return StreamingResponse(
                iter(["data: {}\n\n"]),
                media_type="text/event-stream",
            )

        install_model_version_headers(app, manager)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "stream": True},
            )

        assert response.headers["content-type"].startswith("text/event-stream")
        assert MODEL_VERSION_START_HEADER not in response.headers
        assert MODEL_VERSION_END_HEADER not in response.headers

    asyncio.run(run_test())
