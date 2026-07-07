import asyncio
import json
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from pipelinerl.domains.terminal import environment_server
from pipelinerl.domains.terminal.environment_server import (
    TerminalEnvironmentServer,
    _memory_eviction_decision,
)
from pipelinerl.domains.terminal.rollouts import EnvironmentCapacityError, _post


class DummySession:
    def __init__(self):
        self.closed = False

    def close(self, contamination_sample=True):
        self.closed = True
        return 0, 0


def test_memory_eviction_decision_matrix():
    assert _memory_eviction_decision(0.49, ["a", "b"], False, 0.5, 0.4) == ([], False)
    assert _memory_eviction_decision(0.5, ["a", "b", "c", "d"], False, 0.5, 0.4) == (["a"], True)
    assert _memory_eviction_decision(0.9, list("abcdefghi"), False, 0.5, 0.4) == (["a", "b", "c"], True)
    assert _memory_eviction_decision(0.45, ["a", "b"], True, 0.5, 0.4) == ([], True)
    assert _memory_eviction_decision(0.4, ["a", "b"], True, 0.5, 0.4) == ([], False)
    assert _memory_eviction_decision(0.9, [], False, 0.5, 0.4) == ([], True)
    assert _memory_eviction_decision(None, ["a"], True, 0.5, 0.4) == ([], False)
    assert _memory_eviction_decision(0.9, ["a"], False, 0.0, 0.4) == ([], False)


def test_cgroup_memory_reads_v2_and_handles_unlimited(monkeypatch, tmp_path):
    cgroup_v2 = tmp_path / "v2"
    cgroup_v1 = tmp_path / "v1"
    proc_meminfo = tmp_path / "meminfo"
    cgroup_v2.mkdir()
    cgroup_v1.mkdir()
    proc_meminfo.write_text("MemTotal:       1024 kB\n")
    (cgroup_v2 / "memory.current").write_text("25")
    (cgroup_v2 / "memory.max").write_text("100")

    monkeypatch.setattr(environment_server, "_CGROUP_V2_ROOT", cgroup_v2)
    monkeypatch.setattr(environment_server, "_CGROUP_V1_MEMORY_ROOT", cgroup_v1)
    monkeypatch.setattr(environment_server, "_PROC_MEMINFO", proc_meminfo)

    assert environment_server._cgroup_memory() == (25, 100)

    (cgroup_v2 / "memory.max").write_text("max")
    assert environment_server._cgroup_memory() is None


def test_cgroup_memory_reads_v1_and_ignores_absurd_limit(monkeypatch, tmp_path):
    cgroup_v2 = tmp_path / "v2"
    cgroup_v1 = tmp_path / "v1"
    proc_meminfo = tmp_path / "meminfo"
    cgroup_v2.mkdir()
    cgroup_v1.mkdir()
    proc_meminfo.write_text("MemTotal:       1024 kB\n")
    (cgroup_v1 / "memory.usage_in_bytes").write_text("40")
    (cgroup_v1 / "memory.limit_in_bytes").write_text("80")

    monkeypatch.setattr(environment_server, "_CGROUP_V2_ROOT", cgroup_v2)
    monkeypatch.setattr(environment_server, "_CGROUP_V1_MEMORY_ROOT", cgroup_v1)
    monkeypatch.setattr(environment_server, "_PROC_MEMINFO", proc_meminfo)

    assert environment_server._cgroup_memory() == (40, 80)

    (cgroup_v1 / "memory.limit_in_bytes").write_text(str(2 * 1024 * 1024))
    assert environment_server._cgroup_memory() is None


def test_memory_pressure_eviction_pauses_admission_and_health_reports(monkeypatch):
    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=2,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
            memory_evict_fraction=0.8,
            memory_resume_fraction=0.5,
        )
        old = DummySession()
        fresh = DummySession()
        server._sessions["old"] = old
        server._sessions["fresh"] = fresh
        server._session_last_activity["old"] = time.monotonic() - 10.0
        server._session_last_activity["fresh"] = time.monotonic()
        monkeypatch.setattr(environment_server, "_cgroup_memory", lambda: (90, 100))
        monkeypatch.setattr(environment_server, "_memory_stat_snapshot", lambda: "anon=9 file=1")

        await server._evict_memory_pressure_sessions()
        if server._bg_tasks:
            await asyncio.gather(*list(server._bg_tasks))

        app = web.Application()
        app.add_routes([web.get("/health", server.health), web.post("/start_task", server.start_task)])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            response = await client.post("/start_task", json={"task_data": {"task": "fix it"}})
            body = await response.json()
            health_response = await client.get("/health")
            health = await health_response.json()
        finally:
            await client.close()
            server._executor.shutdown(wait=True)

        assert old.closed
        assert not fresh.closed
        assert "old" not in server._sessions
        assert "fresh" in server._sessions
        assert server._admission_paused
        assert response.status == 503
        assert body == {"error": "memory pressure"}
        assert health["memory_fraction"] == 0.9
        assert health["admission_paused"]

    asyncio.run(run_case())


def test_memory_pressure_resume_clears_admission_pause(monkeypatch):
    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=1,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
            memory_evict_fraction=0.8,
            memory_resume_fraction=0.5,
        )
        server._admission_paused = True
        monkeypatch.setattr(environment_server, "_cgroup_memory", lambda: (50, 100))

        await server._evict_memory_pressure_sessions()
        server._executor.shutdown(wait=True)

        assert not server._admission_paused

    asyncio.run(run_case())


def test_rollout_post_treats_memory_pressure_503_as_capacity():
    class FakeResponse:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def text(self):
            return json.dumps({"error": "memory pressure"})

    class FakeSession:
        def post(self, url, json=None, timeout=None):
            return FakeResponse()

    async def run_case():
        with pytest.raises(EnvironmentCapacityError):
            await _post(FakeSession(), "http://env/start_task", {"task_data": {}}, 1.0)

    asyncio.run(run_case())
