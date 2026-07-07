import asyncio
import json
import time

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from pipelinerl.domains.terminal.environment import TerminalSession
from pipelinerl.domains.terminal.environment_server import TerminalEnvironmentServer


class ScriptedEnv:
    def __init__(self, steps):
        self.steps = list(steps)
        self.commands = []
        self.timeouts = []

    def exec(self, command, timeout):
        self.commands.append(command)
        self.timeouts.append(timeout)
        return self.steps.pop(0)


def _session(tmp_path, steps):
    session = TerminalSession(
        bases_dir=tmp_path,
        proot_bin="proot",
        nameserver="10.150.0.10",
        command_timeout=12.0,
        verifier_timeout=30.0,
        max_observation_chars=200,
        check_initial_state=False,
    )
    session._env = ScriptedEnv(steps)
    return session


def test_replay_executes_all_commands_and_records_successes(tmp_path):
    session = _session(
        tmp_path,
        [
            (True, "one", None),
            (False, "two", None),
            (True, "three", None),
        ],
    )

    result = session.replay(["cmd1", "cmd2", "cmd3"])

    assert result == {
        "n_executed": 3,
        "successes": [True, False, True],
        "abort_kind": None,
        "last_output": "three",
    }
    assert session._env.commands == ["cmd1", "cmd2", "cmd3"]
    assert session._env.timeouts == [12.0, 12.0, 12.0]


def test_replay_stops_on_abort_kind(tmp_path):
    session = _session(
        tmp_path,
        [
            (True, "one", None),
            (False, "timed out", "timeout"),
            (True, "unreached", None),
        ],
    )

    result = session.replay(["cmd1", "cmd2", "cmd3"])

    assert result == {
        "n_executed": 2,
        "successes": [True, False],
        "abort_kind": "timeout",
        "last_output": "timed out",
    }
    assert session._env.commands == ["cmd1", "cmd2"]


def test_replay_empty_command_list(tmp_path):
    session = _session(tmp_path, [])

    result = session.replay([])

    assert result == {"n_executed": 0, "successes": [], "abort_kind": None, "last_output": ""}
    assert session._env.commands == []


def test_replay_endpoint_contract_and_activity_refresh():
    class ReplaySession:
        def __init__(self, server):
            self.server = server
            self.commands = None
            self.activity_seen_during_replay = None

        def replay(self, commands):
            self.commands = list(commands)
            self.activity_seen_during_replay = self.server._session_last_activity["session-1"]
            return {
                "n_executed": len(commands),
                "successes": [True for _ in commands],
                "abort_kind": None,
                "last_output": "ok",
            }

    async def run_case():
        server = TerminalEnvironmentServer(
            bases_dir="/tmp",
            n_envs=1,
            session_ttl_seconds=60.0,
            session_reap_interval_seconds=60.0,
        )
        session = ReplaySession(server)
        old_activity = time.monotonic() - 10.0
        server._sessions["session-1"] = session
        server._session_last_activity["session-1"] = old_activity

        app = web.Application()
        app.add_routes([web.post("/replay", server.replay)])
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            missing = await client.post("/replay", json={"session_id": "missing", "commands": []})
            response = await client.post(
                "/replay",
                json={"session_id": "session-1", "commands": ["pwd", "echo ok"]},
            )
            body = await response.json()
        finally:
            await client.close()
            server._executor.shutdown(wait=True)

        assert missing.status == 404
        assert response.status == 200
        assert body == {
            "n_executed": 2,
            "successes": [True, True],
            "abort_kind": None,
            "last_output": "ok",
        }
        assert session.commands == ["pwd", "echo ok"]
        assert session.activity_seen_during_replay > old_activity
        assert server._session_last_activity["session-1"] >= session.activity_seen_during_replay

    asyncio.run(run_case())
