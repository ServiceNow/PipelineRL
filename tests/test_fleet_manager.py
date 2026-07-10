import pytest
from omegaconf import OmegaConf

from pipelinerl import fleet


A_ID = "11111111-1111-1111-1111-111111111111"
B_ID = "22222222-2222-2222-2222-222222222222"
NEW_ID = "33333333-3333-3333-3333-333333333333"


def _cfg():
    return OmegaConf.create({
        "output_dir": "/tmp/out",
        "environment_key": "terminal",
        "orchestrator": {
            "manage_fleets": True,
            "account": "snow.research.adea",
            "image": "registry/image:tag",
            "mounts": [
                "home:/home/toolkit:rw",
                "data:/mnt/llmd/data:rw",
                "results:/mnt/llmd/results:rw",
            ],
            "fleet_resources": {"cpu": 64, "mem": 512},
            "bid": 9999,
            "conda_env": "pipeline-rl",
            "fleet_name_prefix": "terminal_envs",
            "fleet_restores_per_hour": 3,
        },
        "environments": [{
            "key": "terminal",
            "placement": "external",
            "external_hosts": [
                "dns-acct-terminal-envs-a",
                "dns-acct-terminal-envs-b",
            ],
            "external_start_port": 7777,
            "external_count": 2,
            "n_envs": 1,
        }],
    })


class FakeRunner:
    def __init__(self, alive=None):
        self.alive = list(alive or [])
        self.calls = []
        self.submits = 0

    def __call__(self, cmd):
        cmd = list(cmd)
        self.calls.append(cmd)
        if cmd[:4] == ["eai", "job", "ls", "--me"]:
            return fleet.CommandResult(stdout="\n".join(self.alive) + ("\n" if self.alive else ""))
        if cmd[:3] == ["eai", "job", "kill"]:
            killed = cmd[3]
            self.alive = [job_id for job_id in self.alive if job_id != killed]
            return fleet.CommandResult()
        if cmd[:3] == ["eai", "job", "new"]:
            self.submits += 1
            job_id = NEW_ID if self.submits == 1 else B_ID
            self.alive.append(job_id)
            return fleet.CommandResult(stdout=f"created job {job_id}\n")
        raise AssertionError(cmd)


def _handle(tmp_path, restore_history=None):
    cfg = _cfg()
    endpoints = fleet.derive_topology(cfg, "acct")
    jobs = {
        "a": fleet.FleetJob("a", A_ID, "terminal_envs_a_old", "terminal-envs-a", [7777, 7778], "a.yaml"),
        "b": fleet.FleetJob("b", B_ID, "terminal_envs_b_old", "terminal-envs-b", [7777, 7778], "b.yaml"),
    }
    return fleet.FleetHandle(
        exp_dir=tmp_path,
        manifest_path=tmp_path / "fleet" / "manifest.json",
        orchestrator=cfg.orchestrator,
        config_name="terminal_full_infra",
        endpoints={endpoint.suffix: endpoint for endpoint in endpoints},
        jobs=jobs,
        restore_history=restore_history or {},
        restores_per_hour=3,
    )


def test_fleet_spec_and_submit_command_shape(tmp_path):
    cfg = _cfg()
    endpoint = fleet.derive_topology(cfg, "acct")[0]

    text = fleet.render_fleet_spec(cfg.orchestrator, endpoint, "123")
    assert "name: terminal_envs_a_123" in text
    assert "name: terminal-envs-a" in text
    assert "        - {port: 7777, target-port: 7777, protocol: TCP}" in text
    assert "        - {port: 7778, target-port: 7778, protocol: TCP}" in text
    assert "resources: {cpu: 64, gpu: 0, mem: 512}" in text

    yaml_path = tmp_path / "fleet.yaml"
    cmd = fleet.submit_command(
        cfg.orchestrator,
        yaml_path,
        endpoint,
        tmp_path,
        "terminal_full_infra",
    )
    assert cmd[:4] == ["eai", "job", "new", "-f"]
    assert "--non-preemptable" in cmd
    assert cmd.index("--non-preemptable") < cmd.index("--")
    assert "output_dir=" + str(tmp_path / "fleet" / "env_fleet_a") in cmd[-1]
    assert "--config-name terminal_full_infra" in cmd[-1]


def test_stale_kill_uses_me_and_wait_dead_fail_closed():
    runner = FakeRunner(alive=[A_ID])

    fleet.kill_stale_fleets(_cfg().orchestrator, runner)

    assert ["eai", "job", "kill", A_ID] in runner.calls
    assert any("--me" in call and "-N" in call and "terminal_envs" in call for call in runner.calls)

    stuck = FakeRunner(alive=[A_ID])
    with pytest.raises(RuntimeError):
        fleet.wait_dead_by_filter(stuck, "terminal_envs", "fleet", attempts=2, sleep_s=0)


def test_restore_bounding_appends_event_without_submit(monkeypatch, tmp_path):
    monkeypatch.setattr(fleet, "git_sha", lambda: "sha")
    handle = _handle(tmp_path, restore_history={"a": [100.0, 200.0, 300.0]})
    runner = FakeRunner(alive=[])

    fleet.restore_fleet(handle, "a", runner, now=400.0)
    fleet.restore_fleet(handle, "a", runner, now=500.0)

    assert not any(call[:3] == ["eai", "job", "new"] for call in runner.calls)
    manifest = fleet.read_manifest(handle.manifest_path)
    assert [event["kind"] for event in manifest["events"]].count("restore_limit_exceeded") == 1
    assert "a" in handle.restore_limit_reported
    assert handle.jobs["a"].id == A_ID

    handle.restore_history["a"] = []
    fleet.restore_fleet(handle, "a", runner, now=600.0)

    manifest = fleet.read_manifest(handle.manifest_path)
    assert [event["kind"] for event in manifest["events"]].count("restore_limit_exceeded") == 1
    assert manifest["events"][-1]["kind"] == "restore"
    assert handle.jobs["a"].id == NEW_ID
    assert "a" not in handle.restore_limit_reported
    submit = next(call for call in runner.calls if call[:3] == ["eai", "job", "new"])
    assert "--config-name terminal_full_infra" in submit[-1]


def test_teardown_kills_jobs_and_appends_manifest_event(monkeypatch, tmp_path):
    monkeypatch.setattr(fleet, "git_sha", lambda: "sha")
    handle = _handle(tmp_path)
    runner = FakeRunner(alive=[A_ID, B_ID])

    fleet.teardown_fleets(handle, runner)

    assert ["eai", "job", "kill", A_ID] in runner.calls
    assert ["eai", "job", "kill", B_ID] in runner.calls
    manifest = fleet.read_manifest(handle.manifest_path)
    assert manifest["events"][-1]["kind"] == "teardown"
    assert handle.torn_down


def test_manifest_round_trip_and_event_append(monkeypatch, tmp_path):
    monkeypatch.setattr(fleet, "git_sha", lambda: "sha")
    handle = _handle(tmp_path)

    fleet.write_manifest(handle, {"kind": "startup"})
    fleet.append_event(handle, "restore", "a", job_id=NEW_ID)

    manifest = fleet.read_manifest(handle.manifest_path)
    assert manifest["git_sha"] == "sha"
    assert manifest["config_name"] == "terminal_full_infra"
    assert [event["kind"] for event in manifest["events"]] == ["startup", "restore"]
    assert manifest["events"][1]["job_id"] == NEW_ID


def test_dry_run_writes_specs_without_eai_or_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("EAI_ACCOUNT_ID", "acct")

    def forbidden_runner(cmd):
        raise AssertionError(cmd)

    config_names = []
    real_submit_command = fleet.submit_command

    def record_submit_command(orchestrator, yaml_path, endpoint, exp_dir, config_name):
        config_names.append(config_name)
        return real_submit_command(orchestrator, yaml_path, endpoint, exp_dir, config_name)

    monkeypatch.setattr(fleet, "submit_command", record_submit_command)
    handle = fleet.start_fleets(
        _cfg(),
        tmp_path,
        "terminal_full_infra",
        runner=forbidden_runner,
        dry_run=True,
    )

    assert handle is None
    assert config_names == ["terminal_full_infra", "terminal_full_infra"]
    assert (tmp_path / "fleet" / "env_fleet_a.yaml").exists()
    assert (tmp_path / "fleet" / "env_fleet_b.yaml").exists()
    assert not (tmp_path / "fleet" / "manifest.json").exists()
