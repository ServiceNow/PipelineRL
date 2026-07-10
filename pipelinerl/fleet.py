"""Terminal env-fleet lifecycle helpers for the launcher."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

JOB_ID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0


@dataclass
class FleetEndpoint:
    suffix: str
    dns_name: str
    external_host: str
    start_port: int
    count: int
    n_envs: int

    @property
    def ports(self) -> list[int]:
        return list(range(self.start_port, self.start_port + self.count))

    @property
    def job_base(self) -> str:
        return self.dns_name.replace("-", "_")


@dataclass
class FleetJob:
    suffix: str
    id: str
    name: str
    dns_name: str
    ports: list[int]
    yaml_path: str


@dataclass
class FleetHandle:
    exp_dir: Path
    manifest_path: Path
    orchestrator: DictConfig
    config_name: str
    endpoints: dict[str, FleetEndpoint]
    jobs: dict[str, FleetJob]
    restore_history: dict[str, list[float]] = field(default_factory=dict)
    restore_limit_reported: set[str] = field(default_factory=set)
    restores_per_hour: int = 3
    torn_down: bool = False


Runner = Callable[[Sequence[str]], CommandResult]


def run_eai(cmd: Sequence[str]) -> CommandResult:
    proc = subprocess.run(list(cmd), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr or proc.stdout}")
    return CommandResult(stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)


def _plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def should_manage_fleets(cfg: DictConfig) -> bool:
    orchestrator = getattr(cfg, "orchestrator", None)
    if orchestrator is None or not bool(orchestrator.get("manage_fleets", False)):
        return False
    return bool(external_environment_specs(cfg))


def external_environment_specs(cfg: DictConfig) -> list[DictConfig]:
    specs = []
    for env_cfg in getattr(cfg, "environments", []) or []:
        if env_cfg is not None and str(env_cfg.get("placement", "")) == "external":
            specs.append(env_cfg)
    return specs


def derive_topology(cfg: DictConfig, account_id: str) -> list[FleetEndpoint]:
    endpoints: list[FleetEndpoint] = []
    prefix = f"dns-{account_id}-" if account_id else ""
    for env_cfg in external_environment_specs(cfg):
        hosts = _plain(env_cfg.external_hosts)
        for host in hosts:
            dns_name = str(host)
            if prefix and dns_name.startswith(prefix):
                dns_name = dns_name[len(prefix):]
            suffix = dns_name.rsplit("-", 1)[-1]
            endpoints.append(FleetEndpoint(
                suffix=suffix,
                dns_name=dns_name,
                external_host=str(host),
                start_port=int(env_cfg.external_start_port),
                count=int(env_cfg.external_count),
                n_envs=int(env_cfg.n_envs),
            ))
    return endpoints


def ports_block(endpoint: FleetEndpoint) -> str:
    return "\n".join(
        f"        - {{port: {port}, target-port: {port}, protocol: TCP}}"
        for port in endpoint.ports
    )


def render_fleet_spec(orchestrator: DictConfig, endpoint: FleetEndpoint, run_id: str) -> str:
    mounts = "\n".join(f"  - '{mount}'" for mount in _plain(orchestrator.mounts))
    resources = orchestrator.fleet_resources
    return f"""image: {orchestrator.image}
data:
{mounts}
resources: {{cpu: {resources.cpu}, gpu: 0, mem: {resources.mem}}}
options:
  internal-dns:
      name: {endpoint.dns_name}
      ports:
{ports_block(endpoint)}
name: {endpoint.job_base}_{run_id}
bid: {orchestrator.bid}
workdir: /home/toolkit/PipelineRL
environmentVars:
  - HOME=/home/toolkit
  - PYTHONPATH=/home/toolkit/PipelineRL
"""


def spec_path(exp_dir: Path, endpoint: FleetEndpoint) -> Path:
    return exp_dir / "fleet" / f"env_fleet_{endpoint.suffix}.yaml"


def write_fleet_spec(exp_dir: Path, orchestrator: DictConfig, endpoint: FleetEndpoint, run_id: str) -> Path:
    path = spec_path(exp_dir, endpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_fleet_spec(orchestrator, endpoint, run_id))
    return path


def fleet_command(exp_dir: Path, endpoint: FleetEndpoint, config_name: str) -> str:
    return (
        "python -m pipelinerl.entrypoints.run_environment_fleet "
        f"--config-name {config_name} --config-dir /home/toolkit/PipelineRL/conf "
        f"output_dir={exp_dir / 'fleet' / f'env_fleet_{endpoint.suffix}'} "
        "+fleet.environment_key=terminal "
        f"+fleet.start_port={endpoint.start_port} +fleet.count={endpoint.count}"
    )


def submit_command(
    orchestrator: DictConfig,
    yaml_path: Path,
    endpoint: FleetEndpoint,
    exp_dir: Path,
    config_name: str,
) -> list[str]:
    return [
        "eai", "job", "new", "-f", str(yaml_path), "--account", str(orchestrator.account), "--non-preemptable", "--",
        "/opt/conda/bin/conda", "run", "-n", str(orchestrator.conda_env), "--no-capture-output", "bash", "-c",
        fleet_command(exp_dir, endpoint, config_name),
    ]


def parse_job_id(text: str) -> str | None:
    match = JOB_ID_RE.search(text)
    return match.group(0) if match else None


def preflight(runner: Runner = run_eai) -> str:
    if shutil.which("eai") is None:
        raise RuntimeError("eai CLI not found")
    account_id = os.environ.get("EAI_ACCOUNT_ID", "")
    if not account_id:
        raise RuntimeError("EAI_ACCOUNT_ID unset")
    runner(["eai", "job", "ls", "--me", "-n", "1"])
    return account_id


def list_alive_ids(runner: Runner, name_filter: str | None = None) -> list[str]:
    cmd = ["eai", "job", "ls", "--me", "--state", "alive"]
    if name_filter:
        cmd.extend(["-N", name_filter])
    cmd.extend(["--field", "id"])
    out = runner(cmd).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def kill_ids(runner: Runner, ids: Sequence[str], label: str) -> None:
    if not ids:
        logger.info("no live %s to kill", label)
        return
    for job_id in ids:
        logger.info("kill %s %s", label, job_id)
        try:
            runner(["eai", "job", "kill", job_id])
        except RuntimeError:
            logger.warning("failed to kill %s %s", label, job_id, exc_info=True)


def wait_dead_by_filter(runner: Runner, name_filter: str, label: str, attempts: int = 30, sleep_s: float = 5.0) -> None:
    for _ in range(attempts):
        if not list_alive_ids(runner, name_filter):
            logger.info("all %s dead", label)
            return
        time.sleep(sleep_s)
    raise RuntimeError(f"{label} still alive after wait; refusing to relaunch")


def wait_dead_ids(runner: Runner, ids: Sequence[str], label: str, attempts: int = 12, sleep_s: float = 5.0) -> bool:
    ids = [job_id for job_id in ids if job_id]
    if not ids:
        return True
    wanted = set(ids)
    for _ in range(attempts):
        if wanted.isdisjoint(set(list_alive_ids(runner))):
            logger.info("all %s dead", label)
            return True
        time.sleep(sleep_s)
    logger.error("%s still alive after bounded wait: %s", label, sorted(wanted))
    return False


def git_sha() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def manifest_path(exp_dir: Path) -> Path:
    return exp_dir / "fleet" / "manifest.json"


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_manifest(handle: FleetHandle, event: dict[str, Any] | None = None) -> None:
    path = handle.manifest_path
    events: list[dict[str, Any]] = []
    if path.exists():
        events = read_manifest(path).get("events", [])
    if event is not None:
        events.append(event)
    payload = {
        "git_sha": git_sha(),
        "config_name": handle.config_name,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fleets": [job.__dict__ for job in handle.jobs.values()],
        "events": events,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def append_event(handle: FleetHandle, kind: str, suffix: str | None = None, **extra: Any) -> None:
    event = {"kind": kind, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    if suffix is not None:
        event["suffix"] = suffix
    write_manifest(handle, event)


def stale_filter(orchestrator: DictConfig) -> str:
    return str(orchestrator.fleet_name_prefix)


def kill_stale_fleets(orchestrator: DictConfig, runner: Runner = run_eai) -> None:
    name_filter = stale_filter(orchestrator)
    ids = list_alive_ids(runner, name_filter)
    kill_ids(runner, ids, "fleet")
    wait_dead_by_filter(runner, name_filter, "fleet")


def submit_one(
    exp_dir: Path,
    orchestrator: DictConfig,
    endpoint: FleetEndpoint,
    run_id: str,
    config_name: str,
    runner: Runner = run_eai,
) -> FleetJob:
    yaml_path = write_fleet_spec(exp_dir, orchestrator, endpoint, run_id)
    cmd = submit_command(orchestrator, yaml_path, endpoint, exp_dir, config_name)
    result = runner(cmd)
    job_id = parse_job_id(result.stdout + "\n" + result.stderr)
    if job_id is None:
        ids = list_alive_ids(runner, f"{endpoint.job_base}_{run_id}")
        job_id = ids[0] if ids else None
    if job_id is None:
        raise RuntimeError(f"could not determine fleet job id for {endpoint.job_base}_{run_id}")
    return FleetJob(
        suffix=endpoint.suffix,
        id=job_id,
        name=f"{endpoint.job_base}_{run_id}",
        dns_name=endpoint.dns_name,
        ports=endpoint.ports,
        yaml_path=str(yaml_path),
    )


def start_fleets(
    cfg: DictConfig,
    exp_dir: Path,
    config_name: str,
    runner: Runner = run_eai,
    dry_run: bool = False,
) -> FleetHandle | None:
    if not should_manage_fleets(cfg):
        return None
    orchestrator = cfg.orchestrator
    account_id = os.environ.get("EAI_ACCOUNT_ID", "") if dry_run else preflight(runner)
    endpoints = derive_topology(cfg, account_id)
    run_id = str(int(time.time()))
    if dry_run:
        for endpoint in endpoints:
            yaml_path = write_fleet_spec(exp_dir, orchestrator, endpoint, run_id)
            cmd = submit_command(orchestrator, yaml_path, endpoint, exp_dir, config_name)
            logger.info("DRY_RUN fleet submit: %s", " ".join(cmd))
        return None

    kill_stale_fleets(orchestrator, runner)
    jobs = {
        endpoint.suffix: submit_one(exp_dir, orchestrator, endpoint, run_id, config_name, runner)
        for endpoint in endpoints
    }
    handle = FleetHandle(
        exp_dir=exp_dir,
        manifest_path=manifest_path(exp_dir),
        orchestrator=orchestrator,
        config_name=config_name,
        endpoints={endpoint.suffix: endpoint for endpoint in endpoints},
        jobs=jobs,
        restores_per_hour=int(orchestrator.fleet_restores_per_hour),
    )
    append_event(handle, "startup")
    logger.info("submitted %d terminal fleet jobs; actors will wait for healthy envs", len(jobs))
    return handle


def restore_allowed(handle: FleetHandle, suffix: str, now: float) -> bool:
    history = [t for t in handle.restore_history.get(suffix, []) if now - t < 3600.0]
    handle.restore_history[suffix] = history
    return len(history) < handle.restores_per_hour


def restore_fleet(handle: FleetHandle, suffix: str, runner: Runner = run_eai, now: float | None = None) -> None:
    now = time.monotonic() if now is None else now
    if not restore_allowed(handle, suffix, now):
        if suffix not in handle.restore_limit_reported:
            logger.error("restore limit exceeded for fleet %s", suffix)
            append_event(handle, "restore_limit_exceeded", suffix)
            handle.restore_limit_reported.add(suffix)
        return
    endpoint = handle.endpoints[suffix]
    name_filter = endpoint.job_base
    ids = list_alive_ids(runner, name_filter)
    kill_ids(runner, ids, "fleet")
    wait_dead_by_filter(runner, name_filter, "fleet")
    run_id = str(int(time.time()))
    job = submit_one(
        handle.exp_dir,
        handle.orchestrator,
        endpoint,
        run_id,
        handle.config_name,
        runner,
    )
    handle.jobs[suffix] = job
    handle.restore_history.setdefault(suffix, []).append(now)
    handle.restore_limit_reported.discard(suffix)
    append_event(handle, "restore", suffix, job_id=job.id, job_name=job.name)



def poll_and_restore(handle: FleetHandle, runner: Runner = run_eai, now: float | None = None) -> None:
    alive = set(list_alive_ids(runner))
    for suffix, job in list(handle.jobs.items()):
        if job.id not in alive:
            logger.error("fleet %s job %s is not alive; restoring", suffix, job.id)
            restore_fleet(handle, suffix, runner, now)


def teardown_fleets(handle: FleetHandle | None, runner: Runner = run_eai) -> None:
    if handle is None or handle.torn_down:
        return
    handle.torn_down = True
    ids = [job.id for job in handle.jobs.values()]
    kill_ids(runner, ids, "fleet")
    try:
        wait_dead_ids(runner, ids, "fleet")
    except RuntimeError:
        logger.warning("fleet teardown wait failed", exc_info=True)
    append_event(handle, "teardown")
