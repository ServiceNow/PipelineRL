"""Standalone launcher for a fleet of terminal env servers on a CPU-only job.

Runs ``count`` ``TerminalEnvironmentServer`` instances, one OS process per port,
so a single crashing server cannot take the others down (the supervisor restarts
a dead port while the rest keep serving). No ``WorldMap`` / model / finetune
dependency: this is meant to run on a ``--gpu 0`` eai job that only mounts the
repo + terminal bases/proot + a writable cache.

The GPU training job reaches these servers over account-scoped internal-dns
(``http://dns-<EAI_ACCOUNT_ID>-<name>:<port>``); see the external env placement in
``conf`` and ``WorldMap._place_environments`` (which skips external specs).

Fleet processes rename their Linux ``comm`` field so in-guest broad kills like
``killall python3`` do not match the host supervisor/server processes under
proot's shared pid view. Accepted residual channels: ``pkill -f python`` can
still match cmdline, ``killall bash`` can hit the job wrapper bash or tmp
script shell plus other sessions' in-guest runners, and ``kill -9 -1`` remains
process-wide; command filtering is intentionally out of scope because valid task
solutions include process kills.

Run with hydra overrides selecting the env config and the port range, e.g.::

    python -m pipelinerl.entrypoints.run_environment_fleet \
        --config-name terminal \
        +fleet.environment_key=terminal +fleet.start_port=7777 +fleet.count=12
"""
from __future__ import annotations

import ctypes
import logging
import multiprocessing as mp
import os
import signal
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from pipelinerl import fleet as fleet_manager
from pipelinerl.utils import better_crashing, select_environment_config

logger = logging.getLogger(__name__)

_PR_SET_NAME = 15
_FLEET_SUPERVISOR_COMM = "plenv-fleet"
_FLEET_SERVER_COMM = "plenv-serve"


def _set_process_comm(name: str) -> None:
    try:
        encoded = name.encode("utf-8")[:15]
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(_PR_SET_NAME, ctypes.c_char_p(encoded), 0, 0, 0)
        if result != 0:
            errno = ctypes.get_errno()
            message = os.strerror(errno) if errno else "prctl(PR_SET_NAME) returned non-zero"
            raise OSError(errno, message)
    except Exception as e:
        logger.warning("failed to set process comm to %s: %s", name, e)


def _serve(env_container: dict, port: int) -> None:
    _set_process_comm(_FLEET_SERVER_COMM)
    # Rebuild the server in the child process and block in web.run_app.
    server = hydra.utils.instantiate(OmegaConf.create(env_container))
    server.launch(port=port)


def _spawn(env_container: dict, port: int) -> mp.Process:
    proc = mp.Process(target=_serve, args=(env_container, port), name=f"env-{port}", daemon=False)
    proc.start()
    return proc


@hydra.main(config_path="../../conf", config_name="base", version_base="1.3.2")
def hydra_entrypoint(cfg: DictConfig):
    _set_process_comm(_FLEET_SUPERVISOR_COMM)
    with better_crashing("environment_fleet"):
        logger.info("env fleet git sha: %s", fleet_manager.git_sha())
        fleet = getattr(cfg, "fleet", None)
        if fleet is None:
            raise ValueError("run_environment_fleet requires +fleet.{environment_key,start_port,count}")
        environment_cfg = select_environment_config(
            cfg,
            key=fleet.get("environment_key"),
            index=fleet.get("environment_index", 0),
        )
        if environment_cfg is None:
            raise ValueError("No environment configuration found for fleet")

        start_port = int(fleet.start_port)
        count = int(fleet.count)
        env_container = OmegaConf.to_container(environment_cfg, resolve=True)

        procs: dict[int, mp.Process] = {}
        for i in range(count):
            port = start_port + i
            procs[port] = _spawn(env_container, port)
        start_time = time.monotonic()
        logger.info("env fleet launched: %d servers on ports %d-%d", count, start_port, start_port + count - 1)

        # Log platform shutdown signals so a fleet death is diagnosable: a graceful
        # preemption/drain sends SIGTERM (caught here), whereas a hard SIGKILL (e.g.
        # OOM-kill / node-level evict) cannot be caught and only the heartbeat below
        # pins the death time. Raise RuntimeError (an Exception, not BaseException) so
        # the enclosing better_crashing runs terminate_with_children to reap the
        # non-daemon env child processes instead of leaving them orphaned on exit.
        def _on_signal(signum, _frame):
            alive = sum(1 for p in procs.values() if p.is_alive())
            logger.warning(
                "env fleet parent received signal %d (%s) after %.0fs uptime; %d/%d servers alive; exiting",
                signum, signal.Signals(signum).name, time.monotonic() - start_time, alive, count,
            )
            raise RuntimeError(f"signal {signum} ({signal.Signals(signum).name})")

        for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(_sig, _on_signal)

        # Supervisor: restart any server that dies so one bad port does not shrink
        # the fleet. The actor side already health-checks and load-balances, so a
        # brief gap on one port is tolerated.
        last_heartbeat = time.monotonic()
        while True:
            for port, proc in list(procs.items()):
                if not proc.is_alive():
                    logger.warning("env server on port %d exited (code %s); restarting", port, proc.exitcode)
                    procs[port] = _spawn(env_container, port)
            now = time.monotonic()
            if now - last_heartbeat >= 30.0:
                alive = sum(1 for p in procs.values() if p.is_alive())
                logger.info("env fleet heartbeat: %d/%d servers alive, uptime %.0fs", alive, count, now - start_time)
                last_heartbeat = now
            time.sleep(5.0)


if __name__ == "__main__":
    hydra_entrypoint()
