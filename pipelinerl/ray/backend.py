from __future__ import annotations

import logging
from typing import Any, Protocol

import ray

from pipelinerl.ray.utils import is_expected_ray_shutdown


class RolloutExecutionBackend(Protocol):
    def create_actor(self, actor_cls: type, args: tuple[Any, ...], ray_options: dict[str, Any]) -> Any:
        """Create a rollout actor handle."""

    def get(self, ref: Any, *, timeout: float | None = None) -> Any:
        """Resolve one object reference or a nested container of references."""

    def wait(
        self,
        refs: list[Any],
        *,
        num_returns: int = 1,
        timeout: float | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """Wait for rollout refs to complete."""

    def close_actor(self, actor: Any, logger: logging.Logger, actor_name: str, *, timeout: float = 2.0) -> None:
        """Best-effort actor cleanup."""

    def kill_actor(self, actor: Any, logger: logging.Logger, actor_name: str) -> None:
        """Best-effort actor termination."""


class RayExecutionBackend:
    def create_actor(self, actor_cls: type, args: tuple[Any, ...], ray_options: dict[str, Any]) -> Any:
        return ray.remote(actor_cls).options(**ray_options).remote(*args)

    def get(self, ref: Any, *, timeout: float | None = None) -> Any:
        return ray.get(ref, timeout=timeout) if timeout is not None else ray.get(ref)

    def wait(
        self,
        refs: list[Any],
        *,
        num_returns: int = 1,
        timeout: float | None = None,
    ) -> tuple[list[Any], list[Any]]:
        return ray.wait(refs, num_returns=num_returns, timeout=timeout)

    def close_actor(self, actor: Any, logger: logging.Logger, actor_name: str, *, timeout: float = 2.0) -> None:
        try:
            self.get(actor.close.remote(), timeout=timeout)
            logger.info("Closed %s: %s", actor_name, actor)
        except Exception as exc:
            if is_expected_ray_shutdown(exc):
                logger.info("%s already gone during shutdown: %s", actor_name, actor)
            else:
                logger.exception("Failed to close %s: %s", actor_name, actor)

    def kill_actor(self, actor: Any, logger: logging.Logger, actor_name: str) -> None:
        try:
            ray.kill(actor, no_restart=True)
            logger.info("Killed %s: %s", actor_name, actor)
        except Exception:
            logger.exception("Failed to kill %s: %s", actor_name, actor)


class _SyncObjectRef:
    def __init__(self, value: Any):
        self.value = value


class _SyncRemoteMethod:
    def __init__(self, fn: Any):
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> _SyncObjectRef:
        return _SyncObjectRef(self._fn(*args, **kwargs))


class _SyncActorHandle:
    def __init__(self, actor: Any):
        self._actor = actor

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._actor, name)
        if callable(attr):
            return _SyncRemoteMethod(attr)
        return attr


class SyncExecutionBackend:
    """In-process rollout backend for stepping through worker code with a debugger."""

    def create_actor(self, actor_cls: type, args: tuple[Any, ...], ray_options: dict[str, Any]) -> Any:
        return _SyncActorHandle(actor_cls(*args))

    def get(self, ref: Any, *, timeout: float | None = None) -> Any:
        if isinstance(ref, _SyncObjectRef):
            return ref.value
        if isinstance(ref, list):
            return [self.get(item) for item in ref]
        if isinstance(ref, tuple):
            return tuple(self.get(item) for item in ref)
        if isinstance(ref, dict):
            return type(ref)((key, self.get(value)) for key, value in ref.items())
        return ray.get(ref, timeout=timeout) if timeout is not None else ray.get(ref)

    def wait(
        self,
        refs: list[Any],
        *,
        num_returns: int = 1,
        timeout: float | None = None,
    ) -> tuple[list[Any], list[Any]]:
        local_ready = []
        ray_refs = []
        for ref in refs:
            if isinstance(ref, _SyncObjectRef):
                local_ready.append(ref)
            else:
                ray_refs.append(ref)

        if len(local_ready) >= num_returns:
            return local_ready[:num_returns], local_ready[num_returns:] + ray_refs

        remaining = num_returns - len(local_ready)
        if ray_refs:
            ready, not_ready = ray.wait(ray_refs, num_returns=remaining, timeout=timeout)
            return local_ready + ready, not_ready
        return local_ready, []

    def close_actor(self, actor: Any, logger: logging.Logger, actor_name: str, *, timeout: float = 2.0) -> None:
        try:
            self.get(actor.close.remote(), timeout=timeout)
            logger.info("Closed sync rollout actor %s", actor_name)
        except Exception:
            logger.exception("Failed to close sync rollout actor %s", actor_name)

    def kill_actor(self, actor: Any, logger: logging.Logger, actor_name: str) -> None:
        logger.debug("Sync rollout actor %s does not require ray.kill", actor_name)
