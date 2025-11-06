import importlib
import inspect
from functools import lru_cache
from typing import Awaitable, Callable, Iterable, Mapping

import aiohttp
from omegaconf import DictConfig
from tapeagents.llms.trainable import TrainableLLM

from pipelinerl.rollouts import RolloutResult
from pipelinerl.utils import resolve_environment_key

RolloutCallable = Callable[[DictConfig, TrainableLLM, dict, aiohttp.ClientSession], Awaitable[RolloutResult] | RolloutResult]

_RUNTIME_DOMAIN_ROLLOUTS: dict[str, str] = {}


def _iter_domain_overrides(raw_overrides: Mapping | Iterable | None):
    if raw_overrides is None:
        return

    if isinstance(raw_overrides, Mapping):
        for domain, path in raw_overrides.items():
            yield str(domain), str(path)
        return

    # Support Hydra list syntax: [{domain: math, rollout: path}, ...] or ["math:path"]
    for item in raw_overrides:
        if isinstance(item, Mapping):
            if "domain" in item and "rollout" in item:
                yield str(item["domain"]), str(item["rollout"])
            else:
                for domain, path in item.items():
                    yield str(domain), str(path)
        else:
            text = str(item)
            if ":" in text:
                domain, path = text.split(":", 1)
                yield domain.strip(), path.strip()


def _build_domain_mapping(cfg: DictConfig) -> dict[str, str]:
    overrides = getattr(cfg.actor, "domain_rollouts", None)
    mapping: dict[str, str] = {}
    if overrides:
        for domain, path in _iter_domain_overrides(overrides):
            mapping[domain] = path

    if _RUNTIME_DOMAIN_ROLLOUTS:
        mapping.update(_RUNTIME_DOMAIN_ROLLOUTS)

    if not mapping:
        raise ValueError("`actor.domain_rollouts` produced an empty mapping and no runtime registrations were found")
    return mapping


@lru_cache(maxsize=None)
def _import_callable(path: str) -> RolloutCallable:
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def _get_rollout_callable(cfg: DictConfig, domain: str) -> RolloutCallable:
    mapping = _build_domain_mapping(cfg)
    if domain not in mapping:
        raise ValueError(f"No rollout policy registered for domain '{domain}'")
    return _import_callable(mapping[domain])


async def generate_multidomain_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict,
    session: aiohttp.ClientSession,
) -> RolloutResult:
    domain = problem.get("domain")
    if not domain:
        domain = resolve_environment_key(cfg)
    if not domain:
        raise ValueError("Problem is missing 'domain' and no default could be resolved from config")

    rollout_fn = _get_rollout_callable(cfg, domain)
    result = rollout_fn(cfg, llm, problem, session)
    if inspect.isawaitable(result):
        result = await result  # type: ignore[assignment]
    return result  # type: ignore[return-value]


def register_domain_rollout(domain: str, target: str) -> None:
    domain_key = str(domain).strip()
    target_path = str(target).strip()
    if not domain_key:
        raise ValueError("Domain key for registration cannot be empty")
    if not target_path or "." not in target_path:
        raise ValueError(f"Target '{target}' must be a fully-qualified callable path")
    _RUNTIME_DOMAIN_ROLLOUTS[domain_key] = target_path
    _import_callable.cache_clear()
