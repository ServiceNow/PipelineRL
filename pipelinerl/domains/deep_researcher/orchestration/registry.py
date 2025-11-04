"""Registry for orchestration strategies."""
from typing import Dict, Type
from .base import BaseOrchestrator


class OrchestrationRegistry:
    
    _strategies: Dict[str, Type[BaseOrchestrator]] = {}
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[BaseOrchestrator]):
        cls._strategies[name] = strategy_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseOrchestrator]:
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise ValueError(
                f"Unknown orchestration strategy: '{name}'. "
                f"Available strategies: {available}"
            )
        return cls._strategies[name]
    
    @classmethod
    def list_strategies(cls) -> list:
        return list(cls._strategies.keys())


def register_strategy(name: str):
    def decorator(cls):
        OrchestrationRegistry.register(name, cls)
        return cls
    return decorator
