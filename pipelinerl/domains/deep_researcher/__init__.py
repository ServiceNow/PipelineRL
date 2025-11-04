"""DeepResearcher domain for PipelineRL."""
from .rollouts import generate_deep_researcher_rollout, RewardTable, Metrics
from .load_datasets import load_datasets
from .verifier_api import DeepResearcherEnvironment, verify_answer, verify_answer_rpc
from .orchestration import BaseOrchestrator, OrchestrationRegistry, ReActOrchestrator

__all__ = [
    "generate_deep_researcher_rollout",
    "RewardTable",
    "Metrics",
    "load_datasets",
    "DeepResearcherEnvironment",
    "verify_answer",
    "verify_answer_rpc",
    "BaseOrchestrator",
    "OrchestrationRegistry",
    "ReActOrchestrator",
]
