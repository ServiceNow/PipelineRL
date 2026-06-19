"""Registry of advantage estimators for the RL preprocessor.

To add a new estimator:
  1. Create `my_estimator.py` exporting:
       - MyEstimatorConfig(BaseModel) with `type: Literal["my_estimator"]`
       - MyEstimator implementing the `AdvantageEstimator` protocol
  2. Add the config to `AdvantageConfig` (Pydantic discriminated union below).
  3. Add the class to `ESTIMATORS`.
The discriminator field is `type`, so YAML/CLI overrides look like
`finetune.rl.advantage.type=my_estimator finetune.rl.advantage.foo=bar`.
"""
from __future__ import annotations

from typing import Annotated, TYPE_CHECKING, Union

import pandas as pd
from pydantic import Field

from .base import AdvantageEstimator
from .gigpo import Gigpo, GigpoConfig
from .grpo_loo import GrpoLoo, GrpoLooConfig

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig

AdvantageConfig = Annotated[
    Union[GrpoLooConfig, GigpoConfig],
    Field(discriminator="type"),
]

ESTIMATORS: dict[str, type[AdvantageEstimator]] = {
    GrpoLoo.name: GrpoLoo,
    Gigpo.name: Gigpo,
}


def compute_advantages(df: pd.DataFrame, rl_config: "RLConfig") -> pd.DataFrame:
    """Dispatch to the configured estimator and return its advantage dataframe."""
    adv_cfg = rl_config.advantage
    try:
        estimator_cls = ESTIMATORS[adv_cfg.type]
    except KeyError as e:
        known = sorted(ESTIMATORS)
        raise ValueError(f"Unknown advantage estimator '{adv_cfg.type}'. Registered: {known}") from e
    estimator = estimator_cls()
    missing = set(estimator.required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Advantage estimator '{adv_cfg.type}' requires columns {sorted(missing)} "
            f"not present in the rollout dataframe; check the active domain's rollout producer."
        )
    return estimator.compute(df, rl_config)


__all__ = [
    "AdvantageConfig",
    "AdvantageEstimator",
    "ESTIMATORS",
    "GigpoConfig",
    "GrpoLooConfig",
    "compute_advantages",
]
