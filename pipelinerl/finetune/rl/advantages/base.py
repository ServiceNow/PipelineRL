from __future__ import annotations

from typing import ClassVar, Protocol, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig


class AdvantageEstimator(Protocol):
    """Compute per-row `advantages` from a populated rollout dataframe.

    Inputs always include: group_id, rollout_index, step_index, rewards, input_ids.
    Estimators declare any additional columns they require via `required_columns`;
    the registry checks them up front so a misconfigured rollout producer fails
    fast instead of NaN-ing deep inside a merge.
    """

    name: ClassVar[str]
    required_columns: ClassVar[tuple[str, ...]]

    def compute(self, df: pd.DataFrame, config: "RLConfig") -> pd.DataFrame:
        """Return a dataframe keyed by (group_id, rollout_index, step_index) with:
          - advantages       : list[float] of len(input_ids)
          - step_advantage   : float per row (0.0 if estimator has no notion)
        Estimators may attach extra debug columns; the dispatcher will merge
        them back onto the main df by the three-key prefix.
        """
        ...
