from __future__ import annotations

from typing import ClassVar, Protocol, TYPE_CHECKING
import numpy as np

import pandas as pd

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig

def _attach_local_return_to_go(df: pd.DataFrame, gamma_r: float) -> pd.DataFrame:
    """Compute G_local[t] = sum_{u>=t} gamma_r^{u-t} * step_reward[u] per rollout.

    Implementation: within each (group_id, rollout_index), sort by step_index
    ascending and compute the discounted reverse-cumulative sum:
        G[T-1] = r[T-1]
        G[t]   = r[t] + gamma_r * G[t+1]
    Writes result into a new `g_local` column aligned to the original index.
    """
    out = df.copy()
    out["g_local"] = 0.0
    # Iterate per rollout so step_index ordering is well-defined and missing
    # steps (if any) do not bleed across rollouts.
    for _, sub in out.groupby(["group_id", "rollout_index"], sort=False):
        order = sub["step_index"].astype(int).values.argsort()
        idx = sub.index.values[order]
        rewards = sub["step_reward"].astype(float).values[order]
        g = np.zeros(len(rewards), dtype=float)
        running = 0.0
        for k in range(len(rewards) - 1, -1, -1):
            running = float(rewards[k]) + gamma_r * running
            g[k] = running
        out.loc[idx, "g_local"] = g
    return out

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
          - advantages         : list[float] of len(input_ids)
          - episode_advantage  : float per row (terminal-reward baseline component)
          - step_advantage     : float per row (0.0 if estimator has no notion)
        Estimators may attach extra debug columns; the dispatcher will merge
        them back onto the main df by the three-key prefix.
        """
        ...
