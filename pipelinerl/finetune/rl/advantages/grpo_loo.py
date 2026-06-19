"""GRPO with a leave-one-out baseline — PipelineRL's current default.

Episode advantage per rollout:
    A_i = (R_i - mean_{j!=i}(R_j))                 if not divide_by_std
    A_i = (R_i - mean_{j!=i}(R_j)) / (std(R)+1e-4) if divide_by_std

If `step_reward` is present, an experimental rollout-level shaping baseline
is added: collapse step_reward to a per-rollout mean, subtract the per-group
mean, and add that residual to every token of the rollout. Sibling rollouts
in a group are only prefix-matched at step 0, so we don't compare step k
across siblings — credit assignment within a rollout stays uniform. For
true step-level credit assignment over revisited states, use GiGPO.
"""
from __future__ import annotations

from typing import ClassVar, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig


class GrpoLooConfig(BaseModel):
    type: Literal["grpo_loo"] = "grpo_loo"


class GrpoLoo:
    name: ClassVar[str] = "grpo_loo"
    required_columns: ClassVar[tuple[str, ...]] = (
        "group_id",
        "rollout_index",
        "step_index",
        "rewards",
    )

    def compute(self, df: pd.DataFrame, config: "RLConfig") -> pd.DataFrame:
        df_stats = df[["group_id", "rollout_index", "step_index"]].copy()
        df_stats["rollout_reward"] = df["rewards"].apply(lambda x: x[0])
        assert df_stats.groupby(["group_id", "rollout_index"])["rollout_reward"].nunique().max() == 1, (
            "Terminal `rewards` differ across steps within a rollout; per-step signal belongs in `step_reward`."
        )
        df_stats = df_stats[df_stats["step_index"] == 0].drop(columns=["step_index"])
        df_grouped = (
            df_stats.groupby("group_id")
            .agg(
                rollout_reward_sum=("rollout_reward", "sum"),
                rollout_reward_count=("rollout_reward", "count"),
                rollout_reward_std=("rollout_reward", "std"),
            )
            .reset_index()
        )

        init_cols = ["group_id", "rollout_index", "step_index", "rewards"]
        has_step_reward = "step_reward" in df.columns
        if has_step_reward:
            init_cols.append("step_reward")
        df_adv = pd.merge(df[init_cols], df_grouped, on="group_id", how="left")

        if has_step_reward:
            rollout_step = (
                df_adv.groupby(["group_id", "rollout_index"])["step_reward"]
                .mean()
                .rename("rollout_step_mean")
                .reset_index()
            )
            group_step = (
                rollout_step.groupby("group_id")["rollout_step_mean"]
                .mean()
                .rename("group_step_mean")
                .reset_index()
            )
            df_adv = df_adv.merge(rollout_step, on=["group_id", "rollout_index"], how="left")
            df_adv = df_adv.merge(group_step, on="group_id", how="left")
            df_adv["step_advantage"] = df_adv["rollout_step_mean"] - df_adv["group_step_mean"]
        else:
            df_adv["step_advantage"] = 0.0

        divide_by_std = config.divide_advantage_by_std

        def _calc(row):
            rewards = row["rewards"]
            step_adv = float(row["step_advantage"])
            group_sum = row["rollout_reward_sum"]
            group_count = row["rollout_reward_count"]
            current = rewards[0]
            if group_count > 1:
                loo_mean = (group_sum - current) / (group_count - 1)
            else:
                loo_mean = current
            std = row["rollout_reward_std"]
            if divide_by_std:
                return [(r - loo_mean) / (np.nan_to_num(std) + 1e-4) + step_adv for r in rewards]
            return [(r - loo_mean) + step_adv for r in rewards]

        df_adv["advantages"] = df_adv.apply(_calc, axis=1)
        return df_adv[["group_id", "rollout_index", "step_index", "advantages", "step_advantage"]]
