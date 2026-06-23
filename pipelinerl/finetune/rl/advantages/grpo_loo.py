"""GRPO with a leave-one-out baseline — PipelineRL's current default.

Episode advantage per rollout:
    A_i = (R_i - mean_{j!=i}(R_j))                 if not divide_by_std
    A_i = (R_i - mean_{j!=i}(R_j)) / (std(R)+1e-4) if divide_by_std

This episode advantage is broadcast to every token of every step in the
rollout. If `step_reward` is present, the per-step `step_reward` is centered
within its group (mean over all steps in the group_id) and added to the
broadcast episode advantage, scaled by `step_reward_lambda`:

    step_advantage = λ * (step_reward - mean_{rows in group}(step_reward))

Group-centering mirrors the LOO baseline on the episode side: it removes
any per-batch bias in raw step_reward (which is often systematically
positive for progress-style shaping), so the step term has zero mean within
each group and only contributes a *relative* signal.
"""
from __future__ import annotations

from typing import ClassVar, Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig


class GrpoLooConfig(BaseModel):
    type: Literal["grpo_loo"] = "grpo_loo"
    step_reward_lambda: float = Field(
        default=0.2,
        description="Coefficient on the per-step `step_reward` term added to the episode advantage.",
    )


class GrpoLoo:
    name: ClassVar[str] = "grpo_loo"
    required_columns: ClassVar[tuple[str, ...]] = (
        "group_id",
        "rollout_index",
        "step_index",
        "rewards",
    )

    def compute(self, df: pd.DataFrame, config: "RLConfig") -> pd.DataFrame:
        cfg = config.advantage
        assert isinstance(cfg, GrpoLooConfig), f"Expected GrpoLooConfig, got {type(cfg).__name__}"

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
            group_step_mean = (
                df_adv.groupby("group_id")["step_reward"]
                .mean()
                .rename("group_step_mean")
                .reset_index()
            )
            df_adv = df_adv.merge(group_step_mean, on="group_id", how="left")
            centered = df_adv["step_reward"].astype(float) - df_adv["group_step_mean"].astype(float)
            df_adv["step_advantage"] = cfg.step_reward_lambda * centered
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
                episode_adv = (current - loo_mean) / (np.nan_to_num(std) + 1e-4)
            else:
                episode_adv = current - loo_mean
            return [episode_adv + step_adv for _ in rewards]

        df_adv["advantages"] = df_adv.apply(_calc, axis=1)
        return df_adv[["group_id", "rollout_index", "step_index", "advantages", "step_advantage"]]
