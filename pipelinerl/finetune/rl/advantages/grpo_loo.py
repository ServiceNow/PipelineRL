"""GRPO with a leave-one-out baseline — PipelineRL's current default.

Episode advantage per rollout:
    A_i = (R_i - mean_{j!=i}(R_j))                 if not divide_by_std
    A_i = (R_i - mean_{j!=i}(R_j)) / (std(R)+1e-4) if divide_by_std

This episode advantage is broadcast to every token of every step in the
rollout. If `step_reward` is present, the per-step local rewards are
converted to a per-step *local return-to-go*:

    G_local[t] = sum_{u=t}^{T-1} gamma_r^{u-t} * step_reward[u]

and the step advantage is the group-z-normalized G_local scaled by
`step_reward_lambda`:

    A_step[t] = lambda_local * z_group_step(G_local[t])

`step_advantage_centering` chooses the centering axis for `z_group_step`:
  - "step"    : (group_id, step_index) — subtract the mean over sibling
                rollouts at the SAME step index. This is the literal
                step-aligned z-normalization the spec describes.
  - "group"   : mean over all rows in the group_id (the original behavior).
  - "rollout" : mean over all steps in this trajectory only — measures
                "which step inside this rollout was unusually good".

Combined per row (broadcast to every token of input_ids):

    A = A_E + lambda_local * z_group_step(G_local[t])
      = z_group(final_reward) + lambda_local * z_group_step(G_local[t])

Both `episode_advantage` and `step_advantage` are returned as separate
per-row scalar columns so they can be monitored independently downstream.
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
        description="Coefficient `lambda_local` on the step-level G_local term added to the episode advantage.",
    )
    step_reward_gamma: float = Field(
        default=1.0,
        description=(
            "Discount factor `gamma_r` used to compute the local return-to-go "
            "G_local[t] = sum_{u>=t} gamma_r^{u-t} * step_reward[u]."
        ),
    )
    step_advantage_centering: Literal["step", "group", "rollout"] = Field(
        default="step",
        description=(
            "Axis used by z_group_step to center G_local[t] before scaling. "
            "'step' subtracts the mean over sibling rollouts at the SAME step "
            "index (the spec's z_group_step); 'group' subtracts the mean over "
            "all rows in the group_id; 'rollout' subtracts the within-rollout "
            "mean only."
        ),
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
            gamma_r = float(cfg.step_reward_gamma)
            df_adv = _attach_local_return_to_go(df_adv, gamma_r)

            if cfg.step_advantage_centering == "rollout":
                center_keys = ["group_id", "rollout_index"]
            elif cfg.step_advantage_centering == "step":
                center_keys = ["group_id", "step_index"]
            else:
                center_keys = ["group_id"]
            step_center = (
                df_adv.groupby(center_keys)["g_local"]
                .mean()
                .rename("g_local_center")
                .reset_index()
            )
            df_adv = df_adv.merge(step_center, on=center_keys, how="left")
            centered = df_adv["g_local"].astype(float) - df_adv["g_local_center"].astype(float)
            df_adv["step_advantage"] = cfg.step_reward_lambda * centered
        else:
            df_adv["g_local"] = 0.0
            df_adv["step_advantage"] = 0.0

        divide_by_std = config.divide_advantage_by_std

        def _episode_adv(row):
            rewards = row["rewards"]
            group_sum = row["rollout_reward_sum"]
            group_count = row["rollout_reward_count"]
            current = rewards[0]
            if group_count > 1:
                loo_mean = (group_sum - current) / (group_count - 1)
            else:
                loo_mean = current
            std = row["rollout_reward_std"]
            if divide_by_std:
                return (current - loo_mean) / (np.nan_to_num(std) + 1e-4)
            return current - loo_mean

        df_adv["episode_advantage"] = df_adv.apply(_episode_adv, axis=1).astype(float)
        df_adv["advantages"] = [
            [float(ep + step)] * len(rewards)
            for ep, step, rewards in zip(
                df_adv["episode_advantage"].tolist(),
                df_adv["step_advantage"].tolist(),
                df_adv["rewards"].tolist(),
            )
        ]
        return df_adv[
            [
                "group_id",
                "rollout_index",
                "step_index",
                "advantages",
                "episode_advantage",
                "step_advantage",
                "g_local",
            ]
        ]


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
