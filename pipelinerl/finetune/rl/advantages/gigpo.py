"""Group-in-Group Policy Optimization (GiGPO).

NeurIPS 2025, Feng et al. — https://github.com/langfengQ/verl-agent

Two-level advantage:
  - Episode advantage A_E: normalize trajectory returns within an episode
    group (the standard GRPO baseline, mean or mean/std).
  - Step advantage A_S: cluster steps across sibling rollouts by anchor
    observation (within the same group_id), then normalize the trajectory
    returns *inside each cluster*. This is what gives step-level credit
    assignment over revisited states.

Combined per row: A = A_E + w_S * A_S, broadcast to every token of input_ids.

Notes:
  - We use the trajectory's terminal reward (rewards[0]) as the step-level
    signal. With sparse terminal rewards this equals the MC return-to-go
    from every step (γ=1). For dense per-step shaping signals use grpo_loo,
    which baselines `step_reward` separately — the two estimators answer
    different questions and aren't meant to compose.
  - Step-groups of size 1 (anchors that didn't recur across siblings)
    contribute step_advantage=0: no comparative signal to extract.
  - Empty anchor_obs (domains that don't emit one) also yield 0.
"""
from __future__ import annotations

from typing import ClassVar, Literal, TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig


class GigpoConfig(BaseModel):
    type: Literal["gigpo"] = "gigpo"
    step_advantage_w: float = Field(default=1.0, description="Weight w_S on the step-level term.")
    normalize: Literal["mean", "mean_std"] = Field(
        default="mean",
        description="Whether to also divide by std after centering (both levels).",
    )


class Gigpo:
    name: ClassVar[str] = "gigpo"
    required_columns: ClassVar[tuple[str, ...]] = (
        "group_id",
        "rollout_index",
        "step_index",
        "rewards",
        "anchor_obs",
        "input_ids",
    )

    def compute(self, df: pd.DataFrame, config: "RLConfig") -> pd.DataFrame:
        cfg = config.advantage
        assert isinstance(cfg, GigpoConfig), f"Expected GigpoConfig, got {type(cfg).__name__}"

        df = df[
            ["group_id", "rollout_index", "step_index", "rewards", "anchor_obs", "input_ids"]
        ].copy()
        df["traj_return"] = df["rewards"].apply(lambda x: float(x[0]))
        assert df.groupby(["group_id", "rollout_index"])["traj_return"].nunique().max() == 1, (
            "Terminal `rewards` differ across steps within a rollout; GiGPO assumes "
            "trajectory-uniform terminal reward — put per-step signal in `step_reward` instead."
        )

        # Episode advantage: one return per rollout, baseline within group_id.
        rollout_returns = df[df["step_index"] == 0][["group_id", "rollout_index", "traj_return"]]
        ep_stats = (
            rollout_returns.groupby("group_id")["traj_return"]
            .agg(ep_mean="mean", ep_std="std")
            .reset_index()
        )
        df = df.merge(ep_stats, on="group_id", how="left")
        if cfg.normalize == "mean_std":
            df["episode_advantage"] = (df["traj_return"] - df["ep_mean"]) / (
                df["ep_std"].fillna(0.0) + 1e-6
            )
        else:
            df["episode_advantage"] = df["traj_return"] - df["ep_mean"]

        # Step advantage: cluster by (group_id, anchor_obs); normalize traj_return inside.
        df["step_group"] = (
            df["group_id"].astype(str) + "\x1f" + df["anchor_obs"].fillna("").astype(str)
        )
        sg_stats = (
            df.groupby("step_group")["traj_return"]
            .agg(sg_mean="mean", sg_std="std", sg_count="count")
            .reset_index()
        )
        df = df.merge(sg_stats, on="step_group", how="left")
        if cfg.normalize == "mean_std":
            step_adv = (df["traj_return"] - df["sg_mean"]) / (df["sg_std"].fillna(0.0) + 1e-6)
        else:
            step_adv = df["traj_return"] - df["sg_mean"]
        no_signal = (df["sg_count"] < 2) | (df["anchor_obs"].fillna("") == "")
        step_adv = step_adv.where(~no_signal, 0.0)
        df["step_advantage"] = step_adv.astype(float)

        scalar_adv = df["episode_advantage"] + cfg.step_advantage_w * df["step_advantage"]
        df["advantages"] = [
            [float(a)] * len(ids) for a, ids in zip(scalar_adv.tolist(), df["input_ids"].tolist())
        ]
        return df[
            ["group_id", "rollout_index", "step_index", "advantages", "episode_advantage", "step_advantage"]
        ]
