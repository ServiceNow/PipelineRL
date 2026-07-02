"""Group-in-Group Policy Optimization (GiGPO).

NeurIPS 2025, Feng et al. — https://github.com/langfengQ/verl-agent

Two-level advantage:
  - Episode advantage A_E: leave-one-out baseline over sibling trajectory
    returns in the same group_id (matches grpo_loo). Diverges from paper
    Eq. 3, which uses a self-inclusive group mean; LOO is unbiased and
    keeps the estimator comparable to PipelineRL's grpo_loo default.
  - Step advantage A_S: cluster steps across sibling rollouts by anchor
    observation (within the same group_id), then normalize the *return-to-go*
    R_t = sum_{k>=t} gamma^{k-t} r_k *inside each cluster* (paper Eq. 7).
    This is what gives step-level credit assignment over revisited states.

Combined per row: A = A_E + w_S * A_S, broadcast to every token of input_ids.

Per-step reward used for R_t: `step_reward[k]` when present, with the
trajectory-terminal `rewards[0]` added to the last step of the rollout.
When `step_reward` is absent and gamma=1, R_t collapses to the terminal
reward for every t (i.e. the sparse-reward case).

Notes:
  - Step-groups of size 1 (anchors that didn't recur across siblings)
    contribute step_advantage=0: no comparative signal to extract.
  - Empty anchor_obs (domains that don't emit one) also yield 0.
"""
from __future__ import annotations

from typing import ClassVar, Literal, TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from .base import _attach_local_return_to_go

if TYPE_CHECKING:
    from pipelinerl.finetune.rl import RLConfig


class GigpoConfig(BaseModel):
    type: Literal["gigpo"] = "gigpo"
    step_reward_lambda: float = Field(
        default=1.0,
        description="Coefficient `lambda_local` on the step-level G_local term added to the episode advantage.",
    )
    step_reward_gamma: float = Field(
        default=1.0,
        description=(
            "Discount factor gamma used to compute the return-to-go "
            "R_t = sum_{k>=t} gamma^{k-t} * r_k for the step-level advantage."
        ),
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

        cols = ["group_id", "rollout_index", "step_index", "rewards", "anchor_obs", "input_ids"]
        has_step_reward = "step_reward" in df.columns
        if has_step_reward:
            cols.append("step_reward")
        df = df[cols].copy()
        df["traj_return"] = df["rewards"].apply(lambda x: float(x[0]))
        assert df.groupby(["group_id", "rollout_index"])["traj_return"].nunique().max() == 1, (
            "Terminal `rewards` differ across steps within a rollout; GiGPO assumes "
            "trajectory-uniform terminal reward — put per-step signal in `step_reward` instead."
        )

        divide_by_std = config.divide_advantage_by_std

        # Episode advantage: leave-one-out baseline over sibling trajectory
        # returns in the same group_id (matches grpo_loo). When count==1 the
        # LOO mean falls back to the sample itself → advantage 0.
        rollout_returns = df[df["step_index"] == 0][["group_id", "rollout_index", "traj_return"]]
        ep_stats = (
            rollout_returns.groupby("group_id")["traj_return"]
            .agg(ep_sum="sum", ep_count="count", ep_std="std")
            .reset_index()
        )
        df = df.merge(ep_stats, on="group_id", how="left")
        loo_mean = (df["ep_sum"] - df["traj_return"]) / (df["ep_count"] - 1).where(
            df["ep_count"] > 1, 1.0
        )
        loo_mean = loo_mean.where(df["ep_count"] > 1, df["traj_return"])
        if divide_by_std:
            df["episode_advantage"] = (df["traj_return"] - loo_mean) / (
                df["ep_std"].fillna(0.0) + 1e-6
            )
        else:
            df["episode_advantage"] = df["traj_return"] - loo_mean

        # Return-to-go R_t for the step-level advantage (paper Eq. 7).
        # Effective per-step reward = step_reward[k] (0 if absent), with the
        # trajectory-terminal reward added to the last step of each rollout.
        # Grpo_loo's helper reads from `step_reward`, so we materialize an
        # effective column that includes the terminal on the last step.
        last_step = (
            df.groupby(["group_id", "rollout_index"])["step_index"].transform("max")
        )
        is_last = df["step_index"] == last_step
        base_step_r = df["step_reward"].astype(float) if has_step_reward else 0.0
        effective = base_step_r + is_last.astype(float) * df["traj_return"].astype(float)
        rtg_input = df.assign(step_reward=effective)
        rtg = _attach_local_return_to_go(rtg_input, float(cfg.step_reward_gamma))
        df["return_to_go"] = rtg["g_local"].astype(float)

        # Step advantage: cluster by (group_id, anchor_obs); normalize return-to-go inside.
        df["step_group"] = (
            df["group_id"].astype(str) + "\x1f" + df["anchor_obs"].fillna("").astype(str)
        )
        sg_stats = (
            df.groupby("step_group")["return_to_go"]
            .agg(sg_mean="mean", sg_std="std", sg_count="count")
            .reset_index()
        )
        df = df.merge(sg_stats, on="step_group", how="left")
        if divide_by_std:
            step_adv = (df["return_to_go"] - df["sg_mean"]) / (df["sg_std"].fillna(0.0) + 1e-6)
        else:
            step_adv = df["return_to_go"] - df["sg_mean"]
        no_signal = (df["sg_count"] < 2) | (df["anchor_obs"].fillna("") == "")
        step_adv = step_adv.where(~no_signal, 0.0)
        df["step_advantage"] = step_adv.astype(float)

        scalar_adv = df["episode_advantage"] + cfg.step_reward_lambda * df["step_advantage"]
        df["advantages"] = [
            [float(a)] * len(ids) for a, ids in zip(scalar_adv.tolist(), df["input_ids"].tolist())
        ]
        return df[
            ["group_id", "rollout_index", "step_index", "advantages", "episode_advantage", "step_advantage"]
        ]
