# Potential-based local reward shaping for MiniWob++

This package implements deterministic, potential-based local reward shaping
for MiniWob++ HTML-only browser tasks. The terminal MiniWob success reward
is **unchanged** — this only contributes a dense per-step shaping signal
that the RL trainer combines with the terminal reward.

## Pipeline

```
instruction + initial DOM
        ↓
   GoalSpec DSL
        ↓
deterministic predicate evaluators
        ↓
       Phi(state)
        ↓
local_reward = γ · Phi(next_state) − Phi(prev_state)
```

Because the local reward is a pure potential difference (Ng, Harada, Russell
1999), the optimal policy under the shaped reward equals the optimal policy
under the unshaped (terminal-only) reward. In particular, the agent does
**not** receive positive reward for actions that merely preserve a predicate
that was already satisfied before the action.

## Phi

```
Phi(s) =
    w_constraints  · constraint_score(s)
  + w_terminal     · terminal_score(s)
  − w_forbidden    · violation_score(s)
  − w_error        · failed_tool_count(s)
  − w_noop         · noop_count(s)
```

Defaults: `w_constraints=0.3, w_terminal=0.7, w_forbidden=0.3, w_error=0.1,
w_noop=0.05, γ=1.0`.

## Public API

```python
from miniwob_cube.shaping import (
    build_goal_spec, build_state, compute_local_reward,
    RewardWeights, ToolResult,
)

goal = build_goal_spec(task_text, initial_html)

prev_state = build_state(prev_html, env_metadata={"failed_tool_count": ...})
next_state = build_state(next_html, env_metadata={"failed_tool_count": ...,
                                                  "terminal_success": True})

info = compute_local_reward(
    goal=goal,
    prev_state=prev_state,
    next_state=next_state,
    tool_result=ToolResult(failed=False),  # or failed=True / is_noop=True
    weights=RewardWeights(),
    gamma=1.0,
)
print(info.reward, info.reasons)
```

`info` is a `LocalRewardInfo` dataclass with per-component before/after
scores and a `reasons` list for debugging.

## Supported task families (high-confidence rule parser)

* Checkbox tasks: `"Select X and Y and click Submit"`, `"Select nothing and click Submit"`.
* Pure click tasks: `"Click X"`, `"Click the Submit button"`, `"Click OK"`.
* Text input tasks: `"Type hello"`, `"Enter hello into the text field"`.
* Dropdown/select tasks: `"Choose red"`, `"Select red from the dropdown"`.

Any other instruction shape falls through to `safe_fallback()`, which is a
GoalSpec with `confidence=0.0` and `terminal=TerminalSuccess()` only. The
shaper will then contribute *only* the terminal-success bonus — it never
invents dense rewards when uncertain.

## Integration with PipelineRL training

`MiniWobTask.evaluate(obs)` calls `compute_local_reward(...)` once per step
(when `enable_step_verifier_rewards=True`) and writes the result into
`info["step_reward"]`. The cube `result_builder.py` aggregates this into
`TrainingText.step_reward`, and the `grpo_loo` advantage estimator combines
it with the episode advantage:

```
A_step = z_group(final_reward) + λ_local · z_group_step(step_reward)
```

where `λ_local = finetune.rl.advantage.step_reward_lambda` (default `0.2`).

To **disable shaping entirely** without changing any training code, set
`enable_step_verifier_rewards: false` in the benchmark config. To keep the
env-side signal but ignore it during training, set
`finetune.rl.advantage.step_reward_lambda: 0.0`.

## Inspection

Each training step records:

* `training_text.step_reward` — the scalar local reward at that step.
* `training_text.metadata["local_reward_info_per_step"]` — per-step
  `LocalRewardInfo.to_dict()` payloads, including the before/after
  component scores and `reasons`.
* `training_text.metadata["sum_local_reward"]` — Σ local reward across the
  rollout.
* `training_text.metadata["shaped_return"]` — `final_reward + sum_local_reward`.

## Running tests

```
uv run --extra cube python -m pytest dev_cubes/miniwob/tests/test_shaping.py -v
```

## Guardrails

* No LLM judge or LLM parser anywhere in this pipeline.
* No per-task answer keys — the parser only uses the instruction text and
  the initial DOM.
* `compute_local_reward` always returns a potential difference. It does not
  reward a predicate merely for being satisfied.
* `safe_fallback()` is the default whenever rule parsing is not confident.
* Reward weights are configurable per benchmark instance.
