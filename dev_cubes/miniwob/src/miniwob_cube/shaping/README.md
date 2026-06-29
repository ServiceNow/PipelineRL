# Potential-based local reward shaping for MiniWob++

Deterministic, **instruction-blind** potential-based reward shaping for
MiniWob++ HTML-only browser tasks. The shaper never reads the task
instruction — every signal is computed from DOM state and last-action
quality.

## Pipeline

```
DOM snapshot + last action + tool result + env success
        ↓
generic progress signals  (Tier 1 + Tier 2, no instruction parsing)
        ↓
   Φ(state, counters)
        ↓
local_reward = γ · Φ(next_state, next_counters) − Φ(prev_state, prev_counters)
```

Because the local reward is a pure potential difference (Ng/Harada/Russell
1999), the optimal policy under the shaped reward equals the optimal policy
under the unshaped (terminal-only) reward.

## Φ

```
Φ(s, h) =
    w_terminal              · 1[terminal_success(s)]
  + w_form_completion       · form_completion_fraction(s)
  + w_affordance_breadth    · affordance_engagement_breadth(h)
  − w_error                 · failed_tool_count(h)
  − w_stuck                 · stuck_count(h)
  − w_bad_target            · bad_target_count(h)
```

`h` is an episode-cumulative `HistoryCounters` bundle held inside
`EpisodeShaper`. Φ is a pure function of `(state, counters, weights)`.
The action itself never enters Φ.

## Signals

### Tier 1 — state invariants (DOM only, no action)

- **`form_completion_fraction(s)`** — fraction of visible text inputs that
  hold a non-empty value. Returns `0` when no text inputs exist (signal
  disabled for non-form tasks).
- **`affordance_engagement_breadth(h)`** — fraction of initial-state
  interactive elements whose state has been touched, capped at `1.0`.

### Tier 2 — action quality (DOM + last action, no instruction)

- **`failed_tool_count`** — cumulative count of tool calls that returned a
  `StepError` (e.g. Playwright `Locator.clear` on a checkbox).
- **`stuck_count`** — cumulative count of "should-change" actions
  (`click`, `fill`, `check`, `select_option`, …) that left the DOM
  signature unchanged. Excluded: `focus`, `hover`, `scroll`, `press`,
  `wait`, and any failed action (already debited as a failure).
- **`bad_target_count`** — cumulative count of actions targeting a `bid`
  that is missing, disabled, or non-interactive for click-like actions.

## Public API

```python
from miniwob_cube.shaping import (
    EpisodeShaper, RewardWeights, ToolResult, ActionView,
)

shaper = EpisodeShaper(
    initial_html=initial_html,
    weights=RewardWeights(),
    gamma=1.0,
)

# Per agent step:
info = shaper.step(
    next_html=current_html,
    action=ActionView.from_cube_action(last_action),
    tool_result=ToolResult(failed=False, error_message=None),
    terminal_success=False,
)
print(info.reward, info.reasons)
print(info.to_dict())   # full debug bundle
```

`info` is a `LocalRewardInfo` with per-component before/after scores and
a `reasons` list for debugging.

## Default weights

```python
RewardWeights(
    enable_step_verifier_rewards=False,
    terminal=0.7,
    form_completion=0.15,
    affordance_breadth=0.15,
    error=0.1,
    stuck=0.05,
    bad_target=0.1,
    gamma=1.0,
)
```

`enable_step_verifier_rewards=False` disables shaping entirely at the task
layer — flip it to `true` in `conf/miniwob_cube.yaml` to opt in.

## Integration with PipelineRL training

`MiniWobTask.evaluate(obs)` invokes `EpisodeShaper.step(...)` once per
step (when `enable_step_verifier_rewards=True`) and writes the result
into `info["step_reward"]` + `info["step_reward_info"]`. The cube
`result_builder.py` aggregates this into `TrainingText.step_reward`, and
the `grpo_loo` advantage estimator combines it with the episode advantage:

```
A_step = z_group(R_final) + λ_local · z_group_step(G_local[t])

G_local[t] = Σ_{u≥t} γ_r^(u−t) · step_reward[u]
```

where `λ_local = step_reward_lambda` (default `0.1`) and
`γ_r = step_reward_gamma` (default `1.0`).

**Three knobs, three jobs:**

| Knob | Layer | Purpose |
|---|---|---|
| `shaping_gamma` | env | one-step γ in `γ·Φ(s')−Φ(s)`. Set to 1.0 to match GRPO's undiscounted return. |
| `step_reward_gamma` | trainer | discount in the local return-to-go `G_local[t]`. Lower it to localize credit. |
| `step_reward_lambda` | trainer | weight of shaping vs terminal in the advantage. Lower to dominate-by-terminal. |

To **disable shaping entirely** without changing training: set
`enable_step_verifier_rewards: false`. To keep the env signal but drop it
from training: set `step_reward_lambda: 0.0`.

## Inspection

Each training step records:

- `training_text.step_reward` — scalar local reward at that step.
- `training_text.metadata["local_reward_info_per_step"]` — per-step
  `LocalRewardInfo.to_dict()` bundles with before/after component scores.
- `training_text.metadata["sum_local_reward"]` — Σ local reward across the
  rollout.
- `training_text.metadata["shaped_return"]` — `final_reward + sum_local_reward`.

## Running tests

```
uv run --extra cube python -m pytest dev_cubes/miniwob/tests/test_shaping.py -v
```

## Guardrails

- **No instruction parsing.** The shaper does not read the task text. No
  GoalSpec, no constraint DSL, no per-task answer keys.
- **No LLM judge.** Every signal is a deterministic Python function.
- **Pure potential difference.** Local reward is always
  `γ · Φ(next) − Φ(prev)`. A state predicate that was already satisfied
  at the previous step contributes zero to the new reward.
- **Cumulative counters are monotone non-decreasing.** A failure debits Φ
  at the next step but never inflates a later reward.
- **Easy to disable.** Set `enable_step_verifier_rewards: false` (env) or
  `step_reward_lambda: 0.0` (trainer).
