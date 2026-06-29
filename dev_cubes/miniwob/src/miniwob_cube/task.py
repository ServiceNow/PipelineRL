import logging
from typing import Any

from cube.benchmark import RuntimeContext
from cube.core import Action, Content, Observation
from cube.task import Task, TaskConfig, TaskMetadata  # noqa: F401 — TaskMetadata kept for typing
from cube.tools.browser import BrowserTool
from PIL import Image
from pydantic import PrivateAttr
from miniwob_cube.shaping import RewardWeights


class MiniWobTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for MiniWob++ tasks.
    Adds cube-specific public fields that are safe to ship in task_metadata.json.
    """

    nondeterministic: bool = False


logger = logging.getLogger(__name__)


def _obs_text(obs: Observation | None, content_name: str) -> str | None:
    """Return the first content payload from ``obs`` with the given ``name``."""
    if obs is None:
        return None
    for c in obs.contents:
        if c.name == content_name and isinstance(c.data, str):
            return c.data
    return None


def _last_bgym_action(tool: Any) -> Action | None:
    """Return the agent's most recent cube `Action` if the inner tool records it.

    `MiniWobBgymTool` stores the cube `Action` on `agent_last_action` before
    delegating to `BgymTool.execute_action`, so it survives the in-method
    `page_obs()` reset of `_last_info`. Returns `None` for plain `BgymTool`
    instances (or just after `reset()`); the verifier handles that gracefully.
    """
    return getattr(tool, "agent_last_action", None)


class MiniWobTask(Task):
    validate_per_step: bool = True
    base_url: str = "http://localhost:8000/miniwob"
    remove_human_display: bool = True
    episode_max_time: int = 1000000

    # Auxiliary step shaping reward. The terminal MiniWob success reward
    # remains the source of truth — these signals must be combined with it.
    step_reward_weights: RewardWeights = RewardWeights(
        enable_step_verifier_rewards=False,
        terminal=0.7,
        form_completion=0.15,
        affordance_breadth=0.15,
        error=0.1,
        stuck=0.05,
        bad_target=0.1,
        gamma=1.0,
    )

    _goal: str = PrivateAttr(default="")
    _last_html: str | None = PrivateAttr(default=None)
    _last_axtree: str | None = PrivateAttr(default=None)
    _step_step_index: int = PrivateAttr(default=0)
    # Episode-scoped potential-based reward shaper — built fresh on every
    # reset() and consumed once per evaluate() call.
    _shaper: Any | None = PrivateAttr(default=None)

    @property
    def tool(self) -> BrowserTool:  # type: ignore[override]
        return self._tool  # type: ignore[return-value]

    @property
    def url(self) -> str:
        return f"{self.base_url}/{self.metadata.id}.html"

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        self.tool.goto(self.url)
        setup_result = self.tool.evaluate_js(_build_setup_js(self.remove_human_display, self.episode_max_time))
        goal, info = _parse_setup_result(setup_result)
        page_obs = self.tool.page_obs()
        obs = Observation.from_text(goal) + self.obs_postprocess(page_obs)
        # Track post-reset observation for local-verifier transitions.
        self._goal = goal
        self._last_html = _obs_text(page_obs, "pruned_html")
        self._last_axtree = _obs_text(page_obs, "axtree_txt")
        self._step_step_index = 0
        # Build a fresh per-episode shaper from the initial DOM. We capture
        # the initial DOM here (not on the first evaluate() call) so the
        # affordance-breadth denominator is set BEFORE any agent action.
        from miniwob_cube.shaping import EpisodeShaper

        self._shaper = EpisodeShaper(
            initial_html=self._last_html,
            weights=self.step_reward_weights,
            gamma=self.step_reward_weights.gamma,
        )
        return obs, {**info, "task_id": self.id, "task_url": self.url, "goal": goal}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        result = self.tool.evaluate_js("""() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];}""")
        reward, info = _parse_validation_result(result)

        # Auxiliary, opt-in step shaping signal. Lives in ``info`` only — the
        # returned ``reward`` is the MiniWob terminal/JS reward, untouched.
        # The training pipeline can read ``info["step_reward"]`` and combine it
        # with ``reward`` using a small coefficient.
        #
        # NOTE: ``evaluate`` is invoked once per agent action by
        # ``MonitoredTool._post_execute_wrapping`` (see cube-harness'
        # ``tool.py``) when ``validate_per_step=True``. ``Task.step`` itself is
        # bypassed by the harness, so this is the canonical per-step hook.
        if self.step_reward_weights.enable_step_verifier_rewards:
            self._maybe_add_step_reward(obs, info)
        return reward, info

    def _maybe_add_step_reward(self, obs: Observation | None, info: dict[str, Any]) -> None:
        # Local import keeps the shaping code off the standard inference path.
        from miniwob_cube.shaping import ActionView, ToolResult

        try:
            if self._shaper is None:
                return
            curr_html = _obs_text(obs, "pruned_html")
            curr_axtree = _obs_text(obs, "axtree_txt")

            # Detect a failed tool call via the bgym tool's recorded step error.
            step_error = getattr(self.tool, "agent_last_step_error", None)
            failed = step_error is not None

            terminal_success = bool(float(info.get("raw_reward", 0.0) or 0.0) > 0.0)

            local = self._shaper.step(
                next_html=curr_html,
                action=ActionView.from_cube_action(_last_bgym_action(self.tool)),
                tool_result=ToolResult(
                    failed=failed,
                    error_message=step_error,
                ),
                terminal_success=terminal_success,
            )
            info["step_reward"] = local.reward
            info["step_reward_info"] = local.to_dict()
            self._last_html = curr_html
            self._last_axtree = curr_axtree
            self._step_step_index += 1
        except Exception:  # pragma: no cover — shaper must never break rollout
            logger.exception("step shaper failed; skipping step reward this step")

    def finished(self, obs: Observation | None = None) -> bool:
        return self.tool.evaluate_js("() => {return WOB_DONE_GLOBAL;}")

    def obs_postprocess(self, obs: Observation) -> Observation:
        contents = []
        for content in obs.contents:
            if content.name == "screenshot" and isinstance(content.data, Image.Image):
                # crop to 332x214 because this is the viewport size for MiniWob
                contents.append(Content.from_data(content.data.crop((0, 0, 332, 214)), name=content.name))
            else:
                contents.append(content)
        obs.contents = contents
        return obs


class MiniWobTaskConfig(TaskConfig[MiniWobTaskMetadata]):
    base_url: str = "http://localhost:8000/miniwob"
    remove_human_display: bool = True
    episode_max_time: int = 1000000

    # Auxiliary step shaping reward. The terminal MiniWob success reward
    # remains the source of truth — these signals must be combined with it.
    step_reward_weights: RewardWeights = RewardWeights(
        enable_step_verifier_rewards=False,
        terminal=0.7,
        form_completion=0.15,
        affordance_breadth=0.15,
        error=0.1,
        stuck=0.05,
        bad_target=0.1,
        gamma=1.0,
    )

    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> MiniWobTask:
        _ = runtime_context
        assert self.tool_config is not None, "tool_config must be set"
        return MiniWobTask(
            metadata=self.metadata,
            tool_config=self.tool_config,
            base_url=self.base_url,
            remove_human_display=self.remove_human_display,
            episode_max_time=self.episode_max_time,
            step_reward_weights=self.step_reward_weights,
        )


def _build_setup_js(remove_human_display: bool, episode_max_time: int) -> str:
    if remove_human_display:
        js = r"""
let __display_ids = ['reward-display', 'click-canvas', 'sync-task-cover'];
let __display_divs = {};
let __query_div_hidden_copy = null;

removeDisplay = function() {
  core.clearTimer();
  document.body.removeEventListener('click', core.canvasDrawClick);

  __query_div_hidden_copy = document.getElementById('query').cloneNode(true);
  document.getElementById('query').innerHTML = '';

  for (i in __display_ids) {
    elem_id = __display_ids[i];
    elem = document.getElementById(elem_id);
    // remove elem from the document
    elem.remove();
    // but keep it stored somewhere to bring back later
    __display_divs[elem_id] = elem;
  }
};

bringBackDisplay = function() {
  document.getElementById('query').innerHTML = __query_div_hidden_copy.innerHTML;
  for (var elem_id in __display_divs){
    document.body.appendChild(__display_divs[elem_id]);
  }
  core.createDisplay();
};

core.endEpisode_legacy = core.endEpisode;
core.startEpisodeReal_legacy = core.startEpisodeReal;
core.getUtterance_legacy = core.getUtterance;

core.getUtterance = function () {
  bringBackDisplay();
  utterance = core.getUtterance_legacy();
  removeDisplay();
  return utterance;
};

core.endEpisode = function(reward, time_proportional, reason){
  bringBackDisplay();
  core.endEpisode_legacy(reward, time_proportional, reason);
  removeDisplay();
};

core.startEpisodeReal = function() {
  bringBackDisplay();
  core.startEpisodeReal_legacy();
  removeDisplay();
};

removeDisplay();
"""
    else:
        js = ""
    js += f"""
Math.seedrandom(42);
core.EPISODE_MAX_TIME = {episode_max_time};
core.startEpisodeReal();
while (!WOB_TASK_READY) {{
  await new Promise(resolve => setTimeout(resolve, 100));
}}
return core.getUtterance();
    """
    return f"async () => {{{js}}}"


def _parse_setup_result(setup_result: str | dict) -> tuple[str, dict]:
    if isinstance(setup_result, dict):
        return setup_result["utterance"], {}
    elif isinstance(setup_result, str):
        return setup_result, {}
    else:
        raise ValueError(f"Unexpected setup_result type: {type(setup_result)}")


def _parse_validation_result(validation_result: str | dict | list) -> tuple[float, dict]:
    if isinstance(validation_result, list):
        chunks = validation_result
        done = chunks[3]
    elif isinstance(validation_result, dict):
        raise ValueError("Validation result as dict is not supported")
    else:
        chunks = [c.strip() for c in validation_result.split(",")]
        done = chunks[3].strip().lower() == "true"
    raw_reward = float(chunks[1])
    reward = float(raw_reward > 0)
    return reward, {
        "raw_reward": raw_reward,
        "reward_reason": chunks[2],
        "done": done,
    }
