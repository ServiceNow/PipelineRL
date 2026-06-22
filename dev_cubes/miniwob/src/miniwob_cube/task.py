import logging
from typing import Any

from cube.benchmark import RuntimeContext
from cube.core import Content, Observation
from cube.task import Task, TaskConfig, TaskMetadata  # noqa: F401 — TaskMetadata kept for typing
from cube.tools.browser import BrowserTool
from PIL import Image


class MiniWobTaskMetadata(TaskMetadata):
    """TaskMetadata subclass for MiniWob++ tasks.
    Adds cube-specific public fields that are safe to ship in task_metadata.json.
    """

    nondeterministic: bool = False


logger = logging.getLogger(__name__)


class MiniWobTask(Task):
    validate_per_step: bool = True
    base_url: str = "http://localhost:8000/miniwob"
    remove_human_display: bool = True
    episode_max_time: int = 1000000

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
        obs = Observation.from_text(goal) + self.obs_postprocess(self.tool.page_obs())
        return obs, {**info, "task_id": self.id, "task_url": self.url, "goal": goal}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        result = self.tool.evaluate_js("""() => {
return [WOB_REWARD_GLOBAL, WOB_RAW_REWARD_GLOBAL, WOB_REWARD_REASON, WOB_DONE_GLOBAL, WOB_EPISODE_ID, WOB_TASK_READY];}""")
        return _parse_validation_result(result)

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
