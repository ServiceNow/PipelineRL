"""Canonical WorkArena benchmark configs.

    from workarena_cube import WORKARENA_CONFIGS
    benchmark = WORKARENA_CONFIGS["l1"]

Runs on BrowserGym with axtree + screenshot (the canonical web observation).
"default" is all levels; "l1"/"l2"/"l3" are the canonical WorkArena
difficulty splits; l2/l3 add the infeasible-task tool, matching how the
benchmark is run in practice.
"""

from cube.tool import ToolboxConfig
from cube_browser_tool.bgym_tool import BgymToolConfig
from workarena_cube.tools import WorkArenaInfeasibleToolConfig


def _browser() -> BgymToolConfig:
    return BgymToolConfig(use_html=False, use_axtree=True, use_screenshot=True)


def _browser_with_infeasible() -> ToolboxConfig:
    return ToolboxConfig(tool_configs=[_browser(), WorkArenaInfeasibleToolConfig()])