"""Canonical MiniWoB benchmark configs.

    from miniwob_cube import MINIWOB_CONFIGS
    benchmark = MINIWOB_CONFIGS["default"]

MiniWoB runs on BrowserGym with html + screenshot (html, not axtree, is the
canonical MiniWoB observation).
"""

from cube.core import ConfigRegistry
from cube_browser_tool.bgym_tool import BgymToolConfig
from miniwob_cube.benchmark import MiniWobBenchmarkConfig

MINIWOB_CONFIGS: ConfigRegistry[MiniWobBenchmarkConfig] = ConfigRegistry(
    {
        "default": MiniWobBenchmarkConfig(
            tool_config=BgymToolConfig(use_html=True, use_axtree=False, use_screenshot=True)
        ),
    }
)
