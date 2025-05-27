import os
from pathlib import Path
import sys

from tapeagents.renderers.camera_ready_renderer import CameraReadyRenderer
from tapeagents.tape_browser import TapeBrowser
from pipelinerl.cot_math_agent import RLMathTape

# comment this code out if loading the prompt and completions takes too long for you
tape_dir = Path(sys.argv[1])
exp_dir = tape_dir
# try to find a parent directory for tape_dir path that contains llm_calls.sqlite
while not os.path.exists(exp_dir / "llm_calls.sqlite") and exp_dir != Path("."):
    exp_dir = exp_dir.parent
os.environ["TAPEAGENTS_SQLITE_DB"] = os.path.join(exp_dir, "llm_calls.sqlite")


browser = TapeBrowser(
    RLMathTape, sys.argv[1], CameraReadyRenderer(), file_extension=".json"
)
browser.launch(port=7680 if len(sys.argv) < 3 else int(sys.argv[2]))
