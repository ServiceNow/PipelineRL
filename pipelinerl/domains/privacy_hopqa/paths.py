"""Default local paths for optional Privacy HopQA resources."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRIVACY_HOPQA_ROOT = REPO_ROOT / ".privacy_hopqa"
DEFAULT_INPUTS_ROOT = DEFAULT_PRIVACY_HOPQA_ROOT / "inputs"
DEFAULT_TASK_DATA_ROOT = DEFAULT_INPUTS_ROOT / "tasks"
DEFAULT_LOCAL_INDEX_ROOT = DEFAULT_PRIVACY_HOPQA_ROOT / "local_indices"
