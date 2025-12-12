"""
Personal configuration for EAI cluster launching.

Edit this file with your personal settings.
"""

# Your username (used for job naming)
PERSONAL_ACCOUNT_NAME_SUFFIX = "dheeraj_vattikonda"

# Team account for resources (adea, ui_assist, adea_low)
TEAM_ACCOUNT_NAME_SUFFIX = "adea"


# Data volumes to mount
DATAS_TO_MOUNT = [
    "adea",
    "ui_assist",
]

# Home data settings
HOME_TEAM_ACCOUNT_NAME_SUFFIX = "adea"
HOME_DATA_NAME_SUFFIX = "dheeraj_home"

# Local paths
HOME_MOUNT_PATH = "/home/toolkit"
PIPELINERL_PATH = "/home/toolkit/ui-copilot/PipelineRL"
