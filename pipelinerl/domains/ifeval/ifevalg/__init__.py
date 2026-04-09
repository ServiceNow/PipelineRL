"""IFEvalG - Extended IFEval verification from AllenAI open-instruct.

Source: https://github.com/allenai/open-instruct/tree/main/open_instruct/IFEvalG
License: Apache 2.0

This module supports 67 constraint types vs the original Google IFEval's 25.
"""

from .instructions_registry import INSTRUCTION_DICT

__all__ = ["INSTRUCTION_DICT"]
