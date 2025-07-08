import os
os.environ["VLLM_USE_V1"] = "0"
from pipelinerl.vllm0 import run_llm
from pipelinerl.utils import better_crashing

if __name__ == "__main__":
    with better_crashing("llm"):
        run_llm()
