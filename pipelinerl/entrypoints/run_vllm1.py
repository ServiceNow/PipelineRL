from pipelinerl.vllm1 import run_llm
from pipelinerl.utils import better_crashing

if __name__ == "__main__":
    with better_crashing("llm"):
        run_llm()
