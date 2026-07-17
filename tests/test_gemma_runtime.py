from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Gemma4Config,
)

from pipelinerl.finetune.checkpoints import get_auto_model_class
from pipelinerl.launch import _get_vllm_cache_env


def test_gemma4_is_registered_for_pipelinerl_model_classes():
    config_type = type(Gemma4Config())

    assert config_type in AutoModelForCausalLM._model_mapping
    assert config_type in AutoModelForImageTextToText._model_mapping
    assert get_auto_model_class("vision2seq-language-modeling") is AutoModelForImageTextToText


def test_vllm_compile_cache_is_isolated_per_engine(tmp_path: Path):
    actor_env = _get_vllm_cache_env(tmp_path, "actor_0")
    ref_env = _get_vllm_cache_env(tmp_path, "ref_0")

    assert actor_env["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm_cache" / "actor_0")
    assert actor_env["TORCHINDUCTOR_CACHE_DIR"].endswith("/actor_0/inductor")
    assert actor_env["TRITON_CACHE_DIR"].endswith("/actor_0/triton")
    assert set(actor_env.values()).isdisjoint(ref_env.values())
