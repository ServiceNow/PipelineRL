from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Gemma4Config,
    Gemma4ForCausalLM,
)

import pipelinerl.prerun_evidence as prerun_evidence
from pipelinerl.prerun_evidence import (
    GEMMA_MODEL_ID,
    GEMMA_MODEL_REVISION,
)
from pipelinerl.finetune.checkpoints import (
    get_auto_model_class,
    get_model_loader,
)
from pipelinerl.launch import _get_vllm_cache_env


def test_gemma4_is_registered_for_pipelinerl_model_classes():
    config_type = type(Gemma4Config())

    assert config_type in AutoModelForCausalLM._model_mapping
    assert config_type in AutoModelForImageTextToText._model_mapping
    assert get_auto_model_class("vision2seq-language-modeling") is AutoModelForImageTextToText


def test_gemma4_text_loader_pins_revision_and_excludes_vision(
    monkeypatch,
):
    assert prerun_evidence.GEMMA_MODEL_REVISION_VERIFIED is True
    calls = []
    text_config = SimpleNamespace(model_type="gemma4_text")
    composite_config = SimpleNamespace(
        model_type="gemma4",
        text_config=text_config,
    )

    def from_pretrained(config_name, **kwargs):
        calls.append((config_name, kwargs))
        return composite_config

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        from_pretrained,
    )
    args = SimpleNamespace(
        text_only_gemma4=True,
        model_revision=GEMMA_MODEL_REVISION,
        config_name=GEMMA_MODEL_ID,
        trust_remote_code=False,
    )

    model_cls, loading_args = get_model_loader(
        args,
        "causal-language-modeling",
    )

    assert model_cls is Gemma4ForCausalLM
    assert loading_args["config"] is text_config
    assert loading_args["key_mapping"] == {
        r"^model\.language_model\.": "model."
    }
    assert calls == [
        (
            GEMMA_MODEL_ID,
            {
                "trust_remote_code": False,
                "revision": GEMMA_MODEL_REVISION,
            },
        )
    ]


def test_gemma4_text_loader_rejects_unreviewed_revision():
    args = SimpleNamespace(
        text_only_gemma4=True,
        model_revision="moving-revision",
        config_name=GEMMA_MODEL_ID,
        trust_remote_code=False,
    )

    with pytest.raises(ValueError, match="reviewed Gemma model ID and revision"):
        get_model_loader(args, "causal-language-modeling")


def test_gemma4_text_loader_rejects_unverified_placeholder(monkeypatch):
    monkeypatch.setattr(
        prerun_evidence,
        "GEMMA_MODEL_REVISION_VERIFIED",
        False,
    )
    args = SimpleNamespace(
        text_only_gemma4=True,
        model_revision=GEMMA_MODEL_REVISION,
        config_name=GEMMA_MODEL_ID,
        trust_remote_code=False,
    )

    with pytest.raises(ValueError, match="unverified placeholder"):
        get_model_loader(args, "causal-language-modeling")


def test_vllm_compile_cache_is_isolated_per_engine(tmp_path: Path):
    actor_env = _get_vllm_cache_env(tmp_path, "actor_0")
    ref_env = _get_vllm_cache_env(tmp_path, "ref_0")

    assert actor_env["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm_cache" / "actor_0")
    assert actor_env["TORCHINDUCTOR_CACHE_DIR"].endswith("/actor_0/inductor")
    assert actor_env["TRITON_CACHE_DIR"].endswith("/actor_0/triton")
    assert set(actor_env.values()).isdisjoint(ref_env.values())
