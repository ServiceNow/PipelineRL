import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import transformers
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from pipelinerl.finetune.checkpoints import (
    get_auto_model_class,
    get_model_loader,
)
from pipelinerl.launch import _get_vllm_cache_env
from pipelinerl.prerun_evidence import (
    QWEN35_9B_MODEL_DESCRIPTOR,
    QWEN35_9B_MODEL_ID,
    QWEN35_9B_MODEL_REVISION,
    QWEN35_27B_MODEL_DESCRIPTOR,
    QWEN35_CHAT_TEMPLATE_SHA256,
    QWEN35_TOKENIZER_CONFIG_SHA256,
    QWEN35_TOOL_CALL_PARSER,
    TEXT_MODEL_DESCRIPTORS,
    assert_no_vision_parameters,
    hash_model_snapshot,
    inspect_text_model_snapshot,
)


def _loader_args(descriptor):
    return SimpleNamespace(
        text_only_composite_model=True,
        model_revision=descriptor.revision,
        config_name=descriptor.model_id,
        trust_remote_code=False,
    )


def _mock_composite_config(descriptor):
    text_config_cls = getattr(
        transformers,
        descriptor.text_config_class_name,
    )
    return SimpleNamespace(
        model_type=descriptor.composite_model_type,
        text_config=text_config_cls(),
    )


def _snapshot(
    tmp_path: Path,
    base_descriptor=QWEN35_9B_MODEL_DESCRIPTOR,
):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    config = {
        "model_type": "qwen3_5",
        "tie_word_embeddings": False,
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 2,
            "hidden_size": 8,
            "intermediate_size": 16,
            "max_position_embeddings": 64,
            "vocab_size": 32,
        },
    }
    (snapshot / "config.json").write_text(json.dumps(config))
    shard = "model-00001-of-00001.safetensors"
    weight_map = {
        "model.language_model.embed_tokens.weight": shard,
        "model.language_model.layers.0.self_attn.q_proj.weight": shard,
        "model.language_model.layers.1.self_attn.q_proj.weight": shard,
        "model.language_model.norm.weight": shard,
        "model.visual.blocks.0.weight": shard,
        "mtp.layers.0.weight": shard,
        "lm_head.weight": shard,
    }
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    (snapshot / shard).write_bytes(b"qwen-text-weights")
    (snapshot / "tokenizer.json").write_text('{"version":"test"}')
    (snapshot / "tokenizer_config.json").write_text('{"chat_template":"test"}')
    (snapshot / "chat_template.jinja").write_text("test template")
    identity = hash_model_snapshot(
        snapshot,
        base_descriptor.model_id,
        base_descriptor.revision,
    )
    descriptor = replace(
        base_descriptor,
        num_hidden_layers=2,
        hidden_size=8,
        intermediate_size=16,
        max_position_embeddings=64,
        vocab_size=32,
        artifact_sha256=tuple(
            (artifact.path, artifact.sha256)
            for artifact in identity.artifacts
        ),
    )
    return snapshot, descriptor, identity


def test_composite_text_model_classes_are_registered():
    qwen_composite_config_type = type(transformers.Qwen3_5Config())
    qwen_text_config_type = type(transformers.Qwen3_5TextConfig())

    assert (
        qwen_composite_config_type
        in AutoModelForImageTextToText._model_mapping
    )
    assert qwen_text_config_type in AutoModelForCausalLM._model_mapping
    assert (
        get_auto_model_class("vision2seq-language-modeling")
        is AutoModelForImageTextToText
    )


def test_text_loader_uses_exact_descriptor_class_config_and_remap(
    monkeypatch,
):
    descriptor = QWEN35_9B_MODEL_DESCRIPTOR
    calls = []
    composite_config = _mock_composite_config(descriptor)

    def from_pretrained(config_name, **kwargs):
        calls.append((config_name, kwargs))
        return composite_config

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        from_pretrained,
    )
    model_cls, loading_args = get_model_loader(
        _loader_args(descriptor),
        "causal-language-modeling",
    )

    assert model_cls is getattr(transformers, descriptor.model_class_name)
    assert loading_args["config"] is composite_config.text_config
    assert loading_args["key_mapping"] == {r"^model\.language_model\.": "model."}
    assert calls == [
        (
            descriptor.model_id,
            {
                "trust_remote_code": False,
                "revision": descriptor.revision,
            },
        )
    ]


def test_qwen_27b_is_inspectable_but_rejected_as_run1_policy(
    tmp_path,
    monkeypatch,
):
    snapshot, descriptor, identity = _snapshot(
        tmp_path,
        QWEN35_27B_MODEL_DESCRIPTOR,
    )

    topology = inspect_text_model_snapshot(snapshot, descriptor, identity)
    assert topology.layer_indices == [0, 1]
    assert descriptor.policy_eligible is False

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("ineligible policy was inspected"),
    )
    with pytest.raises(ValueError, match="not eligible for the run-1 policy"):
        get_model_loader(
            _loader_args(QWEN35_27B_MODEL_DESCRIPTOR),
            "causal-language-modeling",
        )


def test_removed_gemma_selector_fails_before_model_inspection():
    args = _loader_args(QWEN35_9B_MODEL_DESCRIPTOR)
    args.text_only_gemma4 = True

    with pytest.raises(ValueError, match="text_only_gemma4 was removed"):
        get_model_loader(args, "causal-language-modeling")


@pytest.mark.parametrize(
    ("model_id", "revision"),
    (
        ("Qwen/Qwen3.5-9B-Base", "68c46c4b3498877f3ef123c856ecfde50c39f404"),
        (QWEN35_9B_MODEL_ID, "moving-revision"),
    ),
)
def test_text_loader_rejects_unreviewed_identity(model_id, revision):
    args = SimpleNamespace(
        text_only_composite_model=True,
        model_revision=revision,
        config_name=model_id,
        trust_remote_code=False,
    )

    with pytest.raises(ValueError, match="No reviewed text-model descriptor"):
        get_model_loader(args, "causal-language-modeling")


def test_qwen_descriptor_registry_pins_both_verified_artifact_sets():
    assert set(TEXT_MODEL_DESCRIPTORS) == {
        (QWEN35_9B_MODEL_ID, QWEN35_9B_MODEL_REVISION),
        (
            QWEN35_27B_MODEL_DESCRIPTOR.model_id,
            QWEN35_27B_MODEL_DESCRIPTOR.revision,
        ),
    }
    assert len(QWEN35_9B_MODEL_DESCRIPTOR.artifact_sha256) == 16
    assert len(QWEN35_27B_MODEL_DESCRIPTOR.artifact_sha256) == 23
    assert QWEN35_TOOL_CALL_PARSER == "qwen3_xml"
    for descriptor in (
        QWEN35_9B_MODEL_DESCRIPTOR,
        QWEN35_27B_MODEL_DESCRIPTOR,
    ):
        paths = dict(descriptor.artifact_sha256)
        assert descriptor.exact_artifact_set is True
        assert {
            ".gitattributes",
            "README.md",
            "chat_template.jinja",
            "config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        } <= set(paths)
        assert paths["tokenizer_config.json"] == QWEN35_TOKENIZER_CONFIG_SHA256
        assert paths["chat_template.jinja"] == QWEN35_CHAT_TEMPLATE_SHA256
    qwen9_paths = dict(QWEN35_9B_MODEL_DESCRIPTOR.artifact_sha256)
    qwen27_paths = dict(QWEN35_27B_MODEL_DESCRIPTOR.artifact_sha256)
    assert "LICENSE" in qwen9_paths
    assert "generation_config.json" not in qwen9_paths
    assert "LICENSE" not in qwen27_paths
    assert "generation_config.json" in qwen27_paths
    assert (
        dict(QWEN35_9B_MODEL_DESCRIPTOR.artifact_sha256)["tokenizer.json"]
        == dict(QWEN35_27B_MODEL_DESCRIPTOR.artifact_sha256)["tokenizer.json"]
    )


def test_snapshot_identity_excludes_huggingface_download_cache(tmp_path):
    snapshot, descriptor, identity = _snapshot(tmp_path)
    cache = snapshot / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "config.json.lock").write_text("")
    (cache / "config.json.metadata").write_text("mutable")

    rehashed = hash_model_snapshot(
        snapshot,
        descriptor.model_id,
        descriptor.revision,
    )

    assert rehashed == identity


def test_exact_qwen_snapshot_rejects_unreviewed_artifact(tmp_path):
    snapshot, descriptor, _ = _snapshot(tmp_path)
    (snapshot / "generation_config.json").write_text("{}")
    identity = hash_model_snapshot(
        snapshot,
        descriptor.model_id,
        descriptor.revision,
    )

    with pytest.raises(ValueError, match="unreviewed artifacts"):
        inspect_text_model_snapshot(snapshot, descriptor, identity)


def test_model_identity_rejects_duplicate_artifact_paths(tmp_path):
    snapshot, descriptor, identity = _snapshot(tmp_path)
    duplicate = identity.model_copy(
        update={"artifacts": [*identity.artifacts, identity.artifacts[0]]}
    )

    with pytest.raises(ValueError, match="duplicate paths"):
        inspect_text_model_snapshot(snapshot, descriptor, duplicate)


def test_exact_qwen_snapshot_rejects_missing_artifact(tmp_path):
    snapshot, descriptor, _ = _snapshot(tmp_path)
    (snapshot / "chat_template.jinja").unlink()
    identity = hash_model_snapshot(
        snapshot,
        descriptor.model_id,
        descriptor.revision,
    )

    with pytest.raises(ValueError, match="artifact SHA256"):
        inspect_text_model_snapshot(snapshot, descriptor, identity)


def test_qwen_snapshot_partitions_vision_and_mtp_outside_transfer(
    tmp_path,
):
    snapshot, descriptor, identity = _snapshot(tmp_path)

    topology = inspect_text_model_snapshot(
        snapshot,
        descriptor,
        identity,
    )

    assert topology.layer_indices == [0, 1]
    assert topology.vision_tensor_count == 1
    assert topology.nontransferred_tensor_count == 1
    assert topology.transfer_categories == [
        "backbone",
        "embedding",
        "output_head",
    ]


@pytest.mark.parametrize(
    "artifact",
    (
        "model-00001-of-00001.safetensors",
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ),
)
def test_qwen_snapshot_rejects_tampered_artifact(tmp_path, artifact):
    snapshot, descriptor, _ = _snapshot(tmp_path)
    (snapshot / artifact).write_bytes(b"wrong")
    identity = hash_model_snapshot(
        snapshot,
        descriptor.model_id,
        descriptor.revision,
    )

    with pytest.raises(ValueError, match="artifact SHA256"):
        inspect_text_model_snapshot(snapshot, descriptor, identity)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_layer", "layer coverage mismatch"),
        ("extra_layer", "layer coverage mismatch"),
        ("unknown_namespace", "unknown parameter namespaces"),
        ("wrong_topology", "text topology"),
    ),
)
def test_qwen_snapshot_fails_closed_on_structure_drift(
    tmp_path,
    mutation,
    message,
):
    snapshot, descriptor, _ = _snapshot(tmp_path)
    if mutation == "wrong_topology":
        config_path = snapshot / "config.json"
        config = json.loads(config_path.read_text())
        config["text_config"]["hidden_size"] = 9
        config_path.write_text(json.dumps(config))
    else:
        index_path = snapshot / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        if mutation == "missing_layer":
            del index["weight_map"][
                "model.language_model.layers.1.self_attn.q_proj.weight"
            ]
        elif mutation == "extra_layer":
            index["weight_map"][
                "model.language_model.layers.2.self_attn.q_proj.weight"
            ] = "model-00001-of-00001.safetensors"
        else:
            index["weight_map"]["unexpected.weight"] = (
                "model-00001-of-00001.safetensors"
            )
        index_path.write_text(json.dumps(index))
    identity = hash_model_snapshot(
        snapshot,
        descriptor.model_id,
        descriptor.revision,
    )
    descriptor = replace(
        descriptor,
        artifact_sha256=tuple(
            (artifact.path, artifact.sha256)
            for artifact in identity.artifacts
        ),
    )

    with pytest.raises(ValueError, match=message):
        inspect_text_model_snapshot(snapshot, descriptor, identity)


def test_qwen_visual_parameters_fail_closed_on_transfer_surface():
    with pytest.raises(ValueError, match="vision parameters"):
        assert_no_vision_parameters(
            ["model.visual.blocks.0.attn.qkv.weight"],
            surface="trainer transfer",
        )


def test_vllm_compile_cache_is_isolated_per_engine(tmp_path: Path):
    actor_env = _get_vllm_cache_env(tmp_path, "actor_0")
    ref_env = _get_vllm_cache_env(tmp_path, "ref_0")

    assert actor_env["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm_cache" / "actor_0")
    assert actor_env["TORCHINDUCTOR_CACHE_DIR"].endswith("/actor_0/inductor")
    assert actor_env["TRITON_CACHE_DIR"].endswith("/actor_0/triton")
    assert set(actor_env.values()).isdisjoint(ref_env.values())
