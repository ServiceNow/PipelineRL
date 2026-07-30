"""Domain-neutral model-transfer and pre-run evidence primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import torch
from pydantic import BaseModel


@dataclass(frozen=True)
class TextModelDescriptor:
    model_id: str
    revision: str
    composite_model_type: str
    text_model_type: str
    model_class_name: str
    text_config_class_name: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    tie_word_embeddings: bool
    key_mapping: tuple[tuple[str, str], ...]
    artifact_sha256: tuple[tuple[str, str], ...]
    nontransferred_prefixes: tuple[str, ...]
    revision_verified: bool
    # Deployment-role decision for run 1, not a technical trainability limit.
    policy_eligible: bool
    provenance: str


GEMMA_MODEL_ID = "google/gemma-4-26B-A4B-it"
# VERIFIED 2026-07-19 by Claude with Rafa in the loop: Hugging Face main resolves
# to this immutable revision, and the local snapshot matches its repository tree.
GEMMA_MODEL_REVISION = "01e5b3ee840d3a9e0b0b493c593e85398a30ef75"
GEMMA_MODEL_REVISION_VERIFIED = True
GEMMA_MODEL_REVISION_PROVENANCE = (
    "Verified 2026-07-19 by Claude with Rafa in the loop: the Hugging Face "
    "refs API reports google/gemma-4-26B-A4B-it main == "
    "01e5b3ee840d3a9e0b0b493c593e85398a30ef75."
)
GEMMA_POLICY_IDENTITY = f"{GEMMA_MODEL_ID}@{GEMMA_MODEL_REVISION}"
GEMMA_EXPECTED_TEXT_TOPOLOGY = (
    30,
    2816,
    128,
    8,
    704,
    262144,
    262144,
)
GEMMA_TOPOLOGY_PROVENANCE = (
    "Verified 2026-07-19 by Claude at revision "
    "01e5b3ee840d3a9e0b0b493c593e85398a30ef75: local snapshot "
    "/mnt/llmd/base_models/gemma-4-26B-A4B-it matched Hugging Face tree LFS "
    "SHA256 OIDs for model-00001-of-00002.safetensors="
    "1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3, "
    "model-00002-of-00002.safetensors="
    "aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7, "
    "and tokenizer.json="
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f; "
    "non-LFS files and config topology also matched."
)

_TEXT_KEY_MAPPING = ((r"^model\.language_model\.", "model."),)
_QWEN_TOKENIZER_SHA256 = (
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
)
QWEN35_TOKENIZER_CONFIG_SHA256 = (
    "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
)
QWEN35_CHAT_TEMPLATE_SHA256 = (
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
)
QWEN35_TOOL_CALL_PARSER = "qwen3_xml"

GEMMA_MODEL_DESCRIPTOR = TextModelDescriptor(
    model_id=GEMMA_MODEL_ID,
    revision=GEMMA_MODEL_REVISION,
    composite_model_type="gemma4",
    text_model_type="gemma4_text",
    model_class_name="Gemma4ForCausalLM",
    text_config_class_name="Gemma4TextConfig",
    num_hidden_layers=30,
    hidden_size=2816,
    intermediate_size=2112,
    num_experts=128,
    top_k_experts=8,
    moe_intermediate_size=704,
    max_position_embeddings=262144,
    vocab_size=262144,
    tie_word_embeddings=True,
    key_mapping=_TEXT_KEY_MAPPING,
    artifact_sha256=(
        (
            "model-00001-of-00002.safetensors",
            "1127684971bbca40465435a5cad69d67ad603bf5e61c6dfd5561fae4a3bcfdb3",
        ),
        (
            "model-00002-of-00002.safetensors",
            "aab47033e1e8a492ef8e581efae1cf36478d0433567e7729b3c1728bc8970db7",
        ),
        (
            "tokenizer.json",
            "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        ),
    ),
    nontransferred_prefixes=(),
    revision_verified=GEMMA_MODEL_REVISION_VERIFIED,
    policy_eligible=True,
    provenance=GEMMA_TOPOLOGY_PROVENANCE,
)

QWEN35_9B_MODEL_ID = "Qwen/Qwen3.5-9B"
QWEN35_9B_MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
QWEN35_9B_POLICY_IDENTITY = f"{QWEN35_9B_MODEL_ID}@{QWEN35_9B_MODEL_REVISION}"
QWEN35_9B_MODEL_DESCRIPTOR = TextModelDescriptor(
    model_id=QWEN35_9B_MODEL_ID,
    revision=QWEN35_9B_MODEL_REVISION,
    composite_model_type="qwen3_5",
    text_model_type="qwen3_5_text",
    model_class_name="Qwen3_5ForCausalLM",
    text_config_class_name="Qwen3_5TextConfig",
    num_hidden_layers=32,
    hidden_size=4096,
    intermediate_size=12288,
    num_experts=0,
    top_k_experts=0,
    moe_intermediate_size=0,
    max_position_embeddings=262144,
    vocab_size=248320,
    tie_word_embeddings=False,
    key_mapping=_TEXT_KEY_MAPPING,
    artifact_sha256=(
        (
            "model.safetensors-00001-of-00004.safetensors",
            "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
        ),
        (
            "model.safetensors-00002-of-00004.safetensors",
            "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
        ),
        (
            "model.safetensors-00003-of-00004.safetensors",
            "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
        ),
        (
            "model.safetensors-00004-of-00004.safetensors",
            "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
        ),
        ("tokenizer.json", _QWEN_TOKENIZER_SHA256),
        ("tokenizer_config.json", QWEN35_TOKENIZER_CONFIG_SHA256),
        ("chat_template.jinja", QWEN35_CHAT_TEMPLATE_SHA256),
    ),
    nontransferred_prefixes=("mtp.",),
    revision_verified=True,
    policy_eligible=True,
    provenance=(
        "Verified 2026-07-30: Hugging Face refs resolve Qwen/Qwen3.5-9B "
        "to c202236235762e1c871ad0ccb60c8ee5ba337b9a; local snapshot "
        "/mnt/llmd/base_models/Qwen3.5-9B has every shard "
        "plus tokenizer matches that revision's LFS SHA256 OID. "
        "tokenizer_config.json and chat_template.jinja were byte-matched "
        "to direct downloads from the same immutable revision."
    ),
)

QWEN35_27B_MODEL_ID = "Qwen/Qwen3.5-27B"
QWEN35_27B_MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
QWEN35_27B_MODEL_DESCRIPTOR = TextModelDescriptor(
    model_id=QWEN35_27B_MODEL_ID,
    revision=QWEN35_27B_MODEL_REVISION,
    composite_model_type="qwen3_5",
    text_model_type="qwen3_5_text",
    model_class_name="Qwen3_5ForCausalLM",
    text_config_class_name="Qwen3_5TextConfig",
    num_hidden_layers=64,
    hidden_size=5120,
    intermediate_size=17408,
    num_experts=0,
    top_k_experts=0,
    moe_intermediate_size=0,
    max_position_embeddings=262144,
    vocab_size=248320,
    tie_word_embeddings=False,
    key_mapping=_TEXT_KEY_MAPPING,
    artifact_sha256=(
        (
            "model.safetensors-00001-of-00011.safetensors",
            "9019228d172c87d5603266c2d56672d119e838facffa164de803a1ebf0d716d2",
        ),
        (
            "model.safetensors-00002-of-00011.safetensors",
            "890ef00c920b01c1c02755088c8d8ca5cdd6a1faa2a9a729eecd00857648b411",
        ),
        (
            "model.safetensors-00003-of-00011.safetensors",
            "8aca03689ad0717fb91455809a6670eecca818bea8c0b6bd3a591a8807cc3223",
        ),
        (
            "model.safetensors-00004-of-00011.safetensors",
            "20f539430c60fa611b3522b8408749a080b63f1fe034e551a6539ef864ca079a",
        ),
        (
            "model.safetensors-00005-of-00011.safetensors",
            "57a0c074c654f05fc2d6b5112eeec387ac04986e928e42f502993e69aa03a49d",
        ),
        (
            "model.safetensors-00006-of-00011.safetensors",
            "cfa4e6fbfc600854ef6c8e5465d0a1e43832a64b5e539f1e880333e3b1086703",
        ),
        (
            "model.safetensors-00007-of-00011.safetensors",
            "d426963325b2319cb9e1442bc8b89b9a7748a8f7a907aa4041c11531cda6b014",
        ),
        (
            "model.safetensors-00008-of-00011.safetensors",
            "fe60dbb9d25354c4eb2a9a59d9b1afd07741c8bebab5870f5eea6eadb4cf9a06",
        ),
        (
            "model.safetensors-00009-of-00011.safetensors",
            "71a153d882242734a1fc7e000734727f00f6ec8ca70b044f5e44fcf97736ef8f",
        ),
        (
            "model.safetensors-00010-of-00011.safetensors",
            "146745698b9f21940e2982beb1816a0eef3b80d77ab9cd884695b9a08c2697eb",
        ),
        (
            "model.safetensors-00011-of-00011.safetensors",
            "d947ce7483c4109b55039f1359f4494d22390cf123568100abd89816802f097d",
        ),
        ("tokenizer.json", _QWEN_TOKENIZER_SHA256),
        ("tokenizer_config.json", QWEN35_TOKENIZER_CONFIG_SHA256),
        ("chat_template.jinja", QWEN35_CHAT_TEMPLATE_SHA256),
    ),
    nontransferred_prefixes=("mtp.",),
    revision_verified=True,
    policy_eligible=False,
    provenance=(
        "Verified 2026-07-30: Hugging Face refs resolve Qwen/Qwen3.5-27B "
        "to fc05daec18b0a78c049392ed2e771dde82bdf654; local snapshot "
        "/mnt/llmd/base_models/Qwen3.5-27B has every shard "
        "plus tokenizer matches that revision's LFS SHA256 OID. "
        "tokenizer_config.json and chat_template.jinja were byte-matched "
        "to direct downloads from the same immutable revision."
    ),
)

TEXT_MODEL_DESCRIPTORS = MappingProxyType(
    {
        (descriptor.model_id, descriptor.revision): descriptor
        for descriptor in (
            GEMMA_MODEL_DESCRIPTOR,
            QWEN35_9B_MODEL_DESCRIPTOR,
            QWEN35_27B_MODEL_DESCRIPTOR,
        )
    }
)
PARITY_MAX_ABS_TOLERANCE = 0.05
PARITY_TOLERANCE_BASIS = (
    "BF16 unit roundoff is 2^-7=0.0078125; 0.05 nats is 6.4 BF16 epsilons "
    "for TP2 reduction ordering and Transformers/vLLM fused-kernel differences "
    "with FP32 output-head logits on both sides."
)
REQUIRED_TRANSFER_CATEGORIES = frozenset(
    {"expert", "router", "embedding", "output_head"}
)
_FORBIDDEN_VISION_PARTS = (
    "vision_tower",
    "embed_vision",
    "model.visual.",
)
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_FINGERPRINT_CHUNK_SIZE = 1 << 20


class ArtifactDigest(BaseModel):
    path: str
    size: int
    sha256: str


class ModelArtifactIdentity(BaseModel):
    model_id: str
    revision: str
    snapshot_digest: str
    artifacts: list[ArtifactDigest]


class TextModelTopology(BaseModel):
    model_type: str
    text_model_type: str
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    tie_word_embeddings: bool
    layer_indices: list[int]
    text_tensor_count: int
    vision_tensor_count: int
    nontransferred_tensor_count: int
    transfer_categories: list[str]


class GemmaTopology(BaseModel):
    model_type: str
    text_model_type: str
    num_hidden_layers: int
    hidden_size: int
    num_experts: int
    top_k_experts: int
    moe_intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    tie_word_embeddings: bool
    text_tensor_count: int
    vision_tensor_count: int
    transfer_categories: list[str]


class TensorFingerprint(BaseModel):
    shape: list[int]
    dtype: str
    numel: int
    finite_count: int
    value_sum: float
    squared_sum: float

    def matches(self, other: "TensorFingerprint") -> bool:
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and self.numel == other.numel
            and self.finite_count == other.finite_count
            and math.isclose(
                self.value_sum, other.value_sum, rel_tol=1e-9, abs_tol=1e-6
            )
            and math.isclose(
                self.squared_sum, other.squared_sum, rel_tol=1e-9, abs_tol=1e-6
            )
        )


class ParameterTransferReceipt(BaseModel):
    source_name: str
    loaded_names: list[str]
    categories: list[str]
    received_fingerprint: TensorFingerprint


class WorkerTransferReceipt(BaseModel):
    rank: int
    receipts: list[ParameterTransferReceipt]


class EndpointTransferReceipt(BaseModel):
    endpoint: str
    workers: list[WorkerTransferReceipt]


class TransferEvidence(BaseModel):
    phase: Literal["initial", "after_optimizer"]
    version: int
    source_fingerprints: dict[str, TensorFingerprint]
    endpoints: list[EndpointTransferReceipt]


class ParityEvidence(BaseModel):
    phase: Literal["before_update", "after_update"]
    version: int
    endpoint: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    trainer_logprobs: list[float]
    vllm_logprobs: list[float]
    max_abs_error: float
    rms_error: float
    tolerance: float = PARITY_MAX_ABS_TOLERANCE
    tolerance_basis: str = PARITY_TOLERANCE_BASIS
    passed: bool


def get_text_model_descriptor(
    model_id: str,
    revision: str | None,
) -> TextModelDescriptor:
    try:
        return TEXT_MODEL_DESCRIPTORS[(model_id, revision)]
    except KeyError as exc:
        raise ValueError(
            f"No reviewed text-model descriptor for {model_id}@{revision}"
        ) from exc


def gemma_revision_is_verified() -> bool:
    return GEMMA_MODEL_REVISION_VERIFIED


def model_revision_is_verified(descriptor: TextModelDescriptor) -> bool:
    if descriptor is GEMMA_MODEL_DESCRIPTOR:
        return gemma_revision_is_verified()
    return descriptor.revision_verified


def require_verified_model_revision(
    descriptor: TextModelDescriptor,
) -> None:
    if not model_revision_is_verified(descriptor):
        raise ValueError(
            f"{descriptor.model_id} model revision is an unverified "
            "placeholder; verify the immutable Hugging Face revision "
            "before an evidence run"
        )


def require_verified_gemma_revision() -> None:
    require_verified_model_revision(GEMMA_MODEL_DESCRIPTOR)


def prerun_enabled(cfg: Any) -> bool:
    prerun = getattr(cfg, "tau2_prerun", None)
    return bool(prerun is not None and getattr(prerun, "enabled", False))


def evidence_directory(cfg: Any) -> Path:
    prerun = getattr(cfg, "tau2_prerun", None)
    configured = (
        getattr(prerun, "evidence_dir", None) if prerun is not None else None
    )
    return Path(configured) if configured else Path(cfg.output_dir) / "tau2_prerun"


def append_evidence_record(
    directory: Path,
    topic: str,
    record: BaseModel | Mapping[str, Any],
) -> None:
    """Append one compact JSONL record using one O_APPEND write."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = (
        record.model_dump(mode="json")
        if isinstance(record, BaseModel)
        else dict(record)
    )
    line = (
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    fd = os.open(
        directory / f"{topic}.jsonl",
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def read_evidence_records(directory: Path, topic: str) -> list[dict[str, Any]]:
    path = directory / f"{topic}.jsonl"
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_snapshot(
    snapshot: Path,
    model_id: str,
    revision: str,
) -> ModelArtifactIdentity:
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Model snapshot {snapshot} contains no artifacts")
    artifacts = [
        ArtifactDigest(
            path=path.relative_to(snapshot).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    ]
    aggregate = hashlib.sha256()
    for artifact in artifacts:
        aggregate.update(
            f"{artifact.path}\0{artifact.size}\0{artifact.sha256}\n".encode()
        )
    return ModelArtifactIdentity(
        model_id=model_id,
        revision=revision,
        snapshot_digest=aggregate.hexdigest(),
        artifacts=artifacts,
    )


def validate_model_descriptor_artifacts(
    identity: ModelArtifactIdentity,
    descriptor: TextModelDescriptor,
) -> None:
    if (
        identity.model_id != descriptor.model_id
        or identity.revision != descriptor.revision
    ):
        raise ValueError(
            "Model artifact identity does not match the reviewed descriptor"
        )
    expected = dict(descriptor.artifact_sha256)
    if len(expected) != len(descriptor.artifact_sha256):
        raise ValueError("Reviewed descriptor has duplicate artifact paths")
    observed = {artifact.path: artifact.sha256 for artifact in identity.artifacts}
    mismatched = sorted(
        artifact
        for artifact, digest in expected.items()
        if observed.get(artifact) != digest
    )
    if mismatched:
        raise ValueError(
            f"Model snapshot does not match reviewed artifact SHA256: {mismatched}"
        )


def has_vision_parameter(name: str) -> bool:
    return any(part in name for part in _FORBIDDEN_VISION_PARTS)


def assert_no_vision_parameters(
    names: Iterable[str],
    *,
    surface: str,
) -> None:
    forbidden = sorted(name for name in names if has_vision_parameter(name))
    if forbidden:
        raise ValueError(
            f"{surface} contains vision parameters: {forbidden[:5]}"
        )


def parameter_layer_indices(names: Iterable[str]) -> set[int]:
    return {
        int(match.group(1))
        for name in names
        if (match := _LAYER_INDEX_RE.search(name)) is not None
    }


def required_transfer_categories(num_experts: int) -> frozenset[str]:
    categories = {"backbone", "embedding", "output_head"}
    if num_experts > 0:
        categories.update(("expert", "router"))
    return frozenset(categories)


def parameter_categories(
    name: str,
    *,
    tied_output_head: bool = True,
) -> list[str]:
    if has_vision_parameter(name):
        return ["vision"]
    categories = set()
    if ".experts." in name or ".moe." in name:
        categories.add("expert")
    if ".router." in name:
        categories.add("router")
    if "embed_tokens" in name:
        categories.add("embedding")
        if tied_output_head:
            categories.add("output_head")
    if "lm_head" in name:
        categories.add("output_head")
    return sorted(categories)


def inspect_text_model_snapshot(
    snapshot: Path,
    descriptor: TextModelDescriptor,
    identity: ModelArtifactIdentity,
) -> TextModelTopology:
    require_verified_model_revision(descriptor)
    validate_model_descriptor_artifacts(identity, descriptor)
    with (snapshot / "config.json").open() as handle:
        config = json.load(handle)
    text = config.get("text_config")
    if (
        config.get("model_type") != descriptor.composite_model_type
        or not isinstance(text, Mapping)
        or text.get("model_type") != descriptor.text_model_type
    ):
        raise ValueError(
            "Model snapshot composite/text types do not match the reviewed descriptor"
        )

    tie_word_embeddings = text.get("tie_word_embeddings")
    if tie_word_embeddings is None:
        tie_word_embeddings = config.get("tie_word_embeddings", False)
    observed_topology = (
        int(text.get("num_hidden_layers", 0)),
        int(text.get("hidden_size", 0)),
        int(text.get("intermediate_size", 0)),
        int(text.get("num_experts") or 0),
        int(text.get("top_k_experts") or 0),
        int(text.get("moe_intermediate_size") or 0),
        int(text.get("max_position_embeddings", 0)),
        int(text.get("vocab_size", 0)),
        bool(tie_word_embeddings),
    )
    expected_topology = (
        descriptor.num_hidden_layers,
        descriptor.hidden_size,
        descriptor.intermediate_size,
        descriptor.num_experts,
        descriptor.top_k_experts,
        descriptor.moe_intermediate_size,
        descriptor.max_position_embeddings,
        descriptor.vocab_size,
        descriptor.tie_word_embeddings,
    )
    if observed_topology != expected_topology:
        raise ValueError(
            "Model snapshot text topology does not match the reviewed "
            f"descriptor: {observed_topology}"
        )

    with (snapshot / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle).get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Model snapshot has no safetensors weight_map")
    expected_shards = {
        artifact
        for artifact, _ in descriptor.artifact_sha256
        if artifact.endswith(".safetensors")
    }
    observed_shards = {str(shard) for shard in weight_map.values()}
    if observed_shards != expected_shards:
        raise ValueError("Model snapshot shard map does not match reviewed artifacts")

    text_names = []
    vision_names = []
    nontransferred_names = []
    unknown_names = []
    for raw_name in weight_map:
        name = str(raw_name)
        if name.startswith("model.language_model.") or name == "lm_head.weight":
            text_names.append(name)
        elif has_vision_parameter(name):
            vision_names.append(name)
        elif any(
            name.startswith(prefix) for prefix in descriptor.nontransferred_prefixes
        ):
            nontransferred_names.append(name)
        else:
            unknown_names.append(name)
    if unknown_names:
        raise ValueError(
            "Model snapshot has unknown parameter namespaces: "
            f"{sorted(unknown_names)[:5]}"
        )

    layer_indices = parameter_layer_indices(text_names)
    expected_layers = set(range(descriptor.num_hidden_layers))
    if layer_indices != expected_layers:
        raise ValueError(
            "Model snapshot layer coverage mismatch: "
            f"observed={sorted(layer_indices)}, "
            f"expected={sorted(expected_layers)}"
        )
    categories = {
        category
        for name in text_names
        for category in parameter_categories(
            name,
            tied_output_head=descriptor.tie_word_embeddings,
        )
    }
    if layer_indices:
        categories.add("backbone")
    missing = required_transfer_categories(descriptor.num_experts) - categories
    if missing:
        raise ValueError(
            f"Model checkpoint is missing transfer categories: {sorted(missing)}"
        )
    return TextModelTopology(
        model_type=descriptor.composite_model_type,
        text_model_type=descriptor.text_model_type,
        num_hidden_layers=descriptor.num_hidden_layers,
        hidden_size=descriptor.hidden_size,
        intermediate_size=descriptor.intermediate_size,
        num_experts=descriptor.num_experts,
        top_k_experts=descriptor.top_k_experts,
        moe_intermediate_size=descriptor.moe_intermediate_size,
        max_position_embeddings=descriptor.max_position_embeddings,
        vocab_size=descriptor.vocab_size,
        tie_word_embeddings=descriptor.tie_word_embeddings,
        layer_indices=sorted(layer_indices),
        text_tensor_count=len(text_names),
        vision_tensor_count=len(vision_names),
        nontransferred_tensor_count=len(nontransferred_names),
        transfer_categories=sorted(categories),
    )


def inspect_gemma_snapshot(snapshot: Path) -> GemmaTopology:
    with (snapshot / "config.json").open() as handle:
        config = json.load(handle)
    text = config.get("text_config")
    if config.get("model_type") != "gemma4" or not isinstance(text, Mapping):
        raise ValueError(
            "Expected a Gemma4 composite config with text_config"
        )

    with (snapshot / "model.safetensors.index.json").open() as handle:
        weight_map = json.load(handle).get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("Gemma snapshot has no safetensors weight_map")
    names = [str(name) for name in weight_map]
    text_names = [
        name for name in names if name.startswith("model.language_model.")
    ]
    categories = sorted(
        {
            category
            for name in text_names
            for category in parameter_categories(
                name,
                tied_output_head=bool(text.get("tie_word_embeddings")),
            )
        }
        - {"backbone"}
    )
    missing = sorted(REQUIRED_TRANSFER_CATEGORIES - set(categories))
    if missing:
        raise ValueError(
            f"Gemma checkpoint is missing transfer categories: {missing}"
        )

    topology = GemmaTopology(
        model_type=str(config["model_type"]),
        text_model_type=str(text.get("model_type")),
        num_hidden_layers=int(text.get("num_hidden_layers", 0)),
        hidden_size=int(text.get("hidden_size", 0)),
        num_experts=int(text.get("num_experts", 0)),
        top_k_experts=int(text.get("top_k_experts", 0)),
        moe_intermediate_size=int(text.get("moe_intermediate_size", 0)),
        max_position_embeddings=int(
            text.get("max_position_embeddings", 0)
        ),
        vocab_size=int(text.get("vocab_size", 0)),
        tie_word_embeddings=bool(text.get("tie_word_embeddings")),
        text_tensor_count=len(text_names),
        vision_tensor_count=sum(
            has_vision_parameter(name) for name in names
        ),
        transfer_categories=categories,
    )
    observed = (
        topology.num_hidden_layers,
        topology.hidden_size,
        topology.num_experts,
        topology.top_k_experts,
        topology.moe_intermediate_size,
        topology.max_position_embeddings,
        topology.vocab_size,
    )
    if (
        observed != GEMMA_EXPECTED_TEXT_TOPOLOGY
        or not topology.tie_word_embeddings
    ):
        raise ValueError(
            "Unexpected Gemma4-26B-A4B text topology: "
            f"{observed}, tied={topology.tie_word_embeddings}"
        )
    return topology


def tensor_fingerprint(tensor: torch.Tensor) -> TensorFingerprint:
    """Fingerprint every value with bounded temporary storage."""
    flat = tensor.detach().reshape(-1)
    device = flat.device
    value_sum = torch.zeros((), dtype=torch.float64, device=device)
    squared_sum = torch.zeros((), dtype=torch.float64, device=device)
    finite_count = torch.zeros((), dtype=torch.int64, device=device)
    for chunk in flat.split(_FINGERPRINT_CHUNK_SIZE):
        values = chunk.to(dtype=torch.float64)
        finite = torch.isfinite(values)
        finite_count += finite.sum()
        values = torch.where(finite, values, 0.0)
        value_sum += values.sum()
        squared_sum += values.square().sum()
    return TensorFingerprint(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        numel=tensor.numel(),
        finite_count=int(finite_count.item()),
        value_sum=float(value_sum.item()),
        squared_sum=float(squared_sum.item()),
    )


def make_parity_evidence(
    *,
    phase: Literal["before_update", "after_update"],
    version: int,
    endpoint: str,
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    trainer_logprobs: Sequence[float],
    vllm_logprobs: Sequence[float],
    tolerance: float = PARITY_MAX_ABS_TOLERANCE,
) -> ParityEvidence:
    trainer_values = [float(value) for value in trainer_logprobs]
    vllm_values = [float(value) for value in vllm_logprobs]
    if (
        not prompt_token_ids
        or not completion_token_ids
        or len(trainer_values) != len(completion_token_ids)
        or len(vllm_values) != len(completion_token_ids)
    ):
        raise ValueError(
            "Parity requires non-empty fixed tokens and one logprob "
            "per completion token"
        )
    if (
        not math.isfinite(tolerance)
        or tolerance <= 0
        or any(
            not math.isfinite(value)
            for value in [*trainer_values, *vllm_values]
        )
    ):
        raise ValueError(
            "Parity requires finite logprobs and a positive tolerance"
        )
    errors = [
        abs(left - right)
        for left, right in zip(trainer_values, vllm_values)
    ]
    max_abs_error = max(errors)
    rms_error = math.sqrt(
        sum(error * error for error in errors) / len(errors)
    )
    return ParityEvidence(
        phase=phase,
        version=version,
        endpoint=endpoint,
        prompt_token_ids=list(prompt_token_ids),
        completion_token_ids=list(completion_token_ids),
        trainer_logprobs=trainer_values,
        vllm_logprobs=vllm_values,
        max_abs_error=max_abs_error,
        rms_error=rms_error,
        tolerance=tolerance,
        passed=max_abs_error <= tolerance,
    )
