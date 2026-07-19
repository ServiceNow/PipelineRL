"""Domain-neutral model-transfer and pre-run evidence primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel


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
PARITY_MAX_ABS_TOLERANCE = 0.05
PARITY_TOLERANCE_BASIS = (
    "BF16 unit roundoff is 2^-7=0.0078125; 0.05 nats is 6.4 BF16 epsilons "
    "for TP2 reduction ordering and Transformers/vLLM fused-kernel differences "
    "with FP32 output-head logits on both sides."
)
REQUIRED_TRANSFER_CATEGORIES = frozenset(
    {"expert", "router", "embedding", "output_head"}
)
_FORBIDDEN_VISION_PARTS = ("vision_tower", "embed_vision")
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


def gemma_revision_is_verified() -> bool:
    return GEMMA_MODEL_REVISION_VERIFIED


def require_verified_gemma_revision() -> None:
    if not gemma_revision_is_verified():
        raise ValueError(
            "Gemma model revision is an unverified placeholder; verify the "
            "immutable Hugging Face revision before an evidence run"
        )


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
