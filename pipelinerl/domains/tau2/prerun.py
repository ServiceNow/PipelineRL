"""Evidence models and validation for the Tau2/Gemma pre-run gates."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel

from pipelinerl.prerun_evidence import (
    GEMMA_MODEL_ID,
    GEMMA_MODEL_REVISION,
    GEMMA_MODEL_REVISION_PROVENANCE,
    GEMMA_TOPOLOGY_PROVENANCE,
    PARITY_MAX_ABS_TOLERANCE,
    PARITY_TOLERANCE_BASIS,
    ModelArtifactIdentity,
    TextModelTopology,
    ParityEvidence,
    TransferEvidence,
    assert_no_vision_parameters,
    get_text_model_descriptor,
    hash_model_snapshot,
    inspect_text_model_snapshot,
    make_parity_evidence,
    model_revision_is_verified,
    parameter_categories,
    parameter_layer_indices,
    read_evidence_records,
    required_transfer_categories,
)

RUN1_POLICY_LOSS = "gspo"
POLICY_LOSS_FALLBACK = "dppo"
POLICY_LOSS_FALLBACK_TRIGGER = "bringup shows drift-attributable instability"
GSPO_TOKEN_UPGRADE_TRIGGER = "demonstrated need for nonuniform token credit"
USER_SIMULATOR_MODEL_ID = "google/gemma-4-31B-it"
USER_SIMULATOR_REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
USER_SIMULATOR_MODEL = (
    f"{USER_SIMULATOR_MODEL_ID}@{USER_SIMULATOR_REVISION}"
)
USER_SIMULATOR_SNAPSHOT = "/mnt/llmd/base_models/gemma-4-31B-it"
USER_SIMULATOR_ARTIFACT_SHA256 = {
    "model-00001-of-00002.safetensors": (
        "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa"
    ),
    "model-00002-of-00002.safetensors": (
        "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10"
    ),
    "tokenizer.json": (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    ),
}
USER_SIMULATOR_SUBMISSION_MODE = "restartable"
USER_SIMULATOR_PROVENANCE = (
    "Verified 2026-07-27 from local snapshot "
    "/mnt/llmd/base_models/gemma-4-31B-it at Hugging Face revision "
    "842da3794eaa0b77d5f08bae87a17459d91ff475: "
    "model-00001-of-00002.safetensors sha256 "
    "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa; "
    "model-00002-of-00002.safetensors sha256 "
    "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10; "
    "tokenizer.json sha256 "
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f. "
    "The separately managed EAI job owns the revision-bearing served alias."
)
CALIBRATION_DOMAINS = ("airline", "retail", "telecom")
CALIBRATION_TASKS_PER_DOMAIN = 4
CALIBRATION_GROUP_SIZE = 16
CALIBRATION_GROUPS = 12
CALIBRATION_ROLLOUTS = 192
CALIBRATION_CAVEAT = (
    "Derived from N=192 rollouts / 12 G16 groups; typed whole-group drops "
    "remain the runtime safety net."
)
GIB = 1 << 30
ZERO3_GPU_BYTES_PER_PARAMETER = 16
ZERO3_CPU_OFFLOAD_GPU_BYTES_PER_PARAMETER = 4
ZERO3_CPU_OFFLOAD_CPU_BYTES_PER_PARAMETER = 12
ACTIVATION_GIB_PER_1K_TOKENS = 0.5
P99_METHOD = "nearest-rank ceil(0.99 * N)"


class SourcePins(BaseModel):
    nemo_gym_sha: str
    strict_tito_patch_sha256: str
    tau2_runtime_sha: str
    tau2_data_sha: str


PINNED_SOURCES = SourcePins(
    nemo_gym_sha="5f92a73217258074b74b7be26526c69f0ce3075d",
    strict_tito_patch_sha256=(
        "0f712c906579803045d8f62576a7db01adbfb6e26a0cc52390b64f6b1be3a1b2"
    ),
    tau2_runtime_sha="befd120003fb55f48b498f6549556dcaf74582d5",
    tau2_data_sha="ce4013b0afe03c873488878b72851414f92f458b",
)


class CalibrationGroupEvidence(BaseModel):
    group_id: str
    dataset: str
    task_id: str
    attempted_rollouts: int
    published_rollouts: int
    actor_result_bytes: int
    training_envelope_bytes: int | None
    call_prefix_tokens: list[int]
    generation_tokens: list[int]
    merged_sequence_tokens: list[int]
    drop_reason: str | None = None
    endpoint_affinity_failures: int = 0
    version_transport_failures: int = 0
    silent_truncations: int = 0


class DerivedLimit(BaseModel):
    observed_values: list[int]
    observed_max: int
    headroom_factor: float
    rounding_quantum: int
    derived_value: int
    calibration_rollouts: int = CALIBRATION_ROLLOUTS
    calibration_groups: int = CALIBRATION_GROUPS
    caveat: str = CALIBRATION_CAVEAT


class CalibrationSummary(BaseModel):
    groups: int
    attempted_rollouts: int
    published_rollouts: int
    drop_counts: dict[str, int]
    domain_groups: dict[str, int]
    call_prefix_tokens: list[int]
    generation_tokens: list[int]
    merged_sequence_tokens: list[int]
    actor_result_bytes: list[int]
    training_envelope_bytes: list[int]
    derived_limits: dict[str, DerivedLimit]
    sample_size_caveat: str = CALIBRATION_CAVEAT


class TrainerMemoryCandidate(BaseModel):
    name: str
    node_count: int
    gpus_per_node: int
    actor_gpus: int
    trainer_gpus: int
    seq_parallel: int
    gpu_memory_bytes: int
    reserve_bytes_per_gpu: int
    optimizer_cpu_offload: bool = False


class TrainerMemoryBudget(BaseModel):
    candidate: TrainerMemoryCandidate
    measured_p99_merged_tokens: int
    p99_method: str = P99_METHOD
    text_parameter_count: int
    zero3_gpu_bytes_per_parameter: int
    sharded_model_optimizer_bytes_per_gpu: int
    optimizer_cpu_bytes_per_rank: int
    activation_gib_per_1k_tokens_seq_parallel_1: float = (
        ACTIVATION_GIB_PER_1K_TOKENS
    )
    activation_bytes_per_gpu: int
    all_gather_temp_bytes_per_gpu: int
    reserve_bytes_per_gpu: int
    estimated_total_bytes_per_gpu: int
    gpu_memory_bytes: int
    fits: bool
    sample_size_caveat: str = CALIBRATION_CAVEAT


class ServiceIdentity(BaseModel):
    model: str
    endpoint: str


class UserSimulatorDeployment(BaseModel):
    service: ServiceIdentity
    model: ModelArtifactIdentity
    snapshot_path: str
    job_spec_sha256: str
    submission_mode: Literal["restartable"]
    thinking_enabled: bool
    gpu_type: str
    gpu_count: int
    tensor_parallel_size: int
    max_model_len: int
    max_num_seqs: int
    measured_peak_in_flight: int
    observed_request_latencies_s: list[float]
    observed_user_prompt_tokens: list[int]
    prompt_headroom_factor: float
    generation_reserve_tokens: int
    snapshot_hash_bytes: int


def validate_user_simulator_endpoint(endpoint: str) -> str:
    normalized = endpoint.rstrip("/")
    parsed = urlsplit(normalized)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "http"
        or parsed.port != 8000
        or parsed.path != "/v1"
        or parsed.query
        or parsed.fragment
        or not host.startswith("dns-")
        or not host.endswith("-tau2-user")
        or "<" in host
        or ">" in host
    ):
        raise ValueError(
            "Tau2 user simulator endpoint must be the resolved account-scoped "
            "http://dns-<account>-tau2-user:8000/v1 address"
        )
    return normalized


def validate_user_simulator_deployment(
    deployment: UserSimulatorDeployment,
    service: ServiceIdentity,
) -> None:
    if deployment.service != service:
        raise ValueError(
            "Tau2 user simulator deployment service identity mismatch"
        )
    if service.model != USER_SIMULATOR_MODEL:
        raise ValueError("Tau2 user simulator model identity mismatch")
    validate_user_simulator_endpoint(service.endpoint)
    if deployment.model.model_id != USER_SIMULATOR_MODEL_ID:
        raise ValueError("Tau2 user simulator artifact model mismatch")
    if deployment.model.revision != USER_SIMULATOR_REVISION:
        raise ValueError("Tau2 user simulator artifact revision mismatch")
    artifact_sha256 = {
        artifact.path: artifact.sha256
        for artifact in deployment.model.artifacts
    }
    if any(
        artifact_sha256.get(path) != expected
        for path, expected in USER_SIMULATOR_ARTIFACT_SHA256.items()
    ):
        raise ValueError(
            "Tau2 user simulator verified artifact SHA256 mismatch"
        )
    if Path(deployment.snapshot_path) != Path(USER_SIMULATOR_SNAPSHOT):
        raise ValueError("Tau2 user simulator snapshot path mismatch")
    if (
        len(deployment.job_spec_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in deployment.job_spec_sha256
        )
    ):
        raise ValueError("Tau2 user simulator job digest is not SHA256")
    if deployment.submission_mode != USER_SIMULATOR_SUBMISSION_MODE:
        raise ValueError(
            "Tau2 user simulator must use restartable submission mode"
        )
    if deployment.thinking_enabled:
        raise ValueError(
            "Tau2 user simulator must set enable_thinking=false"
        )
    if (
        not deployment.gpu_type
        or deployment.gpu_type.startswith("PENDING_")
        or deployment.gpu_count <= 0
        or deployment.tensor_parallel_size <= 0
        or deployment.gpu_count != deployment.tensor_parallel_size
    ):
        raise ValueError("Tau2 user simulator GPU/TP profile is unresolved")
    if (
        deployment.max_model_len <= 0
        or deployment.max_num_seqs <= 0
        or deployment.measured_peak_in_flight <= 0
        or deployment.measured_peak_in_flight
        > deployment.max_num_seqs
    ):
        raise ValueError(
            "Tau2 user simulator context/concurrency profile is invalid"
        )
    if (
        not deployment.observed_request_latencies_s
        or any(
            latency <= 0
            for latency in deployment.observed_request_latencies_s
        )
        or not deployment.observed_user_prompt_tokens
        or any(
            tokens <= 0
            for tokens in deployment.observed_user_prompt_tokens
        )
        or deployment.prompt_headroom_factor <= 1
        or deployment.generation_reserve_tokens <= 0
    ):
        raise ValueError(
            "Tau2 user simulator measured serving evidence is incomplete"
        )
    required_context = (
        math.ceil(
            max(deployment.observed_user_prompt_tokens)
            * deployment.prompt_headroom_factor
        )
        + deployment.generation_reserve_tokens
    )
    if deployment.max_model_len < required_context:
        raise ValueError(
            "Tau2 user simulator max_model_len is below the measured "
            f"user-prompt headroom requirement {required_context}"
        )
    artifact_bytes = sum(
        artifact.size for artifact in deployment.model.artifacts
    )
    if (
        not deployment.model.artifacts
        or deployment.snapshot_hash_bytes != artifact_bytes
    ):
        raise ValueError(
            "Tau2 user simulator snapshot hash IO accounting mismatch"
        )


class GateResult(BaseModel):
    passed: bool
    detail: str


class PreRunSpec(BaseModel):
    model_id: str = GEMMA_MODEL_ID
    model_revision: str = GEMMA_MODEL_REVISION
    model_snapshot: str
    source_pins: SourcePins
    user_simulator: ServiceIdentity
    user_simulator_deployment: UserSimulatorDeployment
    policy_model: str
    policy_endpoints: list[str]
    expected_tp_size: int
    fixed_prompt_token_ids: list[int]
    fixed_completion_token_ids: list[int]
    user_separation_asserted: bool
    packing_enabled: bool
    mixed_version_loss_gate_passed: bool
    policy_loss: str = RUN1_POLICY_LOSS
    policy_loss_fallback: str = POLICY_LOSS_FALLBACK
    policy_loss_fallback_trigger: str = POLICY_LOSS_FALLBACK_TRIGGER
    gspo_token_upgrade_trigger: str = GSPO_TOKEN_UPGRADE_TRIGGER
    production_memory_candidates: list[TrainerMemoryCandidate]
    job_spec_sha256: str


class PreRunManifest(BaseModel):
    schema_version: int = 1
    job_spec_sha256: str
    model: ModelArtifactIdentity
    topology: TextModelTopology
    source_pins: SourcePins
    user_simulator: ServiceIdentity
    user_simulator_deployment: UserSimulatorDeployment
    policy_model: str
    policy_endpoints: list[str]
    expected_tp_size: int
    fixed_prompt_token_ids: list[int]
    fixed_completion_token_ids: list[int]
    policy_loss: str
    policy_loss_fallback: str
    policy_loss_fallback_trigger: str
    gspo_token_upgrade_trigger: str
    parity_tolerance: float = PARITY_MAX_ABS_TOLERANCE
    parity_tolerance_basis: str = PARITY_TOLERANCE_BASIS
    transfer_evidence: list[TransferEvidence]
    parity_evidence: list[ParityEvidence]
    calibration: CalibrationSummary
    trainer_memory_budgets: list[TrainerMemoryBudget]
    recommended_production_topology: TrainerMemoryCandidate | None
    model_revision_provenance: str = GEMMA_MODEL_REVISION_PROVENANCE
    topology_provenance: str = GEMMA_TOPOLOGY_PROVENANCE
    user_simulator_provenance: str = USER_SIMULATOR_PROVENANCE
    user_simulator_snapshot_hash_io: str = (
        "The finalizer streams the complete dense multimodal simulator "
        "snapshot and records the exact byte count; production trusts the "
        "approved job digest plus this manifest identity."
    )
    alignment_activation: str = (
        "Configured Tau2 adapters emit full-length old/ref arrays; legacy adapters "
        "emit target-only suffix arrays. The serialized field shape is the schema "
        "discriminator and mismatched shapes fail."
    )
    tau2_text_views: str = (
        "n_predicted=0 makes generic prompt_text empty and output_text the complete "
        "merged trajectory; metadata.model_calls token_start/token_end spans are "
        "the canonical assistant view."
    )
    audit_semantics: str = (
        "entered_training means actor publication only; downstream preprocess drops "
        "are reported by preprocessor/atomic_* counters because no "
        "preprocess-to-audit feedback channel exists."
    )
    gates: dict[str, GateResult]
    ready: bool


def select_calibration_problems(
    problems: Sequence[Mapping[str, Any]],
    *,
    tasks_per_domain: int = CALIBRATION_TASKS_PER_DOMAIN,
) -> list[dict[str, Any]]:
    selected = []
    for dataset in CALIBRATION_DOMAINS:
        candidates = [
            problem
            for problem in problems
            if str(problem.get("dataset")) == dataset
        ]
        candidates.sort(
            key=lambda problem: str(
                problem.get("task_id")
                or problem.get("task", {}).get("id")
            )
        )
        if len(candidates) < tasks_per_domain:
            raise ValueError(
                f"Calibration needs {tasks_per_domain} {dataset} tasks, "
                f"found {len(candidates)}"
            )
        selected.extend(
            dict(problem) for problem in candidates[:tasks_per_domain]
        )
    return selected


def _derive_limit(
    values: Sequence[int],
    *,
    headroom: float,
    quantum: int,
) -> DerivedLimit:
    if not values or any(value <= 0 for value in values):
        raise ValueError(
            "Derived limits require positive measured values"
        )
    observed_max = max(values)
    derived = math.ceil(observed_max * headroom / quantum) * quantum
    return DerivedLimit(
        observed_values=list(values),
        observed_max=observed_max,
        headroom_factor=headroom,
        rounding_quantum=quantum,
        derived_value=derived,
    )


def summarize_calibration(
    groups: Sequence[CalibrationGroupEvidence],
) -> CalibrationSummary:
    domain_groups = Counter(group.dataset for group in groups)
    attempted = sum(group.attempted_rollouts for group in groups)
    published = sum(group.published_rollouts for group in groups)
    expected_domains = {
        domain: CALIBRATION_TASKS_PER_DOMAIN
        for domain in CALIBRATION_DOMAINS
    }
    if (
        len(groups) != CALIBRATION_GROUPS
        or attempted != CALIBRATION_ROLLOUTS
    ):
        raise ValueError(
            f"Calibration requires {CALIBRATION_GROUPS} groups/"
            f"{CALIBRATION_ROLLOUTS} rollouts, got "
            f"{len(groups)}/{attempted}"
        )
    if (
        any(
            group.attempted_rollouts != CALIBRATION_GROUP_SIZE
            for group in groups
        )
        or len({group.group_id for group in groups}) != len(groups)
        or len({(group.dataset, group.task_id) for group in groups})
        != len(groups)
    ):
        raise ValueError(
            "Calibration requires distinct tasks and group IDs "
            "with exact G16 attempts"
        )
    for group in groups:
        expected_published = (
            0 if group.drop_reason is not None else CALIBRATION_GROUP_SIZE
        )
        if group.published_rollouts != expected_published:
            raise ValueError(
                f"Calibration group {group.group_id} violates "
                "atomic publication"
            )
        if len(group.call_prefix_tokens) != len(group.generation_tokens):
            raise ValueError(
                f"Calibration group {group.group_id} has "
                "unpaired call measurements"
            )
        if group.drop_reason is None and (
            group.training_envelope_bytes is None
            or len(group.merged_sequence_tokens) != CALIBRATION_GROUP_SIZE
        ):
            raise ValueError(
                f"Calibration group {group.group_id} lacks "
                "complete payload measurements"
            )
    if dict(domain_groups) != expected_domains:
        raise ValueError(
            f"Calibration domain composition is {dict(domain_groups)}, "
            f"expected {expected_domains}"
        )

    call_prefixes = [
        value for group in groups for value in group.call_prefix_tokens
    ]
    generations = [
        value for group in groups for value in group.generation_tokens
    ]
    merged = [
        value for group in groups for value in group.merged_sequence_tokens
    ]
    result_bytes = [group.actor_result_bytes for group in groups]
    envelope_bytes = [
        group.training_envelope_bytes
        for group in groups
        if group.training_envelope_bytes is not None
    ]
    if len(call_prefixes) != len(generations):
        raise ValueError(
            "Calibration call-prefix and generation distributions differ"
        )
    call_totals = [
        prefix + generation
        for prefix, generation in zip(call_prefixes, generations)
    ]
    drop_counts = Counter(
        group.drop_reason
        for group in groups
        if group.drop_reason is not None
    )
    return CalibrationSummary(
        groups=len(groups),
        attempted_rollouts=attempted,
        published_rollouts=published,
        drop_counts=dict(sorted(drop_counts.items())),
        domain_groups=dict(sorted(domain_groups.items())),
        call_prefix_tokens=call_prefixes,
        generation_tokens=generations,
        merged_sequence_tokens=merged,
        actor_result_bytes=result_bytes,
        training_envelope_bytes=envelope_bytes,
        derived_limits={
            "actor_queue_entry_bytes": _derive_limit(
                result_bytes,
                headroom=1.25,
                quantum=1 << 20,
            ),
            "training_envelope_entry_bytes": _derive_limit(
                envelope_bytes,
                headroom=1.25,
                quantum=1 << 20,
            ),
            "model_context_tokens": _derive_limit(
                call_totals,
                headroom=1.10,
                quantum=1024,
            ),
            "training_sequence_tokens": _derive_limit(
                merged,
                headroom=1.10,
                quantum=1024,
            ),
            "generation_margin_tokens": _derive_limit(
                generations,
                headroom=1.25,
                quantum=1024,
            ),
        },
    )


def _nearest_rank_p99(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("P99 requires measured values")
    ordered = sorted(int(value) for value in values)
    return ordered[math.ceil(0.99 * len(ordered)) - 1]


def _dtype_nbytes(dtype: str) -> int:
    sizes = {
        "torch.bfloat16": 2,
        "torch.float16": 2,
        "torch.float32": 4,
    }
    try:
        return sizes[dtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported transfer dtype for memory accounting: {dtype}"
        ) from exc


def build_trainer_memory_budgets(
    calibration: CalibrationSummary,
    transfers: Sequence[TransferEvidence],
    candidates: Sequence[TrainerMemoryCandidate],
) -> tuple[list[TrainerMemoryBudget], TrainerMemoryCandidate | None]:
    after_optimizer = [
        event
        for event in transfers
        if event.phase == "after_optimizer"
    ]
    if len(after_optimizer) != 1:
        raise ValueError(
            "Memory accounting requires exactly one after-optimizer transfer"
        )
    if not candidates:
        raise ValueError(
            "Memory accounting requires production topology candidates"
        )

    fingerprints = after_optimizer[0].source_fingerprints
    if not fingerprints:
        raise ValueError(
            "Memory accounting requires trainer tensor fingerprints"
        )
    parameter_count = sum(
        fingerprint.numel for fingerprint in fingerprints.values()
    )
    all_gather_bytes = max(
        fingerprint.numel * _dtype_nbytes(fingerprint.dtype)
        for fingerprint in fingerprints.values()
    )
    p99_tokens = _nearest_rank_p99(
        calibration.merged_sequence_tokens
    )

    budgets = []
    for candidate in candidates:
        total_gpus = candidate.node_count * candidate.gpus_per_node
        if (
            candidate.node_count <= 0
            or candidate.gpus_per_node <= 0
            or candidate.actor_gpus < 0
            or candidate.trainer_gpus <= 0
            or candidate.actor_gpus + candidate.trainer_gpus
            != total_gpus
        ):
            raise ValueError(
                f"Invalid GPU topology for {candidate.name}"
            )
        if (
            candidate.seq_parallel <= 0
            or candidate.trainer_gpus % candidate.seq_parallel != 0
        ):
            raise ValueError(
                f"Invalid seq_parallel for {candidate.name}"
            )
        if (
            candidate.gpu_memory_bytes <= 0
            or candidate.reserve_bytes_per_gpu < 0
            or candidate.reserve_bytes_per_gpu
            >= candidate.gpu_memory_bytes
        ):
            raise ValueError(
                f"Invalid GPU memory capacity for {candidate.name}"
            )

        gpu_bytes_per_parameter = (
            ZERO3_CPU_OFFLOAD_GPU_BYTES_PER_PARAMETER
            if candidate.optimizer_cpu_offload
            else ZERO3_GPU_BYTES_PER_PARAMETER
        )
        cpu_bytes_per_parameter = (
            ZERO3_CPU_OFFLOAD_CPU_BYTES_PER_PARAMETER
            if candidate.optimizer_cpu_offload
            else 0
        )
        sharded_state_bytes = math.ceil(
            parameter_count
            * gpu_bytes_per_parameter
            / candidate.trainer_gpus
        )
        optimizer_cpu_bytes = math.ceil(
            parameter_count
            * cpu_bytes_per_parameter
            / candidate.trainer_gpus
        )
        activation_bytes = math.ceil(
            ACTIVATION_GIB_PER_1K_TOKENS
            * GIB
            * p99_tokens
            / 1000
            / candidate.seq_parallel
        )
        estimated_total = (
            sharded_state_bytes
            + activation_bytes
            + all_gather_bytes
            + candidate.reserve_bytes_per_gpu
        )
        budgets.append(
            TrainerMemoryBudget(
                candidate=candidate,
                measured_p99_merged_tokens=p99_tokens,
                text_parameter_count=parameter_count,
                zero3_gpu_bytes_per_parameter=gpu_bytes_per_parameter,
                sharded_model_optimizer_bytes_per_gpu=(
                    sharded_state_bytes
                ),
                optimizer_cpu_bytes_per_rank=optimizer_cpu_bytes,
                activation_bytes_per_gpu=activation_bytes,
                all_gather_temp_bytes_per_gpu=all_gather_bytes,
                reserve_bytes_per_gpu=(
                    candidate.reserve_bytes_per_gpu
                ),
                estimated_total_bytes_per_gpu=estimated_total,
                gpu_memory_bytes=candidate.gpu_memory_bytes,
                fits=estimated_total <= candidate.gpu_memory_bytes,
            )
        )

    selected = next(
        (budget.candidate for budget in budgets if budget.fits),
        None,
    )
    if (
        not budgets[0].fits
        and selected is not None
        and selected.node_count < 6
        and selected.seq_parallel < 2
    ):
        raise ValueError(
            "A tight baseline must escalate nodes or seq_parallel"
        )
    return budgets, selected


def _validate_transfer(
    events: Sequence[TransferEvidence],
    *,
    endpoints: Sequence[str],
    tp_size: int,
    topology: TextModelTopology,
) -> GateResult:
    candidates = [
        event for event in events if event.phase == "after_optimizer"
    ]
    if len(candidates) != 1:
        return GateResult(
            passed=False,
            detail=(
                "expected one after-optimizer transfer, found "
                f"{len(candidates)}"
            ),
        )
    event = candidates[0]
    if event.version != CALIBRATION_ROLLOUTS:
        return GateResult(
            passed=False,
            detail=(
                "after-optimizer transfer version is "
                f"{event.version}, expected {CALIBRATION_ROLLOUTS}"
            ),
        )
    expected_endpoints = set(endpoints)
    observed_endpoints = [
        receipt.endpoint for receipt in event.endpoints
    ]
    if (
        not expected_endpoints
        or len(endpoints) != len(expected_endpoints)
        or len(observed_endpoints) != len(expected_endpoints)
        or set(observed_endpoints) != expected_endpoints
    ):
        return GateResult(
            passed=False,
            detail="transfer endpoint coverage mismatch",
        )
    source_names = set(event.source_fingerprints)
    if any(
        fingerprint.finite_count != fingerprint.numel
        for fingerprint in event.source_fingerprints.values()
    ):
        return GateResult(
            passed=False,
            detail="trainer transfer contains non-finite tensors",
        )
    if not source_names:
        return GateResult(
            passed=False,
            detail="transfer contains no source tensors",
        )
    expected_layers = set(range(topology.num_hidden_layers))
    topology_layers = set(topology.layer_indices)
    if topology_layers != expected_layers:
        return GateResult(
            passed=False,
            detail="model topology layer coverage mismatch",
        )
    source_layers = parameter_layer_indices(source_names)
    if source_layers != expected_layers:
        return GateResult(
            passed=False,
            detail=(
                "trainer source layer coverage mismatch: "
                f"observed={sorted(source_layers)}, "
                f"expected={sorted(expected_layers)}"
            ),
        )
    assert_no_vision_parameters(
        source_names,
        surface="trainer transfer",
    )
    for endpoint in event.endpoints:
        worker_ranks = [worker.rank for worker in endpoint.workers]
        expected_ranks = set(range(tp_size))
        if tp_size > 0 and len(worker_ranks) > tp_size:
            return GateResult(
                passed=False,
                detail=(
                    f"{endpoint.endpoint} gate 3 requires vLLM "
                    "pipeline parallel size 1"
                ),
            )
        if (
            tp_size <= 0
            or len(worker_ranks) != tp_size
            or set(worker_ranks) != expected_ranks
        ):
            return GateResult(
                passed=False,
                detail=f"{endpoint.endpoint} TP-rank coverage mismatch",
            )
        for worker in endpoint.workers:
            receipts = {
                receipt.source_name: receipt
                for receipt in worker.receipts
            }
            if (
                len(worker.receipts) != len(source_names)
                or set(receipts) != source_names
            ):
                return GateResult(
                    passed=False,
                    detail=(
                        f"{endpoint.endpoint} rank {worker.rank} "
                        "tensor coverage mismatch"
                    ),
                )
            categories = set()
            for name, source in event.source_fingerprints.items():
                receipt = receipts[name]
                if (
                    not receipt.loaded_names
                    or not source.matches(
                        receipt.received_fingerprint
                    )
                ):
                    return GateResult(
                        passed=False,
                        detail=(
                            f"{endpoint.endpoint} rank {worker.rank} "
                            f"mismatch for {name}"
                        ),
                    )
                assert_no_vision_parameters(
                    receipt.loaded_names,
                    surface="vLLM transfer",
                )
                expected_categories = {
                    category
                    for parameter_name in [
                        name,
                        *receipt.loaded_names,
                    ]
                    for category in parameter_categories(
                        parameter_name,
                        tied_output_head=topology.tie_word_embeddings,
                    )
                }
                if set(receipt.categories) != expected_categories:
                    return GateResult(
                        passed=False,
                        detail=(
                            f"{endpoint.endpoint} rank {worker.rank} "
                            f"category mismatch for {name}"
                        ),
                    )
                categories.update(expected_categories)
            loaded_layers = parameter_layer_indices(
                loaded_name
                for receipt in worker.receipts
                for loaded_name in receipt.loaded_names
            )
            if loaded_layers != expected_layers:
                return GateResult(
                    passed=False,
                    detail=(
                        f"{endpoint.endpoint} rank {worker.rank} "
                        "loaded layer coverage mismatch: "
                        f"observed={sorted(loaded_layers)}, "
                        f"expected={sorted(expected_layers)}"
                    ),
                )
            if loaded_layers:
                categories.add("backbone")
            missing = (
                required_transfer_categories(topology.num_experts)
                - categories
            )
            if missing:
                return GateResult(
                    passed=False,
                    detail=(
                        f"{endpoint.endpoint} rank {worker.rank} "
                        f"missing categories {sorted(missing)}"
                    ),
                )
    return GateResult(
        passed=True,
        detail=(
            "all after-optimizer tensors, layers, and required categories "
            "reached every TP worker"
        ),
    )


def _validate_parity(
    events: Sequence[ParityEvidence],
    *,
    endpoints: Sequence[str],
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
) -> GateResult:
    expected = {
        (phase, endpoint)
        for phase in ("before_update", "after_update")
        for endpoint in endpoints
    }
    observed = {
        (event.phase, event.endpoint) for event in events
    }
    if (
        not endpoints
        or len(endpoints) != len(set(endpoints))
        or observed != expected
        or len(events) != len(expected)
    ):
        return GateResult(
            passed=False,
            detail="before/after parity endpoint coverage mismatch",
        )

    expected_versions = {
        "before_update": 0,
        "after_update": CALIBRATION_ROLLOUTS,
    }
    recomputed = []
    for event in events:
        if event.version != expected_versions[event.phase]:
            return GateResult(
                passed=False,
                detail=(
                    f"{event.phase} parity version is {event.version}, "
                    f"expected {expected_versions[event.phase]}"
                ),
            )
        if (
            event.prompt_token_ids != list(prompt_token_ids)
            or event.completion_token_ids
            != list(completion_token_ids)
        ):
            return GateResult(
                passed=False,
                detail="parity evidence did not use manifest-pinned token IDs",
            )
        if (
            event.tolerance != PARITY_MAX_ABS_TOLERANCE
            or event.tolerance_basis != PARITY_TOLERANCE_BASIS
        ):
            return GateResult(
                passed=False,
                detail=(
                    "parity tolerance differs from the registered "
                    "0.05-nat default"
                ),
            )
        try:
            computed = make_parity_evidence(
                phase=event.phase,
                version=event.version,
                endpoint=event.endpoint,
                prompt_token_ids=event.prompt_token_ids,
                completion_token_ids=event.completion_token_ids,
                trainer_logprobs=event.trainer_logprobs,
                vllm_logprobs=event.vllm_logprobs,
                tolerance=event.tolerance,
            )
        except ValueError as exc:
            return GateResult(
                passed=False,
                detail=f"invalid parity vectors: {exc}",
            )
        if (
            event.passed != computed.passed
            or not math.isclose(
                event.max_abs_error,
                computed.max_abs_error,
                rel_tol=0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                event.rms_error,
                computed.rms_error,
                rel_tol=0,
                abs_tol=1e-12,
            )
        ):
            return GateResult(
                passed=False,
                detail="parity summary does not match recorded logprobs",
            )
        recomputed.append(computed)

    failed = [event for event in recomputed if not event.passed]
    if failed:
        return GateResult(
            passed=False,
            detail=(
                f"{len(failed)} parity comparisons exceeded tolerance"
            ),
        )
    maximum = max(event.max_abs_error for event in recomputed)
    return GateResult(
        passed=True,
        detail=(
            "all before/after comparisons passed on manifest-pinned "
            f"tokens; maximum error={maximum:.8f} nats"
        ),
    )


def finalize_prerun_manifest(
    spec: PreRunSpec,
    evidence_dir: Path,
) -> PreRunManifest:
    snapshot = Path(spec.model_snapshot)
    descriptor = get_text_model_descriptor(
        spec.model_id,
        spec.model_revision,
    )
    revision_provenance = (
        GEMMA_MODEL_REVISION_PROVENANCE
        if descriptor.model_id == GEMMA_MODEL_ID
        else descriptor.provenance
    )
    user_simulator_model = hash_model_snapshot(
        Path(spec.user_simulator_deployment.snapshot_path),
        USER_SIMULATOR_MODEL_ID,
        USER_SIMULATOR_REVISION,
    )
    if user_simulator_model != spec.user_simulator_deployment.model:
        raise ValueError(
            "Tau2 user simulator snapshot does not match the approved "
            "deployment identity"
        )
    validate_user_simulator_deployment(
        spec.user_simulator_deployment,
        spec.user_simulator,
    )
    model = hash_model_snapshot(
        snapshot,
        spec.model_id,
        spec.model_revision,
    )
    topology = inspect_text_model_snapshot(
        snapshot,
        descriptor,
        model,
    )
    transfers = [
        TransferEvidence.model_validate(record)
        for record in read_evidence_records(
            evidence_dir,
            "transfer",
        )
    ]
    parity = [
        ParityEvidence.model_validate(record)
        for record in read_evidence_records(
            evidence_dir,
            "parity",
        )
    ]
    groups = [
        CalibrationGroupEvidence.model_validate(record)
        for record in read_evidence_records(
            evidence_dir,
            "calibration_groups",
        )
    ]
    calibration = summarize_calibration(groups)

    memory_error = None
    try:
        trainer_memory_budgets, recommended_topology = (
            build_trainer_memory_budgets(
                calibration,
                transfers,
                spec.production_memory_candidates,
            )
        )
    except ValueError as exc:
        trainer_memory_budgets = []
        recommended_topology = None
        memory_error = str(exc)

    gate1 = GateResult(
        passed=(
            spec.user_separation_asserted
            and spec.user_simulator.endpoint
            not in spec.policy_endpoints
            and spec.user_simulator.model != spec.policy_model
        ),
        detail=(
            f"user model={spec.user_simulator.model}, "
            f"endpoint={spec.user_simulator.endpoint}, "
            f"separation asserted={spec.user_separation_asserted}"
        ),
    )
    source_model_pins_match = (
        spec.source_pins == PINNED_SOURCES
        and descriptor.policy_eligible
        and spec.policy_model
        == f"{descriptor.model_id}@{descriptor.revision}"
        and model_revision_is_verified(descriptor)
    )
    gate2 = GateResult(
        passed=source_model_pins_match,
        detail=(
            "executed source/model pins match="
            f"{source_model_pins_match}; revision provenance="
            f"{revision_provenance}"
        ),
    )
    gate3 = _validate_transfer(
        transfers,
        endpoints=spec.policy_endpoints,
        tp_size=spec.expected_tp_size,
        topology=topology,
    )
    gate4 = _validate_parity(
        parity,
        endpoints=spec.policy_endpoints,
        prompt_token_ids=spec.fixed_prompt_token_ids,
        completion_token_ids=spec.fixed_completion_token_ids,
    )
    gate5 = GateResult(
        passed=not spec.packing_enabled,
        detail="cross-rollout sequence packing is disabled",
    )
    affinity_failures = sum(
        group.endpoint_affinity_failures for group in groups
    )
    gate6 = GateResult(
        passed=affinity_failures == 0,
        detail=f"endpoint-affinity failures={affinity_failures}",
    )
    version_failures = sum(
        group.version_transport_failures for group in groups
    )
    gate7 = GateResult(
        passed=version_failures == 0,
        detail=f"version-transport failures={version_failures}",
    )
    gate8 = GateResult(
        passed=(
            spec.mixed_version_loss_gate_passed
            and spec.policy_loss == RUN1_POLICY_LOSS
            and spec.policy_loss_fallback == POLICY_LOSS_FALLBACK
            and spec.policy_loss_fallback_trigger
            == POLICY_LOSS_FALLBACK_TRIGGER
            and spec.gspo_token_upgrade_trigger
            == GSPO_TOKEN_UPGRADE_TRIGGER
        ),
        detail=(
            "mixed-version comparison recorded="
            f"{spec.mixed_version_loss_gate_passed}; "
            f"selected={spec.policy_loss}; "
            f"fallback={spec.policy_loss_fallback} on "
            f"{spec.policy_loss_fallback_trigger}; "
            f"gspo_token trigger={spec.gspo_token_upgrade_trigger}"
        ),
    )
    silent_truncations = sum(
        group.silent_truncations for group in groups
    )
    dropped_rollouts = sum(
        group.attempted_rollouts
        for group in groups
        if group.drop_reason is not None
    )
    accounted = calibration.published_rollouts + dropped_rollouts
    measured_context_limit = max(
        calibration.derived_limits[
            "model_context_tokens"
        ].derived_value,
        calibration.derived_limits[
            "training_sequence_tokens"
        ].derived_value,
    )
    context_fits = (
        measured_context_limit
        <= topology.max_position_embeddings
    )
    memory_fits = recommended_topology is not None
    p99_tokens = (
        trainer_memory_budgets[0].measured_p99_merged_tokens
        if trainer_memory_budgets
        else None
    )
    recommended_name = (
        recommended_topology.name
        if recommended_topology is not None
        else None
    )
    gate9 = GateResult(
        passed=(
            calibration.attempted_rollouts == CALIBRATION_ROLLOUTS
            and calibration.published_rollouts
            == CALIBRATION_ROLLOUTS
            and accounted == CALIBRATION_ROLLOUTS
            and silent_truncations == 0
            and context_fits
            and memory_fits
        ),
        detail=(
            f"attempted={calibration.attempted_rollouts}, "
            f"published={calibration.published_rollouts}, "
            f"dropped groups={sum(calibration.drop_counts.values())}, "
            f"dropped rollouts={dropped_rollouts}, "
            f"silent truncations={silent_truncations}, "
            f"p99 merged tokens={p99_tokens}, "
            f"derived context limit={measured_context_limit}/"
            f"{topology.max_position_embeddings}, "
            f"production topology={recommended_name}, "
            f"memory error={memory_error}"
        ),
    )
    gates = {
        "1_user_separation": gate1,
        "2_source_pins": gate2,
        "3_model_transfer": gate3,
        "4_policy_parity": gate4,
        "5_packing_isolation": gate5,
        "6_endpoint_affinity": gate6,
        "7_version_transport": gate7,
        "8_mixed_version_loss": gate8,
        "9_context_fit": gate9,
    }
    return PreRunManifest(
        job_spec_sha256=spec.job_spec_sha256,
        model=model,
        topology=topology,
        source_pins=spec.source_pins,
        model_revision_provenance=revision_provenance,
        topology_provenance=descriptor.provenance,
        user_simulator=spec.user_simulator,
        user_simulator_deployment=(
            spec.user_simulator_deployment
        ),
        policy_model=spec.policy_model,
        policy_endpoints=spec.policy_endpoints,
        expected_tp_size=spec.expected_tp_size,
        fixed_prompt_token_ids=spec.fixed_prompt_token_ids,
        fixed_completion_token_ids=(
            spec.fixed_completion_token_ids
        ),
        policy_loss=spec.policy_loss,
        policy_loss_fallback=spec.policy_loss_fallback,
        policy_loss_fallback_trigger=(
            spec.policy_loss_fallback_trigger
        ),
        gspo_token_upgrade_trigger=(
            spec.gspo_token_upgrade_trigger
        ),
        transfer_evidence=transfers,
        parity_evidence=parity,
        calibration=calibration,
        trainer_memory_budgets=trainer_memory_budgets,
        recommended_production_topology=recommended_topology,
        gates=gates,
        ready=all(gate.passed for gate in gates.values()),
    )


def require_ready_manifest(manifest: PreRunManifest) -> None:
    failed = [
        name
        for name, gate in manifest.gates.items()
        if not gate.passed
    ]
    if not manifest.ready or failed:
        raise ValueError(
            f"Tau2/Gemma pre-run gates have not passed: {failed}"
        )
