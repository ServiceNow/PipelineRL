#!/bin/bash
set -euo pipefail

PHASE=${1:-}
if [[ "${PHASE}" != "calibration" && "${PHASE}" != "production" ]]; then
    echo "usage: $0 calibration|production" >&2
    exit 2
fi

require_resolved() {
    local name=$1
    local value=${!name:-}
    if [[ -z "${value}" || "${value}" == PENDING_* ]]; then
        echo "${name} is unresolved" >&2
        exit 2
    fi
}

REQUIRED_NAMES=(
    TAU2_OUTPUT_DIR
    TAU2_MODEL_SNAPSHOT
    TAU2_PRERUN_SPEC
    TAU2_PRERUN_MANIFEST
    TAU2_JOB_SPEC_PATH
    TAU2_PREPARED_DATA_MANIFEST
    TAU2_AIRLINE_JSONL
    TAU2_RETAIL_JSONL
    TAU2_TELECOM_JSONL
    TAU2_USER_SIMULATOR_ENDPOINT
    TAU2_JUDGE_ENDPOINT
    TAU2_USER_SIM_JOB_SPEC_PATH
    TAU2_USER_SIM_SNAPSHOT
    TAU2_USER_API_KEY
    TAU2_RUN_SEED
    TAU2_JUDGE_INITIAL_MAX_TOKENS
    TAU2_JUDGE_RETRY_MAX_TOKENS
    TAU2_AUXILIARY_MODEL_TIMEOUT_S
    TAU2_TRAINER_GPU_MEMORY_BYTES
)
for name in "${REQUIRED_NAMES[@]}"; do
    require_resolved "${name}"
done

require_resolved MASTER_ADDR
require_resolved WORLD_SIZE
require_resolved RANK

if [[ "${PHASE}" == "calibration" && "${WORLD_SIZE}" != "4" ]]; then
    echo "calibration requires exactly four 8-GPU nodes" >&2
    exit 2
fi

NEMO_GYM_SHA=5f92a73217258074b74b7be26526c69f0ce3075d
POLICY_MODEL=Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a
USER_MODEL=qwen3.5-27b-user@fc05daec18b0a78c049392ed2e771dde82bdf654
JUDGE_MODEL=qwen3.5-27b-judge@fc05daec18b0a78c049392ed2e771dde82bdf654
USER_ENDPOINT=${TAU2_USER_SIMULATOR_ENDPOINT}
JUDGE_ENDPOINT=${TAU2_JUDGE_ENDPOINT}
JUDGE_TEMPERATURE=0.6
JUDGE_TOP_P=0.95
JUDGE_TOP_K=20
REQUEST_TIMEOUT_S=3600.0
CLUSTER_BASE=${MASTER_ADDR%-*}
ACTOR_HOST=${CLUSTER_BASE}-$((WORLD_SIZE - 1))
POLICY_ENDPOINTS=(
    "http://${ACTOR_HOST}:8080"
    "http://${ACTOR_HOST}:8082"
    "http://${ACTOR_HOST}:8084"
    "http://${ACTOR_HOST}:8086"
)
POLICY_ENDPOINTS_JSON=$(printf '%s\n' "${POLICY_ENDPOINTS[@]}" | jq -R . | jq -sc .)
PROMPT_TOKEN_IDS=$(jq -c '.fixed_prompt_token_ids' "${TAU2_PRERUN_SPEC}")
COMPLETION_TOKEN_IDS=$(jq -c '.fixed_completion_token_ids' "${TAU2_PRERUN_SPEC}")

export TAU2_GYM_HEAD_URL="http://${MASTER_ADDR}:11000"
export TAU2_MODEL_SNAPSHOT
export TAU2_PRERUN_SPEC
export TAU2_PRERUN_MANIFEST
export TAU2_JOB_SPEC_PATH
export TAU2_PREPARED_DATA_MANIFEST
export TAU2_AIRLINE_JSONL
export TAU2_RETAIL_JSONL
export TAU2_TELECOM_JSONL
export TAU2_USER_SIMULATOR_ENDPOINT
export TAU2_JUDGE_ENDPOINT
export TAU2_USER_SIM_JOB_SPEC_PATH
export TAU2_USER_SIM_SNAPSHOT

LAUNCH_ARGS=(
    python -m pipelinerl.launch
    --config-name tau2_qwen
    "output_dir=${TAU2_OUTPUT_DIR}"
    "tau2_prerun.phase=${PHASE}"
    "tau2_prerun.enabled=$([[ "${PHASE}" == "calibration" ]] && echo true || echo false)"
    "tau2_prerun.policy_endpoints=${POLICY_ENDPOINTS_JSON}"
    "tau2_prerun.fixed_prompt_token_ids=${PROMPT_TOKEN_IDS}"
    "tau2_prerun.fixed_completion_token_ids=${COMPLETION_TOKEN_IDS}"
    "seed=${TAU2_RUN_SEED}"
    "tau2_gym.judge_temperature=${JUDGE_TEMPERATURE}"
    "tau2_gym.judge_top_p=${JUDGE_TOP_P}"
    "tau2_gym.judge_top_k=${JUDGE_TOP_K}"
    "tau2_gym.judge_seed=${TAU2_RUN_SEED}"
    "tau2_gym.judge_initial_max_tokens=${TAU2_JUDGE_INITIAL_MAX_TOKENS}"
    "tau2_gym.judge_retry_max_tokens=${TAU2_JUDGE_RETRY_MAX_TOKENS}"
    "tau2_gym.auxiliary_model_timeout_s=${TAU2_AUXILIARY_MODEL_TIMEOUT_S}"
)
if [[ "${PHASE}" == "calibration" ]]; then
    LAUNCH_ARGS+=(
        finetune.max_train_steps=1
        finetune.interrupt_train_steps=1
    )
fi

# This is the durable launch boundary: revision, spec/manifest readiness, job
# digest, identities, topology, loss decision, and measured caps are checked
# before a Gym, vLLM, trainer, or actor process is started.
PIPELINERL_PREFLIGHT_ONLY=1 "${LAUNCH_ARGS[@]}"

GYM_RUN_DIR=${TAU2_OUTPUT_DIR}/gym
GYM_ROOT=${GYM_RUN_DIR}/nemo-gym
GYM_READY_FILE=${GYM_RUN_DIR}/ready
GYM_PID=

cleanup() {
    if [[ -n "${GYM_PID}" ]]; then
        kill "${GYM_PID}" 2>/dev/null || true
        wait "${GYM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${RANK}" == "0" ]]; then
    mkdir -p "${GYM_RUN_DIR}"
    if [[ ! -d "${GYM_ROOT}/.git" ]]; then
        git clone https://github.com/NVIDIA-NeMo/Gym.git "${GYM_ROOT}"
    fi
    git -C "${GYM_ROOT}" checkout --detach "${NEMO_GYM_SHA}"
    (
        cd "${GYM_ROOT}"
        uv sync --frozen
    )

    GYM_ARGS=(
        python -m pipelinerl.entrypoints.run_tau2_gym
        --gym-root "${GYM_ROOT}"
        --run-dir "${GYM_RUN_DIR}"
        --policy-model-name "${POLICY_MODEL}"
        --user-model-url "${USER_ENDPOINT}"
        --user-model-name "${USER_MODEL}"
        --judge-model-url "${JUDGE_ENDPOINT}"
        --judge-model-name "${JUDGE_MODEL}"
        --policy-thinking-enabled
        --user-thinking-enabled
        --judge-thinking-enabled
        --judge-temperature "${JUDGE_TEMPERATURE}"
        --judge-top-p "${JUDGE_TOP_P}"
        --judge-top-k "${JUDGE_TOP_K}"
        --judge-seed "${TAU2_RUN_SEED}"
        --judge-initial-max-tokens "${TAU2_JUDGE_INITIAL_MAX_TOKENS}"
        --judge-retry-max-tokens "${TAU2_JUDGE_RETRY_MAX_TOKENS}"
        --auxiliary-model-timeout-s "${TAU2_AUXILIARY_MODEL_TIMEOUT_S}"
        --request-timeout-s "${REQUEST_TIMEOUT_S}"
        --uses-reasoning-parser
        --host "${MASTER_ADDR}"
        --head-port 11000
        --service-port-start 12000
        --max-steps 200
    )
    for endpoint in "${POLICY_ENDPOINTS[@]}"; do
        GYM_ARGS+=(--policy-url "${endpoint}")
    done
    "${GYM_ARGS[@]}" >"${GYM_RUN_DIR}/gym.log" 2>&1 &
    GYM_PID=$!

    ready=false
    for _ in $(seq 1 300); do
        if curl -fsS "${TAU2_GYM_HEAD_URL}/global_config_dict_yaml" >/dev/null; then
            ready=true
            break
        fi
        sleep 2
    done
    if [[ "${ready}" != "true" ]]; then
        echo "Tau2 Gym did not become ready within 600 seconds" >&2
        exit 1
    fi
    : >"${GYM_READY_FILE}"
else
    ready=false
    for _ in $(seq 1 300); do
        if [[ -f "${GYM_READY_FILE}" ]]; then
            ready=true
            break
        fi
        sleep 2
    done
    if [[ "${ready}" != "true" ]]; then
        echo "Tau2 Gym readiness barrier timed out" >&2
        exit 1
    fi
fi

"${LAUNCH_ARGS[@]}"

if [[ "${PHASE}" == "calibration" && "${RANK}" == "0" ]]; then
    FINALIZE_ARGS=(python -m pipelinerl.entrypoints.run_tau2_prerun finalize)
    FINALIZE_ARGS+=(--spec "${TAU2_PRERUN_SPEC}")
    FINALIZE_ARGS+=(--evidence-dir "${TAU2_OUTPUT_DIR}/tau2_prerun_evidence")
    FINALIZE_ARGS+=(--output "${TAU2_PRERUN_MANIFEST}")
    "${FINALIZE_ARGS[@]}"
fi
