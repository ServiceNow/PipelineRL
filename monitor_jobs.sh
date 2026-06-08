#!/bin/bash
# Monitor two comparison jobs for failure/cancellation/preemption.
# Usage: bash monitor_jobs.sh <ds_job_id> <fastllm_job_id> <ds_exp_dir> <fastllm_exp_dir>

DS_JOB="${1:-fe9561a0-5c66-4971-88b3-d38bcab0b6e4}"
FL_JOB="${2:-18baa4d1-8f91-4153-9d1c-0affb7d62536}"
DS_DIR="${3:-/mnt/shared/denis/math_7b_results/math_7b_ds_fastllm_4node_20260428_135427}"
FL_DIR="${4:-/mnt/shared/denis/math_7b_results/math_7b_4node_fastllm_gspo_20260428_135448}"

BAD_STATES="FAILED CANCELLED PREEMPTED INTERRUPTED"
INTERVAL=120  # seconds between polls

log() { echo "[$(date '+%H:%M:%S')] $*"; }

check_job() {
    local job_id="$1"
    local label="$2"
    local state
    state=$(eai job get "$job_id" 2>/dev/null | awk 'NR==2{print $2}')
    if [ -z "$state" ]; then
        state="UNKNOWN"
    fi
    for bad in $BAD_STATES; do
        if [ "$state" = "$bad" ]; then
            log "ALERT: $label ($job_id) is $state"
            return 1
        fi
    done
    log "$label ($job_id): $state"
    return 0
}

check_dir() {
    local dir="$1"
    local label="$2"
    local count
    count=$(find "$dir" -maxdepth 1 -mindepth 1 2>/dev/null | wc -l)
    log "$label dir has $count top-level entries"
}

log "Monitoring DS job:     $DS_JOB"
log "Monitoring FastLLM job: $FL_JOB"
log "DS dir:     $DS_DIR"
log "FastLLM dir: $FL_DIR"
log "Polling every ${INTERVAL}s. Ctrl-C to stop."
echo ""

ds_alive=1
fl_alive=1

while true; do
    if [ $ds_alive -eq 1 ]; then
        check_job "$DS_JOB" "DS" || ds_alive=0
    fi
    if [ $fl_alive -eq 1 ]; then
        check_job "$FL_JOB" "FastLLM" || fl_alive=0
    fi
    check_dir "$DS_DIR" "DS"
    check_dir "$FL_DIR" "FastLLM"
    echo ""

    if [ $ds_alive -eq 0 ] && [ $fl_alive -eq 0 ]; then
        log "Both jobs ended. Exiting."
        break
    fi

    sleep "$INTERVAL"
done
