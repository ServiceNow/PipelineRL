#!/bin/bash
# Unified lifecycle for the TMax terminal RL experiment: bring the external env
# fleet and the GPU training job UP and DOWN together so the fleet always runs
# the same commit as the job.
#
# Why: gspo_38 wedged because the env fleets were launched days earlier, ran
# pre-reaper code, and held leaked gspo_37 sessions -> every /start_task got 503
# capacity and the trainer never reached generation. Relaunching the fleet fresh
# on every `up` keeps fleet code == job code and clears any leaked sessions; the
# env reaper (ee534de) self-heals future leaks within its TTL.
#
# Env servers stay OFF the trainer nodes (separate CPU-only fleet jobs) per the
# placement decision; this script only unifies their lifecycle, not placement.
#
# Usage:
#   ./terminal_exp.sh up <exp_name>   # e.g. up gspo_39: fresh fleets -> health-gate -> training job
#   ./terminal_exp.sh down            # kill training job + fleets together
#   ./terminal_exp.sh status          # show alive terminal infra + env /health summary
set -euo pipefail

S=$(dirname "$(readlink -f "$0")")
ACCOUNT=${ACCOUNT:-snow.research.adea}
COUNT=${COUNT:-12}
START_PORT=${START_PORT:-7777}
END_PORT=$((START_PORT + COUNT - 1))
# Required from the environment (set in EAI sessions); no hard-coded fallback so
# the script cannot silently target the wrong account's internal-dns.
EAI_ACCOUNT_ID=${EAI_ACCOUNT_ID:-}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-600}
FLEET_NAME_FILTER=${FLEET_NAME_FILTER:-terminal_envs}
JOB_NAME_FILTER=${JOB_NAME_FILTER:-terminal_qwen35_9b}

# `--field id` returns bare UUIDs with no header; `--me` scopes to our own jobs so
# a kill can never hit a teammate's similarly-named job.
_alive_ids() {  # $1 = name filter
  eai job ls --me --state alive -N "$1" --field id 2>/dev/null || true
}

kill_by_filter() {  # $1 = name filter, $2 = label
  local ids; ids=$(_alive_ids "$1")
  if [ -z "$ids" ]; then echo "  no live $2 to kill"; return 0; fi
  for id in $ids; do echo "  kill $2 $id"; eai job kill "$id" >/dev/null 2>&1 || true; done
}

wait_dead() {  # $1 = name filter, $2 = label; FAIL CLOSED if still alive
  for _ in $(seq 1 30); do
    [ -z "$(_alive_ids "$1")" ] && { echo "  all $2 dead"; return 0; }
    sleep 5
  done
  echo "ERROR: $2 still alive after wait; refusing to relaunch (DNS-name conflict risk)" >&2
  return 1
}

# Gate requires status ok AND active==0 on every endpoint: a fresh fleet has zero
# sessions, so active>0 here means a leaked/stale slot -> never launch training
# into a wedged fleet (the gspo_38 failure mode). Fail closed on timeout.
wait_fleets_healthy() {
  local want=$((COUNT * 2))
  echo "=== waiting for $want env servers status=ok AND active==0 (timeout ${HEALTH_TIMEOUT}s) ==="
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    local ok=0
    for suffix in a b; do
      local host="dns-${EAI_ACCOUNT_ID}-terminal-envs-${suffix}"
      for p in $(seq "$START_PORT" "$END_PORT"); do
        local resp; resp=$(curl -s --max-time 3 "http://${host}:${p}/health" 2>/dev/null || true)
        # whitespace-tolerant to survive compact vs spaced JSON; active==0 (no leaked
        # slot) and capacity==1 (n_envs=1) required, not just HTTP/status.
        if echo "$resp" | rg -q '"status":\s*"ok"' \
           && echo "$resp" | rg -q '"active":\s*0[,}]' \
           && echo "$resp" | rg -q '"capacity":\s*1[,}]'; then
          ok=$((ok + 1))
        fi
      done
    done
    echo "  ready (ok + idle): ${ok}/${want}"
    [ "$ok" -ge "$want" ] && { echo "all env servers healthy and idle"; return 0; }
    sleep 15
  done
  echo "ERROR: not all env servers reached status=ok+active==0 within ${HEALTH_TIMEOUT}s; aborting before training launch" >&2
  return 1
}

case "${1:-}" in
  up)
    EXP_NAME=${2:-}
    [ -z "$EXP_NAME" ] && { echo "usage: $0 up <exp_name>  (e.g. up gspo_39)" >&2; exit 1; }
    # Fail-closed preflight: eai present + authed + account id resolvable.
    command -v eai >/dev/null || { echo "ERROR: eai CLI not found" >&2; exit 1; }
    eai job ls --me -n 1 >/dev/null 2>&1 || { echo "ERROR: eai auth/list failed; run: eai login" >&2; exit 1; }
    [ -n "${EAI_ACCOUNT_ID}" ] || { echo "ERROR: EAI_ACCOUNT_ID unset" >&2; exit 1; }
    echo "using EAI_ACCOUNT_ID=${EAI_ACCOUNT_ID}"
    echo "=== [1/4] kill any stale training job + fleets ==="
    kill_by_filter "$JOB_NAME_FILTER" job
    kill_by_filter "$FLEET_NAME_FILTER" fleet
    wait_dead "$JOB_NAME_FILTER" job
    wait_dead "$FLEET_NAME_FILTER" fleet
    echo "=== [2/4] launch fresh fleets (current code) ==="; bash "${S}/env_fleet.sh"
    echo "=== [3/4] health-gate the fleets ==="; wait_fleets_healthy
    echo "=== [4/4] launch training job ${EXP_NAME} ==="
    JOB_NAME="terminal_qwen35_9b_${EXP_NAME}" bash "${S}/qwen35_9b_terminal.sh"
    echo "=== up complete for ${EXP_NAME} ==="
    ;;
  down)
    # Kill training job FIRST (stops new /start_task load), then the fleets. Issue
    # both kills before waiting so a slow job-cancel cannot leave fleets alive.
    # Wait for CANCELLED so a follow-on `up` cannot collide on the internal-dns name.
    echo "=== kill training job ==="; kill_by_filter "$JOB_NAME_FILTER" job
    echo "=== kill fleets ==="; kill_by_filter "$FLEET_NAME_FILTER" fleet
    echo "=== wait for CANCELLED ==="
    rc=0
    wait_dead "$JOB_NAME_FILTER" job || rc=1
    wait_dead "$FLEET_NAME_FILTER" fleet || rc=1
    [ "$rc" -eq 0 ] && echo "=== teardown complete ===" || { echo "ERROR: teardown incomplete; check eai job ls --me" >&2; exit 1; }
    ;;
  status)
    echo "=== alive terminal infra ==="
    eai job ls --me --state alive 2>/dev/null | rg -i terminal || echo "  none alive"
    echo "=== env /health summary ==="
    if [ -z "${EAI_ACCOUNT_ID}" ]; then
      echo "  EAI_ACCOUNT_ID unset; cannot probe /health"
    else
      for suffix in a b; do
        local_host="dns-${EAI_ACCOUNT_ID}-terminal-envs-${suffix}"
        up=0; idle=0
        for p in $(seq "$START_PORT" "$END_PORT"); do
          resp=$(curl -s --max-time 3 "http://${local_host}:${p}/health" 2>/dev/null || true)
          if echo "$resp" | rg -q '"status":\s*"ok"'; then up=$((up + 1)); fi
          if echo "$resp" | rg -q '"active":\s*0[,}]'; then idle=$((idle + 1)); fi
        done
        echo "  fleet ${suffix}: ${up}/${COUNT} up, ${idle}/${COUNT} idle(active==0)"
      done
    fi
    ;;
  *)
    echo "usage: $0 {up <exp_name>|down|status}" >&2; exit 1
    ;;
esac
