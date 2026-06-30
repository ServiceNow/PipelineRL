#!/bin/bash
# Launch the terminal env-fleet: two single-replica CPU-only eai jobs
# (terminal-envs-a / terminal-envs-b), each running 12 TerminalEnvironmentServer
# instances on ports 7777-7788 via run_environment_fleet. The GPU training job
# (conf/terminal.yaml, placement=external) reaches them over account-scoped
# internal-dns at dns-${EAI_ACCOUNT_ID}-terminal-envs-{a,b}:<port>.
#
# Single-replica + --gpu 0 so these never grab a full GPU-bearing node. 12
# servers/node x 0.9GiB disk cap = 10.8GiB (< 16GiB local-ephemeral envelope).
set -euo pipefail

CONDA_ENV=${CONDA_ENV:-pipeline-rl}
ACCOUNT=${ACCOUNT:-snow.research.adea}
COUNT=${COUNT:-12}
START_PORT=${START_PORT:-7777}
END_PORT=$((START_PORT + COUNT - 1))
S=$(dirname "$(readlink -f "$0")")
# Unique per-launch EAI job-name suffix: a bare `terminal_envs_a` collides with a
# prior FAILED/CANCELLED job of that exact name ("resource with this name already
# exists"). internal-dns names stay terminal-envs-{a,b} so the trainer still finds them.
FLEET_RUN_ID=${FLEET_RUN_ID:-$(date +%s)}

# internal-dns ports fragment (one entry per server port)
ports_yaml() {
  for p in $(seq "$START_PORT" "$END_PORT"); do
    printf '        - {port: %d, target-port: %d, protocol: TCP}\n' "$p" "$p"
  done
}

launch_one() {
  local suffix=$1   # a | b
  local dns="terminal-envs-${suffix}"
  local out="/mnt/llmd/results/exps/rafa/terminal/env_fleet_${suffix}"
  local yaml="${S}/.env_fleet_${suffix}.yaml"
  cat > "$yaml" <<EOF
image: registry.toolkit-sp.yul201.service-now.com/snow.research.tapes/interactive-toolkit:throughput
data:
  - 'snow.research.tapes.rafael_pardinas_home:/home/toolkit:rw'
  - 'snow.research.tapes.data:/mnt/llmd/data:rw'
  - 'snow.research.tapes.results:/mnt/llmd/results:rw'
resources: {cpu: 64, gpu: 0, mem: 512}
options:
  internal-dns:
      name: ${dns}
      ports:
$(ports_yaml)
name: terminal_envs_${suffix}_${FLEET_RUN_ID}
bid: 9999
workdir: /home/toolkit/PipelineRL
environmentVars:
  - HOME=/home/toolkit
  - PYTHONPATH=/home/toolkit/PipelineRL
EOF
  echo "=== launching ${dns} (ports ${START_PORT}-${END_PORT}) ==="
  # --non-preemptable on the CLI: eai silently ignores a YAML `preemptable: false`
  # (the job stays preemptable). Our account's reserved non-preemptable CPU pool is
  # idle, so this keeps the fleet from being evicted off contended GPU nodes.
  eai job new -f "$yaml" --account "$ACCOUNT" --non-preemptable -- \
    /opt/conda/bin/conda run -n "$CONDA_ENV" --no-capture-output bash -c \
    "python -m pipelinerl.entrypoints.run_environment_fleet --config-name terminal --config-dir /home/toolkit/PipelineRL/conf output_dir=${out} +fleet.environment_key=terminal +fleet.start_port=${START_PORT} +fleet.count=${COUNT}"
}

# Optional arg selects which fleet(s) to launch: a | b | both (default both).
# Used to restore a single dead fleet without disturbing the live one.
TARGET=${1:-both}
case "$TARGET" in
  a) launch_one a ;;
  b) launch_one b ;;
  both) launch_one a; launch_one b ;;
  *) echo "usage: $0 [a|b|both]" >&2; exit 1 ;;
esac
