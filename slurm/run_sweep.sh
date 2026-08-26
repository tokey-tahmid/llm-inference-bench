#!/bin/bash
# Execute a sweep on a GPU node.
#
#   sbatch slurm/run_sweep.sh configs/phase1_validation.yaml
#   sbatch --array=0-5 slurm/run_sweep.sh configs/full.yaml   # one server group per task
#
# A gpu job bills the whole 4-GPU node regardless of how many GPUs are used, so
# a TP=1 sweep run one-group-per-node wastes three quarters of the spend. See
# slurm/run_sweep_packed.sh for the packed variant. This script is the
# unpacked, measurement-clean form: use it for headline numbers and for any
# group with tensor_parallel_size > 1.
#
#SBATCH --job-name=lib-sweep
#SBATCH --account=p201362
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/sweep_%A_%a.out

set -euo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -eu
fi

find_repo() {
    local c
    for c in "${LIB_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)" \
             "/home/users/u104240/apps/inference-work/llm-inference-bench"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
HERE="$(find_repo)" || { echo "cannot locate repo root" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

CONFIG="${1:?usage: run_sweep.sh <config.yaml> [extra lib-run args...]}"
shift || true
[[ "$CONFIG" = /* ]] || CONFIG="${HERE}/${CONFIG}"

module load "$LIB_MOD_ENV"
module load "$LIB_MOD_APPTAINER"

log() { printf '[sweep %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "host=$(hostname) job=${SLURM_JOB_ID} array_task=${SLURM_ARRAY_TASK_ID:-none}"
log "config=${CONFIG}"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader

GROUP_ARG=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    GROUP_ARG=(--group-index "${SLURM_ARRAY_TASK_ID}")
    log "array task -> server group index ${SLURM_ARRAY_TASK_ID}"
fi

# Startup is dominated by weight loading off Lustre. When many array tasks start
# at once they contend for the same files; stagger by task id so the contention
# does not land in the measured startup time.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${SLURM_ARRAY_TASK_ID}" != "0" ]]; then
    STAGGER=$(( SLURM_ARRAY_TASK_ID * 15 ))
    log "staggering start by ${STAGGER}s to avoid Lustre contention on model load"
    sleep "$STAGGER"
fi

cd "$HERE"
"${LIB_VENV}/bin/python" -m llm_inference_bench.cli run "$CONFIG" "${GROUP_ARG[@]}" "$@"

log "SWEEP TASK COMPLETE"
