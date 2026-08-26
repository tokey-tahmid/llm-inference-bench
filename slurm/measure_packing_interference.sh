#!/bin/bash
# Measure what packing four TP=1 runs onto one node actually costs.
#
#   sbatch slurm/measure_packing_interference.sh configs/phase2_tp_scaling.yaml 0
#
# Packing four independent single-GPU runs onto one billed 4-GPU node is a 4x
# budget multiplier, and this project needs it. But the four replicas share host
# memory bandwidth, the PCIe root complex and the Lustre mount, so they are not
# independent. The honest move is to quantify the coupling rather than assume it
# away, and to report it as a result in its own right: "how much does co-location
# perturb a single-GPU inference measurement" is a genuinely interesting number
# on a 4x A100 node, and it decides whether the rest of the packed data can carry
# headline claims or only trend claims.
#
# Method: run the SAME server group twice on the same node, in the same job, so
# hardware and software are identical and only placement differs.
#
#   phase A  isolated: one replica, GPU 0, three GPUs left idle
#   phase B  packed:   four replicas, one per GPU, NUMA-pinned
#
# Only GPU 0's replica is compared across phases; the other three exist solely to
# create contention. Artifacts are tagged placement=isolated / placement=packed
# so analysis can pair them.
#
# Isolated runs first: if it ran second it would inherit a page cache warmed by
# the packed phase and look artificially fast.
#
#SBATCH --job-name=lib-packing
#SBATCH --account=p201362
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/packing_%j.out

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

CONFIG="${1:?usage: measure_packing_interference.sh <config.yaml> <group-index>}"
GROUP="${2:?give the server group index to replicate}"
[[ "$CONFIG" = /* ]] || CONFIG="${HERE}/${CONFIG}"

module load "$LIB_MOD_ENV"
module load "$LIB_MOD_APPTAINER"

log() { printf '[packing %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

declare -A GPU_CORES=(
    [0]="24-31,88-95"
    [1]="8-15,72-79"
    [2]="56-63,120-127"
    [3]="40-47,104-111"
)

log "host=$(hostname) job=${SLURM_JOB_ID} config=${CONFIG} group=${GROUP}"
cd "$HERE"

# --- phase A: isolated -------------------------------------------------------
log "phase A: isolated, GPU 0 only, GPUs 1-3 deliberately idle"
taskset -c "${GPU_CORES[0]}" \
    "${LIB_VENV}/bin/python" -m llm_inference_bench.cli run "$CONFIG" \
        --group-index "$GROUP" \
        --gpu-ids 0 \
        --note placement=isolated \
        --note "packing_job=${SLURM_JOB_ID}" \
    2>&1 | sed 's/^/[isolated] /'

log "phase A complete"

# --- phase B: packed ---------------------------------------------------------
log "phase B: packed, four replicas of the same group, one per GPU"
PIDS=()
for i in 0 1 2 3; do
    sleep $(( i * 20 ))
    # Every replica runs the identical configuration. Only GPU 0's artifacts are
    # compared against phase A; GPUs 1-3 are load, not measurements. They are
    # still tagged, so they are identifiable rather than mysterious duplicates.
    if [[ "$i" -eq 0 ]]; then
        ROLE="measured"
    else
        ROLE="contention_source"
    fi
    taskset -c "${GPU_CORES[$i]}" \
        "${LIB_VENV}/bin/python" -m llm_inference_bench.cli run "$CONFIG" \
            --group-index "$GROUP" \
            --gpu-ids "$i" \
            --note placement=packed \
            --note "packing_role=${ROLE}" \
            --note "packing_gpu=${i}" \
            --note "packing_job=${SLURM_JOB_ID}" \
        > "${LIB_LOGS}/packing_${SLURM_JOB_ID}_gpu${i}.log" 2>&1 &
    PIDS+=($!)
done

FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || { log "packed replica on GPU ${i} FAILED"; FAILED=1; }
done

log "phase B complete (failed=${FAILED})"
log "compare with: python analysis/plot_packing_interference.py"
log "PACKING INTERFERENCE MEASUREMENT COMPLETE"
exit "$FAILED"
