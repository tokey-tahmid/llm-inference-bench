#!/bin/bash
# Run four independent TP=1 server groups on one billed GPU node, one per GPU.
#
#   sbatch slurm/run_sweep_packed.sh configs/foo.yaml 0 1 2 3
#
# WHY: a gpu job bills the whole 4-GPU node whatever it uses, so a TP=1 sweep run
# one-group-per-node wastes 75% of every billed hour. Against a 117 node-hour
# monthly allocation that is the difference between covering the matrix and not.
#
# WHAT IT COSTS: the four runs share host memory bandwidth, the PCIe root
# complex, and the Lustre mount. They are NOT independent, and packing is
# therefore not free. Each replica is pinned to its GPU's NUMA-local cores to
# minimise the coupling, but the residual interference is real and must be
# measured, not assumed away: run slurm/measure_packing_interference.sh first and
# report the delta. Do not use this script for headline numbers until that delta
# is known and stated.
#
# Never use this for tensor_parallel_size > 1: those groups need every GPU on the
# node and the NVLink mesh between them.
#
#SBATCH --job-name=lib-packed
#SBATCH --account=p201362
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/packed_%j.out

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

CONFIG="${1:?usage: run_sweep_packed.sh <config.yaml> <group-index>...}"
shift
GROUPS=("$@")
[[ ${#GROUPS[@]} -gt 0 ]] || { echo "give at least one group index" >&2; exit 1; }
[[ ${#GROUPS[@]} -le 4 ]] || { echo "at most 4 groups (one per GPU)" >&2; exit 1; }
[[ "$CONFIG" = /* ]] || CONFIG="${HERE}/${CONFIG}"

module load "$LIB_MOD_ENV"
module load "$LIB_MOD_APPTAINER"

log() { printf '[packed %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# NUMA-local cores per GPU, measured on mel2017 (probe 5143256). Pinning each
# replica to its own GPU's cores keeps the packed runs from fighting over the
# same memory controllers, which is the largest avoidable source of interference.
declare -A GPU_CORES=(
    [0]="24-31,88-95"
    [1]="8-15,72-79"
    [2]="56-63,120-127"
    [3]="40-47,104-111"
)

log "host=$(hostname) job=${SLURM_JOB_ID} groups=${GROUPS[*]}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Verify the topology still matches what the pinning table assumes. A silent
# change here would degrade every packed measurement without failing anything.
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$GPU_COUNT" -ne 4 ]]; then
    log "FATAL: expected 4 GPUs, found ${GPU_COUNT}. The NUMA pinning table is stale."
    exit 1
fi

cd "$HERE"
PIDS=()
for i in "${!GROUPS[@]}"; do
    gid="${GROUPS[$i]}"
    cores="${GPU_CORES[$i]}"
    log "group ${gid} -> GPU ${i}, cores ${cores}"
    # Stagger launches: four simultaneous model loads off the same Lustre mount
    # contend badly, and that contention would land in the measured startup time.
    sleep $(( i * 20 ))
    taskset -c "$cores" \
        "${LIB_VENV}/bin/python" -m llm_inference_bench.cli run "$CONFIG" \
            --group-index "$gid" \
            --gpu-ids "$i" \
        > "${LIB_LOGS}/packed_${SLURM_JOB_ID}_g${gid}.log" 2>&1 &
    PIDS+=($!)
done

log "launched ${#PIDS[@]} replicas, waiting"
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        log "group ${GROUPS[$i]} completed"
    else
        log "group ${GROUPS[$i]} FAILED (see packed_${SLURM_JOB_ID}_g${GROUPS[$i]}.log)"
        FAILED=1
    fi
done

log "PACKED SWEEP COMPLETE (failed=${FAILED})"
exit "$FAILED"
