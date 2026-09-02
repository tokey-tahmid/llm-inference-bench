#!/bin/bash
# Probe the vLLM container for Ray and multi-node headroom.
#
# Cheap: one node, one gpu bill, ~5 minutes. Answers three questions before
# any 2-node attempt is worth submitting:
#
#   1. Is Ray installed in the vLLM image at a usable version?
#   2. Can `ray start --head` and `ray status` succeed inside Apptainer?
#   3. What IP does the node advertise to peers, and does it match the IB HCA?
#
# All output goes to the shared log; nothing is written under results/.
#
#SBATCH --job-name=lib-rayprobe
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/rayprobe_%j.out

set -uo pipefail
if ! type module >/dev/null 2>&1; then set +eu; source /etc/profile; set -u; fi
HERE=/home/users/u104240/apps/inference-work/llm-inference-bench
source "$HERE/env.sh"

module load "$LIB_MOD_ENV"
module load "$LIB_MOD_APPTAINER"

log() { printf '[rayprobe %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "host=$(hostname) job=${SLURM_JOB_ID}"
log "ip on default route:"; ip -4 addr show 2>/dev/null | awk '/inet /{print "  "$0}' | head -10
log "ib0 addr (if any):"
ip -4 addr show ib0 2>/dev/null | awk '/inet /{print "  "$0}' || echo "  no ib0"

APPT_BIND=(--bind "${LIB_DATA}:${LIB_DATA}:rw")
APPT_ENV=(
    --env "TMPDIR=/tmp"
    --env "VLLM_CACHE_ROOT=/tmp/vllm_cache"
    --env "TRITON_CACHE_DIR=/tmp/triton_cache"
    --env "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache"
    --env "TORCH_HOME=/tmp/torch_home"
    --env "XDG_CACHE_HOME=/tmp/xdg_cache"
    --env "HF_HUB_OFFLINE=1"
    --env "TRANSFORMERS_OFFLINE=1"
)

log "=== ray version and import ==="
apptainer exec --cleanenv "${APPT_BIND[@]}" "${APPT_ENV[@]}" "$LIB_SIF_VLLM" \
    python3 -c 'import ray, sys; print("ray", ray.__version__); print("python", sys.version.split()[0])' \
    2>&1 | sed 's/^/  /'

log "=== ray start --head (single node self-test) ==="
apptainer exec --cleanenv --nv "${APPT_BIND[@]}" "${APPT_ENV[@]}" "$LIB_SIF_VLLM" \
    bash -c '
        set +e
        ray start --head --port=6379 --num-gpus=4 --temp-dir=/tmp/ray 2>&1 | head -60
        rc=$?
        echo "--- ray status ---"
        ray status 2>&1 | head -40
        echo "--- ray stop ---"
        ray stop --force 2>&1 | head -20
        exit $rc
    ' 2>&1 | sed 's/^/  /'

log "PROBE COMPLETE"
