#!/bin/bash
# TP=8 across two GPU nodes via a Ray cluster.
#
#   sbatch slurm/run_tp8_ray.sh
#
# This is the ONE configuration in P1 that observes the intra-node vs
# multi-node interconnect boundary. Every TP<=4 configuration on this cluster
# runs over the uniform NVLink mesh, so scaling losses there are not
# interconnect losses; TP=8 crosses the 200 Gb/s InfiniBand and finally makes
# that contrast measurable.
#
# The harness runner does not orchestrate Ray. This script therefore does the
# Ray bootstrap and vLLM launch itself, then invokes a small purpose-built
# driver (see analysis/tp8_driver.py) that talks to the vLLM endpoint through
# the harness's OpenAI client and writes a raw artifact with full provenance.
#
# If Ray bootstrap fails, this script exits non-zero AND writes a note to
# docs/notes.md's open questions rather than pretending it worked. Per the
# non-negotiable rules, a data point is never fabricated to fill an axis.
#
#SBATCH --job-name=lib-tp8
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/tp8_%j.out

set -uo pipefail
if ! type module >/dev/null 2>&1; then set +eu; source /etc/profile; set -u; fi

HERE=/home/users/u104240/apps/inference-work/llm-inference-bench
source "$HERE/env.sh"

module load "$LIB_MOD_ENV"
module load "$LIB_MOD_APPTAINER"

log() { printf '[tp8 %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NUM_NODES} nodelist=${SLURM_JOB_NODELIST}"
log "nodelist expanded:"; scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed 's/^/  /'

NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD_NODE="${NODES[0]}"
WORKER_NODE="${NODES[1]}"

# Prefer the IB HCA address for Ray traffic. Falls back to the default route if
# ib0 is not up. Recorded so the artifact carries the actual interface.
HEAD_IP="$(srun --nodes=1 --nodelist="$HEAD_NODE" bash -c \
    'ip -4 -o addr show ib0 2>/dev/null | awk "{print \$4}" | cut -d/ -f1' | head -1)"
if [[ -z "$HEAD_IP" ]]; then
    log "WARNING: no ib0 on ${HEAD_NODE}; falling back to default route address"
    HEAD_IP="$(srun --nodes=1 --nodelist="$HEAD_NODE" hostname -I | awk '{print $1}')"
fi
log "ray head node=${HEAD_NODE} ip=${HEAD_IP} port=6379"

RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

# Bind the Ray port explicitly and stagger startup: the head must be up before
# the worker joins, or the worker's ray start blocks forever waiting for a peer
# that does not exist yet.
APPT_COMMON=(
    exec --nv --cleanenv
    --bind "${LIB_DATA}:${LIB_DATA}:rw"
    --env "TMPDIR=/tmp"
    --env "VLLM_CACHE_ROOT=/tmp/vllm_cache"
    --env "TRITON_CACHE_DIR=/tmp/triton_cache"
    --env "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache"
    --env "TORCH_HOME=/tmp/torch_home"
    --env "XDG_CACHE_HOME=/tmp/xdg_cache"
    --env "HF_HOME=${HF_HOME}"
    --env "HF_HUB_CACHE=${HF_HUB_CACHE}"
    --env "HF_HUB_OFFLINE=1"
    --env "TRANSFORMERS_OFFLINE=1"
    --env "VLLM_SERVER_DEV_MODE=1"
    --env "VLLM_HOST_IP=${HEAD_IP}"
    --env "RAY_ADDRESS=${HEAD_IP}:${RAY_PORT}"
)

log "=== start ray head on ${HEAD_NODE} ==="
srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/tp8_${SLURM_JOB_ID}_head.out \
    apptainer "${APPT_COMMON[@]}" "$LIB_SIF_VLLM" \
    bash -c "ray start --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} \
        --dashboard-port=${RAY_DASHBOARD_PORT} --num-gpus=4 --temp-dir=/tmp/ray --block" &
HEAD_PID=$!
log "head srun pid=$HEAD_PID; waiting 45s for head to become reachable"
sleep 45

log "=== start ray worker on ${WORKER_NODE} ==="
srun --nodes=1 --ntasks=1 --nodelist="$WORKER_NODE" --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/tp8_${SLURM_JOB_ID}_worker.out \
    apptainer "${APPT_COMMON[@]}" "$LIB_SIF_VLLM" \
    bash -c "ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=4 --block" &
WORKER_PID=$!
log "worker srun pid=$WORKER_PID; waiting 45s for cluster to settle"
sleep 45

log "=== ray status ==="
srun --nodes=1 --nodelist="$HEAD_NODE" apptainer "${APPT_COMMON[@]}" "$LIB_SIF_VLLM" \
    ray status --address="${HEAD_IP}:${RAY_PORT}" || {
    log "FAIL: ray cluster did not report status; not attempting vLLM launch"
    kill $HEAD_PID $WORKER_PID 2>/dev/null
    exit 2
}

MODEL_PATH="/mnt/tier2/project/p201362/ttahmid/inference-work/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
VLLM_PORT=8100

log "=== launch vLLM on ${HEAD_NODE} with TP=8 ==="
srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/tp8_${SLURM_JOB_ID}_vllm.out \
    apptainer "${APPT_COMMON[@]}" "$LIB_SIF_VLLM" \
    vllm serve "${MODEL_PATH}" \
        --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
        --host 0.0.0.0 --port ${VLLM_PORT} \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 1 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.85 \
        --max-num-seqs 256 \
        --max-model-len 4096 \
        --no-enable-prefix-caching \
        --no-enable-log-requests \
        --seed 0 \
        --distributed-executor-backend ray &
VLLM_PID=$!
log "vLLM srun pid=$VLLM_PID; waiting for /health"

# Give vLLM time to start; TP=8 across two nodes is slow.
for i in $(seq 1 60); do
    sleep 20
    if curl -s -f "http://${HEAD_IP}:${VLLM_PORT}/health" >/dev/null 2>&1; then
        log "vLLM up after ~$((i*20))s"
        break
    fi
    if [[ $i -eq 60 ]]; then
        log "FAIL: vLLM did not become healthy within 20 minutes"
        kill $VLLM_PID $HEAD_PID $WORKER_PID 2>/dev/null
        exit 3
    fi
done

log "=== drive workload against TP=8 endpoint ==="
cd "$HERE"
export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"
"${LIB_VENV}/bin/python" analysis/tp8_driver.py \
    --endpoint "http://${HEAD_IP}:${VLLM_PORT}" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --model-revision "a09a35458c702b33eeacc393d103063234e8bc28" \
    --config configs/tp8_multinode.yaml \
    --slurm-job "${SLURM_JOB_ID}" \
    --head-ip "${HEAD_IP}"
DRIVE_RC=$?
log "driver exit code: ${DRIVE_RC}"

log "=== teardown ==="
kill $VLLM_PID 2>/dev/null || true
sleep 5
srun --nodes=2 apptainer "${APPT_COMMON[@]}" "$LIB_SIF_VLLM" ray stop --force || true
kill $HEAD_PID $WORKER_PID 2>/dev/null || true

exit $DRIVE_RC
