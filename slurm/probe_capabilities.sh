#!/bin/bash
# Reconcile the backend adapters against the actual pinned container images.
#
# This is a GATE, not a diagnostic: it runs before any GPU time is spent. The
# adapters encode flag names for vLLM and SGLang, and a flag that was renamed
# between releases would otherwise surface as a wall of identical launch_failed
# artifacts halfway through a job array, after the allocation is already gone.
#
# Produces, under docs/:
#   help_<backend>.txt          verbatim --help from the pinned image
#   capabilities_<backend>.json version, and which declared flags actually exist
#
# Runs on the cpu partition: no GPU is needed to print a help text.
#
#SBATCH --job-name=lib-capprobe
#SBATCH --account=p201362
#SBATCH --partition=cpu
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/capprobe_%j.out

set -euo pipefail

# See ../CLAUDE.md: module is uninitialised in batch and the system init is not
# strict-mode clean.
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

module load "$LIB_MOD_ENV"
module load "$LIB_MOD_APPTAINER"

DOCS="${HERE}/docs"
mkdir -p "$DOCS"

log() { printf '[capprobe %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Fail fast and cheap. The login node has python 3.6, so nothing in this repo can
# be syntax-checked before submission; do it here, in seconds, before spending
# ~20 minutes importing torch twice to dump help text.
log "compile-checking the harness (login node python is 3.6, so this is the first chance)"
PYTHONPATH="${HERE}/src" "${LIB_VENV}/bin/python" -m compileall -q "${HERE}/src" "${HERE}/slurm" \
    || { log "FATAL: harness does not compile"; exit 1; }
PYTHONPATH="${HERE}/src" "${LIB_VENV}/bin/python" -c '
from llm_inference_bench.backends import available_backends, get_adapter
from pathlib import Path
print("[capprobe] backends registered:", available_backends())
for b in available_backends():
    a = get_adapter(b, sif_path=Path("unused"))
    print(f"[capprobe]   {b}: {len(a.capabilities)} declared capabilities")
' || { log "FATAL: harness does not import"; exit 1; }

# Run a command inside an image using EXACTLY the invocation the runner uses.
# This must mirror BackendAdapter.container_argv, or the probe validates an
# environment the sweep will never actually run in.
#
# The read-only Lustre trap (job 5146938): torch resolves and creates an inductor
# cache directory at import time, so a cache env var pointing at /mnt/tier2 makes
# `import vllm` fail with Errno 30 before argparse runs. Write caches go to the
# node's tmpfs; only the read-only weight cache points at project storage.
apptainer_exec() {
    local sif="$1"; shift
    apptainer exec --cleanenv \
        --bind "${LIB_DATA}:${LIB_DATA}:rw" \
        --env "HF_HOME=${HF_HOME}" \
        --env "HF_HUB_CACHE=${HF_HUB_CACHE}" \
        --env "TMPDIR=/tmp" \
        --env "HOME=/tmp" \
        --env "VLLM_CACHE_ROOT=/tmp/vllm_cache" \
        --env "TRITON_CACHE_DIR=/tmp/triton_cache" \
        --env "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache" \
        --env "TORCH_HOME=/tmp/torch_home" \
        --env "XDG_CACHE_HOME=/tmp/xdg_cache" \
        --env "HF_HUB_OFFLINE=1" \
        --env "TRANSFORMERS_OFFLINE=1" \
        "$sif" "$@"
}

# Confirm the mount situation rather than inferring it from a downstream failure.
diagnose_mounts() {
    local name="$1" sif="$2"
    log "${name}: mount and writability diagnostic"
    apptainer_exec "$sif" sh -c '
        echo "  whoami: $(id -un) ($(id -u))"
        for d in /tmp /dev/shm "$HF_HUB_CACHE" "$TORCHINDUCTOR_CACHE_DIR"; do
            if [ -z "$d" ]; then continue; fi
            mkdir -p "$d" 2>/dev/null
            if touch "$d/.wtest" 2>/dev/null; then
                echo "  WRITABLE   $d"; rm -f "$d/.wtest"
            else
                echo "  READ-ONLY  $d"
            fi
        done
        echo "  tier2 mount: $(grep -m1 tier2 /proc/mounts 2>/dev/null || echo "not mounted")"
    ' 2>&1 | tee "${DOCS}/mounts_${name}.txt" || log "${name}: mount diagnostic failed"
}

# Rather than guess where the interpreter lives, try the plausible locations in
# order and record which one worked.
find_python() {
    local sif="$1" candidate
    for candidate in python3 python /usr/bin/python3 /opt/venv/bin/python \
                     /usr/local/bin/python3 /opt/conda/bin/python; do
        if apptainer_exec "$sif" "$candidate" -c 'import sys' >/dev/null 2>&1; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

probe_backend() {
    local name="$1" sif="$2" import_name="$3"
    shift 3
    local help_cmd=("$@")

    log "=== ${name}: ${sif} ==="
    [[ -f "$sif" ]] || { log "${name}: SIF missing, skipping"; return; }

    diagnose_mounts "$name" "$sif"

    local py
    if ! py="$(find_python "$sif")"; then
        log "${name}: FAILED to find a working python inside the image"
        apptainer_exec "$sif" sh -c 'echo PATH=$PATH; ls /usr/bin/python* /usr/local/bin/python* 2>&1' \
            | tee "${DOCS}/pythonprobe_${name}.txt" || true
        return
    fi
    log "${name}: interpreter = ${py}"

    local version
    version="$(apptainer_exec "$sif" "$py" -c \
        "import ${import_name}; print(${import_name}.__version__)" 2>&1 | tail -1)" || version="UNKNOWN"
    log "${name}: version = ${version}"

    log "${name}: dumping --help (this imports torch, expect it to be slow)"
    if apptainer_exec "$sif" "${help_cmd[@]}" > "${DOCS}/help_${name}.txt" 2>&1; then
        log "${name}: help captured, $(wc -l < "${DOCS}/help_${name}.txt") lines"
    else
        log "${name}: WARNING help command exited nonzero; captured output anyway"
    fi

    # A help dump that is actually a traceback would make every flag look absent
    # and produce a page of bogus OVERCLAIM findings. Catch that here rather than
    # letting it masquerade as a capability result.
    if grep -q '^Traceback' "${DOCS}/help_${name}.txt"; then
        log "${name}: ERROR help capture is a traceback, not help text. Root cause:"
        tail -3 "${DOCS}/help_${name}.txt" | sed 's/^/[capprobe]     /'
        log "${name}: reconciliation for this backend would be meaningless; see the file"
    fi

    printf '%s\n' "$version" > "${DOCS}/version_${name}.txt"
    printf '%s\n' "$py" > "${DOCS}/python_${name}.txt"
}

probe_backend vllm "$LIB_SIF_VLLM" vllm \
    vllm serve --help

probe_backend sglang "$LIB_SIF_SGLANG" sglang \
    python3 -m sglang.launch_server --help

# --- reconcile the adapters against what the images actually accept ----------
log "reconciling adapter flags against captured help text"
PYTHONPATH="${HERE}/src" "${LIB_VENV}/bin/python" "${HERE}/slurm/reconcile_capabilities.py" \
    --docs "$DOCS" --lockfile "$LIB_LOCKFILE"

log "CAPABILITY PROBE COMPLETE"
