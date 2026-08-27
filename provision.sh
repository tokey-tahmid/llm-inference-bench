#!/bin/bash
# Provision llm-inference-bench: containers, harness python env, model weights.
#
# On MeluXina the usual HPC split is INVERTED: the login node has no `module`,
# no `apptainer`, no `uv` and no `cargo`, while compute nodes have all of those
# AND outbound internet. So provisioning runs as a `cpu`-partition batch job.
#
#   sbatch provision.sh              # all stages
#   sbatch provision.sh env images   # selected stages
#
# Stages are idempotent: each records a marker under $LIB_ENV/.provisioned/ and
# is skipped on re-run unless FORCE=1.
#
#SBATCH --job-name=lib-provision
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/provision_%j.out

set -euo pipefail

# `module` is not a function in a Slurm batch environment on MeluXina. Sourcing
# /etc/profile reproduces exactly what `bash -l` does (the path verified working
# in probe job 5143260), which matters because the profile.d scripts must run in
# order: 00-modulepath.sh sets MODULEPATH before modules.sh defines `module`.
#
# Strict mode must be off across this bootstrap. The system scripts are neither
# `set -u` clean (00-modulepath.sh expands $MODULEPATH before defining it) nor
# `set -e` clean, and under strict mode the job aborts before it starts.
if ! type module >/dev/null 2>&1; then
    set +eu
    # shellcheck disable=SC1091
    source /etc/profile
    set -eu
fi
type module >/dev/null 2>&1 || { echo "[provision ERROR] module never initialised" >&2; exit 1; }

# Resolving the repo root is not as simple as dirname $BASH_SOURCE here: Slurm
# copies the batch script to /var/spool/parastation/jobs/<jobid> before running
# it, so BASH_SOURCE points at the spool copy and every sibling path resolves to
# nothing. Prefer the submit directory, which sbatch preserves.
find_repo() {
    local c
    for c in "${LIB_REPO:-}" "${SLURM_SUBMIT_DIR:-}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)" \
             "/home/users/u104240/apps/inference-work/llm-inference-bench"; do
        if [[ -n "$c" && -f "${c}/env.sh" ]]; then printf '%s' "$c"; return 0; fi
    done
    return 1
}
HERE="$(find_repo)" || { echo "[provision ERROR] cannot locate repo root (env.sh)" >&2; exit 1; }
# shellcheck disable=SC1091
source "${HERE}/env.sh"

MARKERS="${LIB_ENV}/.provisioned"
mkdir -p "$MARKERS"
FORCE="${FORCE:-0}"

log()  { printf '[provision %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die()  { printf '[provision ERROR] %s\n' "$*" >&2; exit 1; }

done_marker() { [[ "$FORCE" != "1" && -f "${MARKERS}/$1" ]]; }
mark_done()   { date -u +%Y-%m-%dT%H:%M:%SZ > "${MARKERS}/$1"; }

# --- provenance lockfile helpers -------------------------------------------
# The lockfile is the single source of truth for "which exact artifact did we
# measure". Every raw result records the values written here.
lock_set() {
    local key="$1" val="$2"
    python3 - "$LIB_LOCKFILE" "$key" "$val" <<'PY'
import json, os, sys
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
if os.path.exists(path):
    with open(path) as fh:
        data = json.load(fh)
node = data
parts = key.split(".")
for p in parts[:-1]:
    node = node.setdefault(p, {})
node[parts[-1]] = val
tmp = path + ".tmp"
with open(tmp, "w") as fh:
    json.dump(data, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, path)
PY
}

lock_get() {
    python3 - "$LIB_LOCKFILE" "$1" <<'PY'
import json, os, sys
path, key = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    sys.exit(0)
with open(path) as fh:
    node = json.load(fh)
for p in key.split("."):
    if not isinstance(node, dict) or p not in node:
        sys.exit(0)
    node = node[p]
print(node)
PY
}

# Resolve a Docker Hub tag to an immutable manifest digest, so that every
# subsequent pull is byte-identical and the SIF is traceable to a registry
# artifact rather than to a mutable :latest tag.
resolve_digest() {
    local repo="$1" tag="$2" token manifest_accept digest
    token="$(curl -fsS "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" \
             | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')"
    manifest_accept='application/vnd.oci.image.index.v1+json,application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json'
    digest="$(curl -fsSI -H "Authorization: Bearer ${token}" -H "Accept: ${manifest_accept}" \
              "https://registry-1.docker.io/v2/${repo}/manifests/${tag}" \
              | tr -d '\r' | awk 'tolower($1)=="docker-content-digest:"{print $2}')"
    [[ -n "$digest" ]] || die "could not resolve digest for ${repo}:${tag}"
    printf '%s' "$digest"
}

# --- stage: env -------------------------------------------------------------
stage_env() {
    if done_marker env; then log "stage env: already provisioned, skipping"; return; fi
    log "stage env: installing uv into ${UV_INSTALL_DIR}"
    mkdir -p "$UV_INSTALL_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
    if [[ ! -x "${UV_INSTALL_DIR}/uv" ]]; then
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$UV_INSTALL_DIR" INSTALLER_NO_MODIFY_PATH=1 sh
    fi
    export PATH="${UV_INSTALL_DIR}:${PATH}"
    uv --version

    log "stage env: creating venv at ${LIB_VENV}"
    uv venv --python 3.12 "$LIB_VENV"
    # `uv pip` needs to be told which interpreter to target when not activated.
    VIRTUAL_ENV="$LIB_VENV" uv pip install --python "${LIB_VENV}/bin/python" \
        "httpx[http2]>=0.27" \
        "numpy>=1.26" \
        "pyyaml>=6.0" \
        "pydantic>=2.7" \
        "transformers>=4.44" \
        "huggingface_hub>=0.24" \
        "hf_transfer>=0.1" \
        "pandas>=2.2" \
        "matplotlib>=3.9" \
        "structlog>=24.1" \
        "rich>=13.7" \
        "pytest>=8.2" \
        "ruff>=0.5"

    lock_set "harness.python" "$("${LIB_VENV}/bin/python" -c 'import platform; print(platform.python_version())')"
    lock_set "harness.uv" "$(uv --version | awk '{print $2}')"
    mark_done env
    log "stage env: done"
}

# --- stage: images ----------------------------------------------------------
pull_image() {
    local name="$1" repo="$2" tag="$3" sif="$4" version_cmd="$5"
    local digest
    digest="$(lock_get "images.${name}.digest")"
    if [[ -z "$digest" ]]; then
        log "resolving ${repo}:${tag} -> digest"
        digest="$(resolve_digest "$repo" "$tag")"
        lock_set "images.${name}.repo" "$repo"
        lock_set "images.${name}.tag_at_pull" "$tag"
        lock_set "images.${name}.digest" "$digest"
        lock_set "images.${name}.pulled_utc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    fi
    log "${name}: pinned to ${repo}@${digest}"

    if [[ ! -f "$sif" ]]; then
        log "${name}: pulling into ${sif} (this is tens of GB of layers, be patient)"
        apptainer pull "$sif" "docker://${repo}@${digest}"
    else
        log "${name}: ${sif} already present, skipping pull"
    fi

    # Record the backend version as reported from *inside* the image. This is
    # the number that goes into every raw result; the tag is not trustworthy.
    local ver
    if ver="$(apptainer exec "$sif" python3 -c "$version_cmd" 2>/dev/null)"; then
        lock_set "images.${name}.version" "$ver"
        log "${name}: version ${ver}"
    else
        log "${name}: WARNING could not read version from inside image"
    fi
    lock_set "images.${name}.sif" "$sif"
    lock_set "images.${name}.sif_bytes" "$(stat -c %s "$sif")"
}

stage_images() {
    if done_marker images; then log "stage images: already provisioned, skipping"; return; fi
    # env/release must come first; see the note in env.sh.
    module load "$LIB_MOD_ENV"
    module load "$LIB_MOD_APPTAINER"
    command -v apptainer >/dev/null || die "apptainer not on PATH after module load"
    apptainer --version

    pull_image vllm   "vllm/vllm-openai" "latest" "$LIB_SIF_VLLM" \
        'import vllm; print(vllm.__version__)'
    pull_image sglang "lmsysorg/sglang"  "latest" "$LIB_SIF_SGLANG" \
        'import sglang; print(sglang.__version__)'

    mark_done images
    log "stage images: done"
}

# --- stage: models ----------------------------------------------------------
# Only ungated models: there is no HuggingFace token on this system and the
# Llama repos are gated. See ../CLAUDE.md "Settled model decisions".
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-32B-Instruct"
)

stage_models() {
    if done_marker models; then log "stage models: already provisioned, skipping"; return; fi
    [[ -x "${LIB_VENV}/bin/python" ]] || die "stage models needs stage env first"

    # hf_transfer is a download accelerator, not a requirement. Enable it only if
    # it actually imports: huggingface_hub raises on startup when the flag is set
    # without the package, which would fail the stage for a pure optimisation.
    # (Note: huggingface_hub 1.x dropped the `[hf_transfer]` extra, so it is now
    # installed as a separate package and older envs will not have it.)
    if "${LIB_VENV}/bin/python" -c "import hf_transfer" 2>/dev/null; then
        export HF_HUB_ENABLE_HF_TRANSFER=1
        log "stage models: hf_transfer enabled"
    else
        log "stage models: hf_transfer unavailable, using the standard downloader"
    fi

    for repo in "${MODELS[@]}"; do
        log "downloading ${repo} -> ${HF_HUB_CACHE}"
        "${LIB_VENV}/bin/python" - "$repo" <<'PY'
import json, os, sys
from huggingface_hub import snapshot_download, HfApi

repo = sys.argv[1]
# Pin the revision: record the exact commit sha we downloaded, never "main".
sha = HfApi().model_info(repo).sha
path = snapshot_download(
    repo_id=repo,
    revision=sha,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.py"],
    max_workers=8,
)
print(json.dumps({"repo": repo, "revision": sha, "path": path}))
PY
    done

    # Record revisions in the lockfile.
    for repo in "${MODELS[@]}"; do
        # lock_set treats '.' as a nesting separator, so a model name like
        # Qwen2.5-7B-Instruct would otherwise be split into bogus sub-objects.
        key="models.$(echo "$repo" | tr './' '__')"
        sha="$("${LIB_VENV}/bin/python" -c \
            "from huggingface_hub import HfApi; print(HfApi().model_info('${repo}').sha)")"
        lock_set "${key}.repo" "$repo"
        lock_set "${key}.revision" "$sha"
    done

    mark_done models
    log "stage models: done"
}

# --- stage: harness ---------------------------------------------------------
# Editable install of this package into the harness venv, so batch scripts can
# `import llm_inference_bench` without every one of them setting PYTHONPATH.
stage_harness() {
    [[ -x "${LIB_VENV}/bin/python" ]] || die "stage harness needs stage env first"
    export PATH="${UV_INSTALL_DIR}:${PATH}"
    log "stage harness: editable install of ${HERE}"
    VIRTUAL_ENV="$LIB_VENV" uv pip install --python "${LIB_VENV}/bin/python" \
        --no-deps --editable "$HERE"
    "${LIB_VENV}/bin/python" -c "import llm_inference_bench, sys; print('harness import OK', sys.version)"
    mark_done harness
    log "stage harness: done"
}

# --- driver -----------------------------------------------------------------
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then STAGES=(env images models harness); fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none} stages=${STAGES[*]}"
log "data root: ${LIB_DATA}"
df -h /mnt/tier2 | tail -1

for s in "${STAGES[@]}"; do
    case "$s" in
        env)     stage_env ;;
        images)  stage_images ;;
        models)  stage_models ;;
        harness) stage_harness ;;
        *)       die "unknown stage: $s" ;;
    esac
done

log "lockfile contents:"
cat "$LIB_LOCKFILE"
log "ALL REQUESTED STAGES COMPLETE"
