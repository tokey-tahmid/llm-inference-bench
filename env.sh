# Shared environment for llm-inference-bench on MeluXina.
# Source this from every provision/run script: `source "$(dirname "$0")/../env.sh"`.
#
# NOTE: `module` is not initialised in a Slurm batch environment on MeluXina.
# Scripts that need modules must run their body under `bash -lc`, or source
# /etc/profile.d/modules.sh before calling `module load`.

export LIB_ACCOUNT="p201362"

# Repo root (this file's directory).
LIB_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIB_REPO

# Bulk data lives on project storage, never in the git repo.
export LIB_DATA="/mnt/tier2/project/p201362/ttahmid/inference-work"
export LIB_IMAGES="${LIB_DATA}/images"
export LIB_MODELS="${LIB_DATA}/models"
export LIB_LOGS="${LIB_DATA}/logs"
export LIB_ENV="${LIB_DATA}/env"

# Never rely on $HOME defaults for caches: home is small and shared.
export HF_HOME="${LIB_DATA}/hf_cache"
export HF_HUB_CACHE="${LIB_DATA}/hf_cache/hub"
export VLLM_CACHE_ROOT="${LIB_DATA}/vllm_cache"
export TRITON_CACHE_DIR="${LIB_DATA}/vllm_cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${LIB_DATA}/vllm_cache/inductor"
export OUTLINES_CACHE_DIR="${LIB_DATA}/vllm_cache/outlines"

# Apptainer: cache on Lustre, tmpdir on Lustre too. /dev/shm is faster but its
# usage counts against the job's cgroup memory limit, and a multi-GB OCI->SIF
# conversion will OOM a modestly sized allocation.
export APPTAINER_CACHEDIR="${LIB_DATA}/apptainer_cache"
export APPTAINER_TMPDIR="${LIB_DATA}/apptainer_tmp"

# Modules confirmed present on compute nodes (see ../CLAUDE.md).
#
# LIB_MOD_ENV must be loaded FIRST and is not optional. The EasyBuild module tree
# lives behind the `env/release/*` meta-module, and it is preloaded on `gpu`
# nodes but NOT on `cpu` nodes: MODULEPATH there is missing
# /apps/USE/easybuild/release/2025.1/modules/all, so `module load Apptainer/...`
# fails with the misleading "exist but cannot be loaded as requested". Verified
# in probe job 5143547 on mel0193.
export LIB_MOD_ENV="env/release/2025.1"
export LIB_MOD_APPTAINER="Apptainer/1.4.2-GCCcore-14.2.0"
export LIB_MOD_CUDA="CUDA/12.8.0"

# Harness python env, built by provision.sh stage `env`.
export LIB_VENV="${LIB_ENV}/venv"
export UV_INSTALL_DIR="${LIB_ENV}/uv"
export UV_CACHE_DIR="${LIB_ENV}/uv_cache"
export UV_PYTHON_INSTALL_DIR="${LIB_ENV}/uv_python"

# Container image paths, written by provision.sh stage `images`.
export LIB_SIF_VLLM="${LIB_IMAGES}/vllm.sif"
export LIB_SIF_SGLANG="${LIB_IMAGES}/sglang.sif"

# Provenance lockfile: resolved image digests, backend versions, model revisions.
export LIB_LOCKFILE="${LIB_REPO}/provision.lock.json"

mkdir -p "$LIB_IMAGES" "$LIB_MODELS" "$LIB_LOGS" "$LIB_ENV" \
         "$HF_HUB_CACHE" "$VLLM_CACHE_ROOT" \
         "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" 2>/dev/null || true
