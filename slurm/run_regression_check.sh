#!/bin/bash
# Capture or verify the P1 performance regression baseline.
#
#   sbatch slurm/run_regression_check.sh --sweep-name full-7b --update-baseline
#   sbatch slurm/run_regression_check.sh --sweep-name full-7b
#
# No GPU is needed: the check reads results/raw/ and produces a comparison.
# Runs on the gpu partition because p201466 has no CPU allocation.
#
#SBATCH --job-name=lib-regress
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/regress_%j.out

set -uo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -u
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

cd "$HERE"
export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"

"${LIB_VENV}/bin/python" analysis/regression_check.py "$@"
