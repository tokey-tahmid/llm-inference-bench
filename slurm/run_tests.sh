#!/bin/bash
# Unit tests and lint, on the cpu partition.
#
# No GPU is needed: everything tested here is pure logic (workload generation,
# metric computation, sweep expansion, provenance enforcement). The parts that
# need a GPU are validated by the phase-1 sweep instead.
#
# The login node has python 3.6, so this is the only place this repo can be
# tested at all.
#
#   sbatch slurm/run_tests.sh
#
#SBATCH --job-name=lib-tests
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/tests_%j.out

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

cd "$HERE"
export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"

echo "=== ruff ==="
"${LIB_VENV}/bin/ruff" check src tests slurm || RUFF_FAILED=1
"${LIB_VENV}/bin/ruff" format --check src tests || true

echo
echo "=== pytest ==="
"${LIB_VENV}/bin/python" -m pytest -q --tb=short

echo
echo "=== identifier verification (guards rule 1 in prose) ==="
"${LIB_VENV}/bin/python" analysis/verify_identifiers.py || IDENT_FAILED=1

echo
echo "=== sweep expansion dry run (no GPU, no cost) ==="
"${LIB_VENV}/bin/python" -m llm_inference_bench.cli expand configs/phase1_validation.yaml

echo
echo "TESTS COMPLETE${RUFF_FAILED:+ (ruff issues)}${IDENT_FAILED:+ (UNVERIFIED IDENTIFIERS)}"
[ -n "${IDENT_FAILED:-}" ] && exit 1
exit 0
