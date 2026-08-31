#!/bin/bash
# Regenerate every figure and table from results/raw/.
#
#   sbatch slurm/run_analysis.sh
#
# This is THE reproduce command required by ../CLAUDE.md rule 4. It reads only
# results/raw/ and writes only results/figures/. It never touches raw artifacts
# and never invents a value: an analysis that has no data to plot says so and
# exits non-zero rather than producing an empty figure.
#
# Exit codes: 0 clean, 1 nothing to plot, 2 plotted but integrity problems found
# (which are also stamped onto the figure itself, so a figure with a known
# caveat cannot be separated from its caveat by copying the PNG somewhere).
#
# Runs on the gpu partition because p201466 has no CPU allocation; no GPU is
# actually used.
#
#SBATCH --job-name=lib-analysis
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/analysis_%j.out

set -uo pipefail

if ! type module >/dev/null 2>&1; then
    set +eu
    source /etc/profile
    set -u
fi
# Explicitly NOT errexit. Each analysis script reports its own status through an
# exit code, and "no data for this figure yet" (1) is an expected, non-fatal
# outcome that must not stop the remaining figures from being regenerated.
# Note the trap: the module bootstrap above restores `set -u` only. Restoring
# `set -eu` there would silently re-enable errexit despite the shebang block
# asking for `set -uo pipefail`, and the first script with no data would kill the
# whole run (which is exactly what job 5150858 did).
set +e

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

log() { printf '[analysis %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# Each script is independent: one having no data yet must not stop the others.
SCRIPTS=(
    analysis/plot_phase1.py
    analysis/plot_scaling.py
    analysis/plot_pareto_and_cache.py
    analysis/plot_packing_interference.py
    analysis/make_results_tables.py
)

WORST=0
for s in "${SCRIPTS[@]}"; do
    [[ -f "$s" ]] || continue
    log "=== $s ==="
    "${LIB_VENV}/bin/python" "$s"
    rc=$?
    case "$rc" in
        0) log "$s: clean" ;;
        1) log "$s: no data yet, skipped" ;;
        2) log "$s: PLOTTED WITH INTEGRITY WARNINGS"; WORST=2 ;;
        *) log "$s: FAILED (exit $rc)"; WORST=3 ;;
    esac
    echo
done

log "figures in ${HERE}/results/figures:"
ls -la "${HERE}/results/figures" 2>/dev/null || true
log "ANALYSIS COMPLETE (worst status ${WORST})"
exit "$WORST"
