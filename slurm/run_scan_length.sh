#!/bin/bash
#SBATCH --job-name=lib-scan-len
#SBATCH --account=p201466
#SBATCH --partition=gpu
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/tier2/project/p201362/ttahmid/inference-work/logs/scanlen_%j.out
set -uo pipefail
if ! type module >/dev/null 2>&1; then set +eu; source /etc/profile; set -u; fi
HERE=/home/users/u104240/apps/inference-work/llm-inference-bench
source "$HERE/env.sh"; cd "$HERE"
export PYTHONPATH="$HERE/src:${PYTHONPATH:-}"
"${LIB_VENV}/bin/python" analysis/scan_length_sweep.py
