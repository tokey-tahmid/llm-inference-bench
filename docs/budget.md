# Node-hour budget: llm-inference-bench

**The allocation on `p201362` is MONTHLY, not a one-off pool: 117 GPU node-hours
and 23 CPU node-hours per month, reset on the 1st.** The workspace CLAUDE.md
originally implied ~100 GPU node-hours total across all three projects; that was
wrong and is corrected there. Run `myquota` before planning any sweep.

Two billing facts that dominate planning:

- **A `gpu` job bills the whole 4-GPU node**, whatever it uses. A TP=1 sweep run
  one-configuration-per-node wastes 75% of the spend, hence the packed variant.
- **A `cpu` job also bills the whole node** (256 threads), regardless of
  `--cpus-per-task`. Verified from `AllocTRES` on jobs `5143559` and `5146938`,
  both of which requested 32 or 16 CPUs and were allocated `cpu=256`. There is no
  point requesting fewer cores on this partition.

## Ledger

Every entry traces to a Slurm job ID. Elapsed times are from `sacct`.

### GPU partition

| Job | Purpose | Elapsed | Node-hours | Outcome |
|---|---|---|---|---|
| 5143240 | environment probe (output lost to node-local `/tmp`) | 00:00:10 | 0.003 | failed usefully |
| 5143256 | environment probe: GPUs, topology, network, disk | 00:00:14 | 0.004 | complete |
| 5143260 | module/apptainer probe | 00:00:41 | 0.011 | complete |
| **GPU subtotal** | | **00:01:05** | **0.018** | |

### CPU partition

| Job | Purpose | Elapsed | Node-hours | Outcome |
|---|---|---|---|---|
| 5143500 | provision (failed: `set -u` vs module init) | 00:00:07 | 0.002 | failed |
| 5143535 | provision (failed: `BASH_SOURCE` in psslurm spool) | 00:00:09 | 0.003 | failed |
| 5143542 | provision (failed: `env/release` not loaded on cpu nodes) | 00:00:22 | 0.006 | partial, `env` stage done |
| 5143547 | cpu-node module probe | 00:00:20 | 0.006 | complete, root-caused the above |
| 5143559 | provision: images + models | 02:16:00 | 2.267 | complete |
| 5146585 | provision re-run (all stages already marked) | 00:00:11 | 0.003 | no-op |
| 5146938 | capability probe (failed: read-only Lustre in container) | 00:02:27 | 0.041 | failed usefully |
| **CPU subtotal** | | **02:19:36** | **2.328** | |

## Consumed to date

| | Used | Monthly allocation |
|---|---|---|
| GPU node-hours | **0.018** | 117 |
| CPU node-hours | **2.328** | 23 |

Essentially the entire cost so far has been provisioning on the CPU partition
(container pulls plus ~77 GB of weights), which does not touch the GPU budget at
all. **No GPU time has been spent on measurement yet, and no measurements exist.**

## Planned

| Phase | Estimate | Notes |
|---|---|---|
| Capability probe (re-run) | ~0.1 CPU node-hours | gate before any GPU spend |
| Phase 1 validation sweep | ~0.5 GPU node-hours | 1 group x 2 phases x 4 reps, 7B at TP=1 |
| Phase 2 (widen) | TBD | not planned until phase 1 produces a figure |

August has ~45 GPU node-hours remaining as of 2026-08-25 (72 of 117 already
consumed by prior unrelated work). The full P1 matrix is budgeted at ~45
node-hours, so it spans the September reset; sequence accordingly rather than
racing the month end.
