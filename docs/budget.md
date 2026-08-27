# Node-hour budget: llm-inference-bench

**Allocations are MONTHLY, not one-off pools**, reset on the 1st. Run `myquota`
before planning any sweep; do not trust a stale number.

**Compute bills `p201466`; storage lives under `p201362`.** The two are
unrelated: filesystem access follows unix group membership, not the Slurm
account. As of 2026-08-27, `p201466` had **113 of 126 GPU node-hours free**
against ~44 remaining on `p201362`, so it is the account in use. The intent is to
consume that allocation within the month, so under-sweeping to save hours is the
wrong instinct: unspent hours expire at the reset.

Two billing facts that dominate planning:

- **A `gpu` job bills the whole 4-GPU node**, whatever it uses. A TP=1 sweep run
  one-configuration-per-node wastes 75% of the spend, hence the packed variant
  (`slurm/run_sweep_packed.sh`) and the experiment that measures what packing
  costs (`slurm/measure_packing_interference.sh`).
- **`p201466` has no CPU allocation at all** (`gres/cpun=0`), so every job runs
  on the `gpu` partition, including provisioning and unit tests. They do not need
  a GPU but do need an account that can pay for the node, and gpu nodes have 128
  cores anyway. Submitting to `cpu` under `p201466` is rejected (`DenyOnLimit`).

## Ledger

Every entry traces to a Slurm job ID; elapsed times are from `sacct`. Failures
are listed, not hidden: several were the cheapest part of the whole exercise.

### Billed to p201362 (before the 2026-08-27 switch)

| Job | Purpose | Part | Elapsed | Node-h | Outcome |
|---|---|---|---|---|---|
| 5143240 | environment probe | gpu | 00:00:10 | 0.003 | output lost to node-local `/tmp` |
| 5143256 | environment probe: GPUs, topology, network | gpu | 00:00:14 | 0.004 | complete |
| 5143260 | module/apptainer probe | gpu | 00:00:41 | 0.011 | complete |
| 5147074 | capability probe | gpu | 00:00:00 | 0.000 | cancelled while queued |
| 5147155 | capability probe | gpu | 00:03:59 | 0.066 | found read-only-Lustre cause |
| 5147176 | capability probe | gpu | 00:03:52 | 0.064 | found `--help` pagination |
| 5148047 | capability probe | gpu | 00:03:34 | 0.059 | **both backends verified** |
| 5148060 | phase-1 validation sweep | gpu | 00:19:47 | 0.330 | 8 artifacts, superseded |
| **GPU subtotal** | | | **00:31:57** | **0.537** | |
| 5143500/35/42 | provision (3 script bugs) | cpu | 00:00:38 | 0.011 | failed, each root-caused |
| 5143547 | cpu-node module probe | cpu | 00:00:20 | 0.006 | complete |
| 5143559 | provision: images + models | cpu | 02:16:00 | 2.267 | complete |
| 5146585/6937 | provision re-runs | cpu | 00:00:28 | 0.008 | idempotent no-ops |
| 5146938/7030 | capability probe | cpu | 00:05:53 | 0.098 | failed: needs a GPU node |
| 5147029/073/136/165 | unit tests + lint | cpu | 00:01:32 | 0.026 | 43 tests, lint clean |
| **CPU subtotal** | | | **02:24:51** | **2.416** | |

### Billed to p201466 (current)

| Job | Purpose | Part | Elapsed | Node-h | Outcome |
|---|---|---|---|---|---|
| 5150722 | phase-1 analysis | gpu | 00:00:34 | 0.009 | figure + integrity report |
| 5150729 | vLLM reset-endpoint probe | gpu | 00:00:26 | 0.007 | narrowed the 404 |
| 5150753 | vLLM reset-endpoint probe | gpu | 00:00:22 | 0.006 | found the dev-mode gate |
| 5150758 | phase-1 re-run, clean tree | gpu | in progress | ~0.33 | |
| **GPU subtotal** | | | | **~0.35** | |

## Consumed to date

| | p201362 | p201466 |
|---|---|---|
| GPU node-hours | 0.537 | ~0.35 |
| CPU node-hours | 2.416 | n/a (no allocation) |

Almost the entire cost so far was provisioning (container pulls plus ~77 GB of
weights) on the CPU partition, which never touched the GPU allocation. **Under
0.9 GPU node-hours have been spent in total**, and the P1 budget is ~45.

## Planned

| Phase | Estimate | Gate |
|---|---|---|
| Packing interference | ~1.5 node-h | decides whether packed runs can carry headline numbers |
| Phase 2: TP scaling {1,2,4}, 7B | ~6 node-h | after packing delta is known |
| Phase 2: prefix caching on/off x ratio | ~5 node-h | |
| Phase 2: backend comparison, matched | ~4 node-h | needs the `--mem-fraction-static` caveat settled |
| Phase 3: 32B at TP {2,4} | ~8 node-h | |
| Phase 3: TP=8 multi-node | ~4 node-h | 2 nodes + Ray |
| Phase 3: speculative decoding | ~6 node-h | |

Roughly 35 node-hours of measurement against 113 available this month, leaving
real headroom for reruns. Nothing above is measured yet.
