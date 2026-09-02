# Node-hour budget: llm-inference-bench

**Allocations are MONTHLY, not one-off pools**, reset on the 1st. Run `myquota`
before planning any sweep; do not trust a stale number.

**Compute bills `p201466`; storage lives under `p201362`.** The two are
unrelated: filesystem access follows unix group membership, not the Slurm
account. As of 2026-08-27, `p201466` had 113 of 126 GPU node-hours free
against ~44 remaining on `p201362`, so it is the account in use. The intent is
to consume that allocation within the month, so under-sweeping to save hours is
the wrong instinct: unspent hours expire at the reset.

Two billing facts that dominate planning:

- **A `gpu` job bills the whole 4-GPU node**, whatever it uses. A TP=1 sweep
  run one-configuration-per-node wastes 75% of the spend, hence the packed
  variant (`slurm/run_sweep_packed.sh`) and the experiment that measured what
  packing costs (`slurm/measure_packing_interference.sh`). The verdict: packing
  is not measurement-safe for headline numbers.
- **`p201466` has no CPU allocation at all** (`gres/cpun=0`), so every job runs
  on the `gpu` partition, including provisioning and unit tests. They do not
  need a GPU but do need an account that can pay for the node, and gpu nodes
  have 128 cores anyway. Submitting to `cpu` under `p201466` is rejected
  (`DenyOnLimit`).

## Ledger

Every subtotal is `sacct` `ElapsedRaw * NNodes / 3600`, aggregated by job name.
Failures are counted, not hidden: several were the cheapest part of the whole
exercise. Regenerate this ledger with:

```bash
sacct -u u104240 --starttime=2026-08-24 --endtime=2026-10-01 \
  --format=JobID,JobName,Account,Partition,State,ElapsedRaw,NNodes,Start -X -P
```

### August 2026 — billed to p201362 (before the 2026-08-27 account switch)

| Job name         | Part | Jobs | Node-h | Notes |
|---|---|---:|---:|---|
| envprobe         | gpu | 2    | 0.007  | environment probe |
| envprobe2        | gpu | 1    | 0.011  | module/apptainer probe |
| lib-capprobe     | gpu | 5    | 0.288  | capability probe iterations |
| lib-sweep        | gpu | 1    | 0.330  | phase-1 validation sweep |
| **GPU subtotal** | gpu | **9** | **0.636** | |
| cpumodprobe      | cpu | 1    | 0.006  | cpu-node module probe |
| lib-provision    | cpu | 6    | 2.285  | containers + weights + env |
| lib-capprobe     | cpu | 5    | 0.098  | failed: needs a gpu node |
| lib-tests        | cpu | 4    | 0.026  | unit tests + lint |
| **CPU subtotal** | cpu | **16** | **2.416** | |

### August 2026 — billed to p201466 (after the account switch)

| Job name           | Part | Jobs   | Node-h  | Notes |
|---|---|---:|---:|---|
| lib-analyze        | gpu  | 1      | 0.009   | early analysis |
| lib-analysis       | gpu  | 5      | 0.079   | figure regeneration |
| lib-tests          | gpu  | 2      | 0.016   | unit tests + lint on gpu |
| lib-expand         | gpu  | 3      | 0.016   | sweep expansion dry runs |
| lib-expandall      | gpu  | 1      | 0.006   | multi-config expand |
| lib-resetprobe     | gpu  | 1      | 0.007   | narrowed the 404 |
| lib-resetprobe2    | gpu  | 1      | 0.006   | found the dev-mode gate |
| lib-packing        | gpu  | 1      | 2.386   | packing interference measurement |
| lib-sweep          | gpu  | 82     | 93.399  | full-7b + full-32b + phase-2-tp-scaling + specdec + saturation + long-context + length-sweep + memory-utilisation + paged-attention + seed replicate |
| **GPU subtotal**   | gpu  | **97** | **95.924** | |

### September 2026 — billed to p201466 (current)

| Job name           | Part | Jobs | Node-h | Notes |
|---|---|---:|---:|---|
| lib-tests          | gpu  | 1    | 0.011  | 58 tests pass, adds tests/test_plot_phase1.py |
| lib-analysis       | gpu  | 1    | 0.124  | figure regeneration after scoped-integrity fix |
| lib-regress        | gpu  | (queued) | — | baseline capture for full-7b |
| lib-scan-len       | gpu  | (queued) | — | length-sweep summary scan |
| **GPU subtotal so far** | gpu | **2** | **0.135** | plus queued jobs |

## Consumed to date

| Month          | Account  | Partition | Node-hours | Allocation | Remaining |
|---|---|---|---:|---:|---:|
| August 2026    | p201362  | gpu       | 0.636      | (no longer billed here) | — |
| August 2026    | p201362  | cpu       | 2.416      | — | — |
| August 2026    | p201466  | gpu       | **95.924** | 126 | ~30.1 |
| September 2026 | p201466  | gpu       | 0.135      | ~130 (fresh) | ~129.9 |

August finished with roughly **96 of 126** node-hours consumed on `p201466`,
close to the ~100 h estimate carried in the workspace CLAUDE.md. September is
fresh at ~130 h and the sweeps are already in place, so the remaining work is
(a) reruns and analysis, and (b) the TP=8 multi-node attempt.

## Planned September consumption

| Task                                          | Estimate | Comment |
|---|---:|---|
| Regenerate figures / baselines / RESULTS.md   | ~0.5 h   | analysis only, no measurement |
| TP=8 multi-node vLLM (Ray, 2 nodes)           | ~6 h     | one interconnect data point; may fail |
| Reserve for reruns / follow-ups               | ~10 h    | packing revisit, prefix cache eviction |

The rest of the September allocation is not earmarked. Under-spending it costs
nothing but earns nothing; if follow-up questions arise (e.g. saturation curves
at more concurrency points, or measurement noise on the p95 tails), that is
what the headroom is for.
