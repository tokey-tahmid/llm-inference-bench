# Measured results

**Generated file. Do not edit.** Regenerate with:

```bash
python analysis/make_results_tables.py
```

Generated 2026-08-31T10:47:01+00:00 from `results/raw/` (1379 artifacts).

## Provenance

- hardware: **4x NVIDIA A100-SXM4-40GB**, driver 595.71.05, CUDA 13.2
- backends: **['sglang', 'vllm']**, versions **['0.27.1', '0.5.18']**
- models: **['Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']**
- git sha: `fc7c2a2c9ad602f22ae45cddf9797cac98b0a636-dirty`

Every row below traces to a committed artifact under `results/raw/`.
Summaries are median with [min, max] across repetitions, never a bare mean.

## What was measured

- artifacts on disk: **1379**
- analysed (warmup excluded): **1137**
- warmup repetitions recorded and excluded: **242**
- status breakdown: `{'ok': 1283, 'launch_failed': 96}`
- **WARNING**: 1027 run(s) from a dirty git tree

### Coverage

| backend | model | TP | prefix caching | phases | runs |
|---|---|---|---|---|---|
| sglang | Qwen2.5-32B-Instruct | 2 | False | 7 | 31 |
| sglang | Qwen2.5-32B-Instruct | 2 | True | 6 | 30 |
| sglang | Qwen2.5-7B-Instruct | 1 | False | 8 | 39 |
| sglang | Qwen2.5-7B-Instruct | 1 | True | 8 | 39 |
| sglang | Qwen2.5-7B-Instruct | 2 | False | 12 | 57 |
| sglang | Qwen2.5-7B-Instruct | 2 | True | 12 | 58 |
| sglang | Qwen2.5-7B-Instruct | 4 | False | 19 | 91 |
| sglang | Qwen2.5-7B-Instruct | 4 | True | 19 | 93 |
| vllm | Qwen2.5-32B-Instruct | 2 | False | 7 | 33 |
| vllm | Qwen2.5-32B-Instruct | 2 | True | 7 | 35 |
| vllm | Qwen2.5-32B-Instruct | 4 | False | 8 | 40 |
| vllm | Qwen2.5-32B-Instruct | 4 | True | 8 | 40 |
| vllm | Qwen2.5-7B-Instruct | 1 | False | 18 | 116 |
| vllm | Qwen2.5-7B-Instruct | 1 | True | 8 | 39 |
| vllm | Qwen2.5-7B-Instruct | 2 | False | 15 | 74 |
| vllm | Qwen2.5-7B-Instruct | 2 | True | 12 | 60 |
| vllm | Qwen2.5-7B-Instruct | 4 | False | 18 | 90 |
| vllm | Qwen2.5-7B-Instruct | 4 | True | 19 | 92 |

## Tensor-parallel scaling

### sglang · Qwen2.5-32B-Instruct · strong scaling at c=64

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 2 | 356.8 [354.9, 357.4] N=6 | n/a | n/a |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

### sglang · Qwen2.5-7B-Instruct · strong scaling at c=128

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 1 | 2375 [2372, 2378] N=5 | 1.00x | 1.00 |
| 2 | 2823 [2813, 2835] N=10 | 1.19x | 0.59 |
| 4 | 6173 [6157, 6194] N=15 | 2.60x | 0.65 |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

### vllm · Qwen2.5-32B-Instruct · strong scaling at c=64

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 2 | 429.1 [428.9, 429.3] N=8 | n/a | n/a |
| 4 | 968.3 [965.2, 974.2] N=10 | n/a | n/a |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

### vllm · Qwen2.5-7B-Instruct · strong scaling at c=128

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 1 | 2184 [2176, 2186] N=8 | 1.00x | 1.00 |
| 2 | 3463 [3459, 3474] N=10 | 1.59x | 0.79 |
| 4 | 5838 [5820, 5854] N=15 | 2.67x | 0.67 |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

## Automatic prefix caching

Measured hit rate is computed as a **per-phase delta** of the engine's
cumulative counters, because the cache-reset endpoint clears the cache without
resetting the counters. The theoretical bound is what this workload could
achieve with no eviction; the gap between them is eviction and block-granularity
rounding.

| backend | shared-prefix ratio | measured hit rate | theoretical bound |
|---|---|---|---|
| sglang | 0 | 0 [0, 0] N=110 | 0 |
| sglang | 0.3 | 0 [0, 0] N=67 | 0.145 |
| sglang | 0.6 | 0 [0, 0] N=40 | 0.292 |
| sglang | 0.9 | 0 [0, 0] N=3 | 0.391 |
| vllm | 0 | 0 [0, 0] N=125 | 0 |
| vllm | 0.3 | 0.145 [0.0938, 0.148] N=69 | 0.145 |
| vllm | 0.6 | 0.294 [0.208, 0.298] N=65 | 0.294 |
| vllm | 0.9 | 0.391 [0.391, 0.391] N=2 | 0.391 |

## Cost of packed placement

Identical server group, same node, same job: once alone with three GPUs idle,
once as four NUMA-pinned replicas with only GPU 0 measured.

| phase | metric | isolated | packed | change |
|---|---|---|---|---|
| closed_c1 | TTFT p50 (s) | 0.07198 [0.07198, 0.09382] N=3 | 0.1805 [0.156, 0.2161] N=3 | **+150.7%** worse |
| closed_c1 | ITL p50 (s/tok) | 0.01221 [0.01221, 0.01221] N=3 | 0.01212 [0.01212, 0.01212] N=3 | **-0.8%** better |
| closed_c1 | output tok/s | 80.1 [79.51, 80.14] N=3 | 76.07 [76.06, 76.1] N=3 | **-5.0%** worse |
| closed_c32 | TTFT p50 (s) | 0.3856 [0.3854, 0.387] N=3 | 0.6261 [0.5994, 0.6272] N=3 | **+62.4%** worse |
| closed_c32 | ITL p50 (s/tok) | 0.0146 [0.0146, 0.01462] N=3 | 0.02168 [0.02167, 0.0217] N=3 | **+48.4%** worse |
| closed_c32 | output tok/s | 1448 [1447, 1453] N=3 | 544 [542.4, 544.3] N=3 | **-62.4%** worse |
| closed_c8 | TTFT p50 (s) | 0.2664 [0.2191, 0.3177] N=3 | 0.407 [0.2856, 0.5099] N=3 | **+52.8%** worse |
| closed_c8 | ITL p50 (s/tok) | 0.01259 [0.01259, 0.0126] N=3 | 0.01458 [0.01458, 0.01461] N=3 | **+15.8%** worse |
| closed_c8 | output tok/s | 550.3 [549.7, 550.3] N=3 | 362.6 [362.2, 362.7] N=3 | **-34.1%** worse |

## Configurations that did not run

Recorded rather than dropped: a configuration that cannot run is a real
boundary of the system, and its absence from a sweep should be visible in
the data rather than only in someone's memory.

| backend | model | TP | status | runs | cause |
|---|---|---|---|---|---|
| sglang | Qwen2.5-32B-Instruct | 4 | `launch_failed` | 80 | sglang exited with code -9 during startup. Log tail: |

