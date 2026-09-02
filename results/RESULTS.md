# Measured results

**Generated file. Do not edit.** Regenerate with:

```bash
python analysis/make_results_tables.py
```

Generated 2026-09-02T13:57:23+00:00 from `results/raw/` (4573 artifacts).

## Provenance

- hardware: **4x NVIDIA A100-SXM4-40GB**, driver 595.71.05, CUDA 13.2
- backends: **['sglang', 'vllm']**, versions **['0.27.1', '0.5.18']**
- models: **['Qwen/Qwen2.5-32B-Instruct', 'Qwen/Qwen2.5-7B-Instruct']**
- git sha: `fc7c2a2c9ad602f22ae45cddf9797cac98b0a636-dirty`

Every row below traces to a committed artifact under `results/raw/`.
Summaries are median with [min, max] across repetitions, never a bare mean.

## What was measured

- artifacts on disk: **4573**
- analysed (warmup excluded): **3615**
- warmup repetitions recorded and excluded: **958**
- status breakdown: `{'ok': 4383, 'launch_failed': 104, 'failed': 86}`
- **WARNING**: 2267 run(s) from a dirty git tree

### Coverage

| backend | model | TP | prefix caching | phases | runs |
|---|---|---|---|---|---|
| sglang | Qwen2.5-32B-Instruct | 2 | False | 8 | 40 |
| sglang | Qwen2.5-32B-Instruct | 2 | True | 8 | 40 |
| sglang | Qwen2.5-7B-Instruct | 1 | False | 57 | 368 |
| sglang | Qwen2.5-7B-Instruct | 1 | True | 24 | 192 |
| sglang | Qwen2.5-7B-Instruct | 2 | False | 49 | 270 |
| sglang | Qwen2.5-7B-Instruct | 2 | True | 24 | 192 |
| sglang | Qwen2.5-7B-Instruct | 4 | False | 48 | 264 |
| sglang | Qwen2.5-7B-Instruct | 4 | True | 24 | 192 |
| vllm | Qwen2.5-32B-Instruct | 2 | False | 8 | 40 |
| vllm | Qwen2.5-32B-Instruct | 2 | True | 8 | 40 |
| vllm | Qwen2.5-32B-Instruct | 4 | False | 8 | 40 |
| vllm | Qwen2.5-32B-Instruct | 4 | True | 8 | 40 |
| vllm | Qwen2.5-7B-Instruct | 1 | False | 66 | 519 |
| vllm | Qwen2.5-7B-Instruct | 1 | True | 24 | 192 |
| vllm | Qwen2.5-7B-Instruct | 2 | False | 59 | 350 |
| vllm | Qwen2.5-7B-Instruct | 2 | True | 24 | 192 |
| vllm | Qwen2.5-7B-Instruct | 4 | False | 54 | 300 |
| vllm | Qwen2.5-7B-Instruct | 4 | True | 24 | 192 |

## Tensor-parallel scaling

### sglang · Qwen2.5-32B-Instruct · strong scaling at c=64

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 2 | 356.8 [353.4, 357.4] N=10 | n/a | n/a |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

### sglang · Qwen2.5-7B-Instruct · strong scaling at c=128

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 1 | 2373 [2358, 2422] N=50 | 1.00x | 1.00 |
| 2 | 2821 [2807, 2835] N=32 | 1.19x | 0.59 |
| 4 | 6173 [6142, 6203] N=32 | 2.60x | 0.65 |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

### vllm · Qwen2.5-32B-Instruct · strong scaling at c=64

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 2 | 429.1 [428.7, 429.4] N=10 | n/a | n/a |
| 4 | 968.3 [965.2, 974.2] N=10 | n/a | n/a |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

### vllm · Qwen2.5-7B-Instruct · strong scaling at c=1024

| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |
|---|---|---|---|
| 1 | 1787 [1778, 1792] N=6 | 1.00x | 1.00 |
| 2 | 2789 [2785, 2792] N=6 | 1.56x | 0.78 |
| 4 | 4916 [4907, 4932] N=6 | 2.75x | 0.69 |

Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node
PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology
effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler
overhead.

## Prefill vs decode

One server per (backend, TP), six length combinations per server as phases.
Short input / long output isolates decode (memory-bandwidth bound at low
concurrency); long input / short output isolates prefill (compute bound).

### sglang · TP=1 · Qwen2.5-7B-Instruct

| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |
|---|---|---|---|---|---|
| 1 | 128 | 512 | 0.02658 [0.02555, 0.02719] N=3 | 12 [12, 12] N=3 | 83.2 [83.15, 83.22] N=3 |
| 1 | 512 | 512 | 0.0459 [0.04532, 0.04633] N=3 | 12 [12, 12] N=3 | 82.82 [82.81, 82.84] N=3 |
| 1 | 1024 | 256 | 0.07791 [0.07761, 0.07812] N=3 | 12 [12, 12] N=3 | 81.46 [81.44, 81.47] N=3 |
| 1 | 2048 | 256 | 0.138 [0.1378, 0.138] N=3 | 12 [12, 12] N=3 | 79.64 [79.64, 79.64] N=3 |
| 1 | 4096 | 32 | 0.2698 [0.2695, 0.2698] N=3 | 12.1 [12.1, 12.1] N=3 | 49.34 [49.26, 49.44] N=3 |
| 1 | 8192 | 32 | 0.5683 [0.5681, 0.5685] N=3 | 12.4 [12.4, 12.4] N=3 | 33.55 [33.51, 33.55] N=3 |
| 64 | 128 | 512 | 0.5002 [0.4726, 0.5011] N=3 | 17.9 [17.8, 17.9] N=3 | 3411 [3406, 3413] N=3 |
| 64 | 512 | 512 | 1.132 [1.131, 1.148] N=3 | 19.5 [19.5, 19.5] N=3 | 2939 [2939, 2940] N=3 |
| 64 | 1024 | 256 | 2.202 [2.201, 2.209] N=3 | 25.6 [25.6, 25.6] N=3 | 1865 [1865, 1866] N=3 |
| 64 | 2048 | 256 | 4.198 [4.194, 4.263] N=3 | 36.5 [36.5, 36.5] N=3 | 1209 [1208, 1210] N=3 |
| 64 | 4096 | 32 | 8.611 [8.603, 8.612] N=3 | 287 [287, 288] N=3 | 116.5 [116.4, 116.6] N=3 |
| 64 | 8192 | 32 | 22.27 [22.26, 22.27] N=3 | 428 [428, 428] N=3 | 55.19 [55.19, 55.19] N=3 |

### sglang · TP=2 · Qwen2.5-7B-Instruct

| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |
|---|---|---|---|---|---|
| 1 | 128 | 512 | 0.01845 [0.01782, 0.01919] N=3 | 7.05 [7.05, 7.05] N=3 | 141 [141, 141.1] N=3 |
| 1 | 512 | 512 | 0.03179 [0.03131, 0.03189] N=3 | 7.05 [7.05, 7.05] N=3 | 140.5 [140.4, 140.5] N=3 |
| 1 | 1024 | 256 | 0.05091 [0.05084, 0.05093] N=3 | 7.07 [7.07, 7.07] N=3 | 137.6 [137.4, 137.7] N=3 |
| 1 | 2048 | 256 | 0.08561 [0.08524, 0.08566] N=3 | 7.1 [7.09, 7.1] N=3 | 134.7 [134.3, 134.7] N=3 |
| 1 | 4096 | 32 | 0.1658 [0.1658, 0.1658] N=3 | 7.13 [7.13, 7.13] N=3 | 82.17 [81.68, 82.25] N=3 |
| 1 | 8192 | 32 | 0.3464 [0.3464, 0.3464] N=3 | 7.25 [7.25, 7.25] N=3 | 55.68 [55.58, 55.84] N=3 |
| 64 | 128 | 512 | 0.3393 [0.3376, 0.3434] N=3 | 25.3 [25.2, 25.4] N=3 | 2472 [2455, 2479] N=3 |
| 64 | 512 | 512 | 0.7415 [0.7333, 0.7451] N=3 | 25.7 [25.6, 25.9] N=3 | 2351 [2340, 2358] N=3 |
| 64 | 1024 | 256 | 1.375 [1.371, 1.379] N=3 | 29.3 [29.2, 29.5] N=3 | 1843 [1842, 1849] N=3 |
| 64 | 2048 | 256 | 2.61 [2.593, 2.641] N=3 | 34.2 [34.2, 34.3] N=3 | 1436 [1435, 1439] N=3 |
| 64 | 4096 | 32 | 5.275 [5.275, 5.282] N=3 | 185 [185, 185] N=3 | 185 [184.3, 185] N=3 |
| 64 | 8192 | 32 | 11.26 [11.26, 11.27] N=3 | 366 [366, 366] N=3 | 90.28 [90.26, 90.32] N=3 |

### sglang · TP=4 · Qwen2.5-7B-Instruct

| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |
|---|---|---|---|---|---|
| 1 | 128 | 512 | 0.01559 [0.01557, 0.0157] N=3 | 4.23 [4.23, 4.23] N=3 | 233.9 [233.8, 234.1] N=3 |
| 1 | 512 | 512 | 0.02175 [0.02171, 0.02183] N=3 | 4.24 [4.24, 4.24] N=3 | 233.2 [232.8, 233.2] N=3 |
| 1 | 1024 | 256 | 0.03277 [0.03272, 0.03327] N=3 | 4.25 [4.24, 4.25] N=3 | 228.4 [228.2, 228.5] N=3 |
| 1 | 2048 | 256 | 0.05136 [0.05132, 0.05182] N=3 | 4.26 [4.26, 4.26] N=3 | 223.7 [223.6, 224] N=3 |
| 1 | 4096 | 32 | 0.09552 [0.09549, 0.09566] N=3 | 4.27 [4.27, 4.27] N=3 | 138.4 [135.5, 139] N=3 |
| 1 | 8192 | 32 | 0.1998 [0.1997, 0.2] N=3 | 4.34 [4.33, 4.34] N=3 | 94.58 [94.4, 94.76] N=3 |
| 64 | 128 | 512 | 0.2161 [0.2105, 0.2197] N=3 | 6.36 [6.34, 6.39] N=3 | 9373 [9364, 9380] N=3 |
| 64 | 512 | 512 | 0.4737 [0.4522, 0.4779] N=3 | 7.16 [7.16, 7.18] N=3 | 7899 [7877, 7936] N=3 |
| 64 | 1024 | 256 | 0.8265 [0.8025, 0.8272] N=3 | 9.38 [9.37, 9.4] N=3 | 5069 [5046, 5104] N=3 |
| 64 | 2048 | 256 | 1.499 [1.483, 1.552] N=3 | 12.7 [12.7, 12.7] N=3 | 3419 [3402, 3431] N=3 |
| 64 | 4096 | 32 | 3.037 [3.026, 3.038] N=3 | 100 [100, 100] N=3 | 330.9 [330.8, 331] N=3 |
| 64 | 8192 | 32 | 6.489 [6.485, 6.493] N=3 | 207 [207, 207] N=3 | 157.9 [157.4, 157.9] N=3 |

### vllm · TP=1 · Qwen2.5-7B-Instruct

| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |
|---|---|---|---|---|---|
| 1 | 128 | 512 | 0.02066 [0.02029, 0.02091] N=3 | 12.3 [12.3, 12.3] N=3 | 81.21 [81.17, 81.21] N=3 |
| 1 | 512 | 512 | 0.04085 [0.04071, 0.0409] N=3 | 12.3 [12.3, 12.3] N=3 | 80.86 [80.84, 80.86] N=3 |
| 1 | 1024 | 256 | 0.07135 [0.07125, 0.07157] N=3 | 12.3 [12.3, 12.3] N=3 | 79.6 [79.59, 79.62] N=3 |
| 1 | 2048 | 256 | 0.1318 [0.1316, 0.1319] N=3 | 12.3 [12.3, 12.3] N=3 | 77.89 [77.88, 77.92] N=3 |
| 1 | 4096 | 32 | 0.2674 [0.2673, 0.2674] N=3 | 12.4 [12.4, 12.4] N=3 | 48.76 [48.72, 48.81] N=3 |
| 1 | 8192 | 32 | 0.5725 [0.5725, 0.5726] N=3 | 12.5 [12.5, 12.5] N=3 | 33.15 [33.07, 33.17] N=3 |
| 64 | 128 | 512 | 0.3273 [0.3266, 0.3523] N=3 | 15.6 [15.6, 15.6] N=3 | 3926 [3922, 3931] N=3 |
| 64 | 512 | 512 | 0.5112 [0.5094, 0.5131] N=3 | 19.4 [19.3, 19.4] N=3 | 3140 [3136, 3140] N=3 |
| 64 | 1024 | 256 | 0.5452 [0.5439, 0.5465] N=3 | 31.9 [31.9, 32] N=3 | 1889 [1888, 1892] N=3 |
| 64 | 2048 | 256 | 0.6403 [0.6341, 0.6408] N=3 | 54.3 [54.2, 54.7] N=3 | 1127 [1126, 1128] N=3 |
| 64 | 4096 | 32 | 14.01 [14, 14.01] N=3 | 143 [143, 143] N=3 | 110.7 [110.7, 110.7] N=3 |
| 64 | 8192 | 32 | 34.51 [34.51, 34.52] N=3 | 153 [153, 153] N=3 | 52.09 [52.08, 52.09] N=3 |

### vllm · TP=2 · Qwen2.5-7B-Instruct

| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |
|---|---|---|---|---|---|
| 1 | 128 | 512 | 0.01408 [0.01405, 0.01432] N=3 | 7.52 [7.52, 7.52] N=3 | 132.3 [132.3, 132.4] N=3 |
| 1 | 512 | 512 | 0.02643 [0.02636, 0.02652] N=3 | 7.53 [7.53, 7.53] N=3 | 131.8 [131.7, 131.8] N=3 |
| 1 | 1024 | 256 | 0.04795 [0.04793, 0.0481] N=3 | 7.53 [7.53, 7.53] N=3 | 129.6 [129.6, 129.8] N=3 |
| 1 | 2048 | 256 | 0.08722 [0.08701, 0.08725] N=3 | 7.55 [7.55, 7.55] N=3 | 126.8 [126.4, 126.9] N=3 |
| 1 | 4096 | 32 | 0.1753 [0.1752, 0.1754] N=3 | 7.5 [7.5, 7.5] N=3 | 77.73 [77.67, 77.87] N=3 |
| 1 | 8192 | 32 | 0.3673 [0.3671, 0.3673] N=3 | 7.52 [7.51, 7.52] N=3 | 52.9 [52.8, 52.93] N=3 |
| 64 | 128 | 512 | 0.2186 [0.2179, 0.2246] N=3 | 9.72 [9.7, 9.77] N=3 | 6292 [6251, 6294] N=3 |
| 64 | 512 | 512 | 0.3412 [0.3267, 0.484] N=3 | 11.7 [11.4, 11.7] N=3 | 5121 [5120, 5140] N=3 |
| 64 | 1024 | 256 | 0.4278 [0.4273, 0.4281] N=3 | 18.9 [18.8, 18.9] N=3 | 3101 [3097, 3106] N=3 |
| 64 | 2048 | 256 | 0.4816 [0.4776, 0.4824] N=3 | 32.1 [32, 32.1] N=3 | 1884 [1884, 1885] N=3 |
| 64 | 4096 | 32 | 8.746 [8.745, 8.747] N=3 | 89.1 [89.1, 89.1] N=3 | 177 [177, 177] N=3 |
| 64 | 8192 | 32 | 21.47 [21.47, 21.48] N=3 | 94.8 [94.8, 94.9] N=3 | 83.64 [83.62, 83.65] N=3 |

### vllm · TP=4 · Qwen2.5-7B-Instruct

| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |
|---|---|---|---|---|---|
| 1 | 128 | 512 | 0.01153 [0.01144, 0.01158] N=3 | 5.16 [5.16, 5.16] N=3 | 192.6 [192.6, 192.7] N=3 |
| 1 | 512 | 512 | 0.01957 [0.01951, 0.0197] N=3 | 5.16 [5.16, 5.16] N=3 | 192.1 [192, 192.2] N=3 |
| 1 | 1024 | 256 | 0.03118 [0.03104, 0.0312] N=3 | 5.16 [5.16, 5.16] N=3 | 189.4 [189.2, 189.5] N=3 |
| 1 | 2048 | 256 | 0.05407 [0.05404, 0.05416] N=3 | 5.16 [5.16, 5.16] N=3 | 186.1 [185.9, 186.2] N=3 |
| 1 | 4096 | 32 | 0.1066 [0.1065, 0.1066] N=3 | 5.08 [5.08, 5.08] N=3 | 118.7 [117.9, 120] N=3 |
| 1 | 8192 | 32 | 0.221 [0.2209, 0.2211] N=3 | 5.01 [5.01, 5.01] N=3 | 84.02 [83.75, 84.05] N=3 |
| 64 | 128 | 512 | 0.1785 [0.159, 0.1803] N=3 | 6.09 [6.09, 6.09] N=3 | 9865 [9857, 9922] N=3 |
| 64 | 512 | 512 | 0.2786 [0.2366, 0.2839] N=3 | 7.16 [7.16, 7.17] N=3 | 8307 [8291, 8312] N=3 |
| 64 | 1024 | 256 | 0.2002 [0.2001, 0.3059] N=3 | 11.5 [11, 11.5] N=3 | 5153 [5135, 5159] N=3 |
| 64 | 2048 | 256 | 0.4351 [0.3832, 0.4361] N=3 | 18.1 [18, 18.2] N=3 | 3223 [3219, 3225] N=3 |
| 64 | 4096 | 32 | 5.154 [5.151, 5.156] N=3 | 52.5 [52.5, 52.5] N=3 | 299.6 [299.5, 299.8] N=3 |
| 64 | 8192 | 32 | 12.87 [12.85, 12.88] N=3 | 56.7 [56.6, 56.8] N=3 | 139.5 [139.4, 139.6] N=3 |

## Automatic prefix caching

Measured hit rate is computed as a **per-phase delta** of the engine's
cumulative counters, because the cache-reset endpoint clears the cache without
resetting the counters. The theoretical bound is what this workload could
achieve with no eviction; the gap between them is eviction and block-granularity
rounding.

| backend | shared-prefix ratio | measured hit rate | theoretical bound |
|---|---|---|---|
| sglang | 0 | 0 [0, 0] N=164 | 0 |
| sglang | 0.3 | 0 [0, 0] N=144 | 0.146 |
| sglang | 0.6 | 0 [0, 0] N=164 | 0.294 |
| sglang | 0.9 | 0 [0, 0] N=144 | 0.446 |
| vllm | 0 | 0 [0, 0] N=179 | 0 |
| vllm | 0.3 | 0.146 [0.0938, 0.148] N=144 | 0.146 |
| vllm | 0.6 | 0.294 [0.208, 0.298] N=184 | 0.294 |
| vllm | 0.9 | 0.446 [0.391, 0.448] N=144 | 0.446 |

## Cost of packed placement

Identical server group, same node, same job: once alone with three GPUs idle,
once as four NUMA-pinned replicas with only GPU 0 measured.

| phase | metric | isolated | packed | change |
|---|---|---|---|---|
| closed_c1 | TTFT p50 (s) | 0.07198 [0.07198, 0.09382] N=3 | 0.1805 [0.156, 0.2161] N=3 | **+150.7%** worse |
| closed_c1 | ITL p50 (s/tok) | 0.01221 [0.01221, 0.01221] N=3 | 0.01212 [0.01212, 0.01212] N=3 | **-0.8%** better |
| closed_c1 | output tok/s | 80.1 [79.51, 80.14] N=3 | 76.07 [76.06, 76.1] N=3 | **-5.0%** worse |
| closed_c128 | TTFT p50 (s) | 0.9598 [0.8107, 1.09] N=3 | 35.06 [34.61, 35.54] N=3 | **+3553.1%** worse |
| closed_c128 | ITL p50 (s/tok) | 0.02168 [0.02167, 0.02169] N=3 | 0.1634 [0.1628, 0.1979] N=3 | **+653.5%** worse |
| closed_c128 | output tok/s | 2185 [2184, 2185] N=3 | 475.7 [469, 482.4] N=3 | **-78.2%** worse |
| closed_c32 | TTFT p50 (s) | 0.3856 [0.3854, 0.387] N=3 | 0.6261 [0.5994, 0.6272] N=3 | **+62.4%** worse |
| closed_c32 | ITL p50 (s/tok) | 0.0146 [0.0146, 0.01462] N=3 | 0.02168 [0.02167, 0.0217] N=3 | **+48.4%** worse |
| closed_c32 | output tok/s | 1448 [1447, 1453] N=3 | 544 [542.4, 544.3] N=3 | **-62.4%** worse |
| closed_c8 | TTFT p50 (s) | 0.2664 [0.2191, 0.3177] N=3 | 0.407 [0.2856, 0.5099] N=3 | **+52.8%** worse |
| closed_c8 | ITL p50 (s/tok) | 0.01259 [0.01259, 0.0126] N=3 | 0.01458 [0.01458, 0.01461] N=3 | **+15.8%** worse |
| closed_c8 | output tok/s | 550.3 [549.7, 550.3] N=3 | 362.6 [362.2, 362.7] N=3 | **-34.1%** worse |
| open_r8 | TTFT p50 (s) | 0.2378 [0.2279, 0.2382] N=3 | 87.77 [77.5, 89.74] N=3 | **+36800.7%** worse |
| open_r8 | ITL p50 (s/tok) | 0.01934 [0.01933, 0.01946] N=3 | 0.1674 [0.1668, 0.1823] N=3 | **+765.3%** worse |
| open_r8 | output tok/s | 1820 [1818, 1821] N=3 | 497.8 [497.8, 498.3] N=3 | **-72.6%** worse |
| weak_closed_c16pergpu | TTFT p50 (s) | 0.373 [0.3373, 0.3736] N=3 | 0.5272 [0.4126, 0.5567] N=3 | **+41.3%** worse |
| weak_closed_c16pergpu | ITL p50 (s/tok) | 0.01321 [0.01321, 0.01321] N=3 | 0.01796 [0.01796, 0.01797] N=3 | **+35.9%** worse |
| weak_closed_c16pergpu | output tok/s | 945.1 [943.9, 945.1] N=3 | 474.8 [469.3, 478.3] N=3 | **-49.8%** worse |

## Configurations that did not run

Recorded rather than dropped: a configuration that cannot run is a real
boundary of the system, and its absence from a sweep should be visible in
the data rather than only in someone's memory.

| backend | model | TP | status | runs | cause |
|---|---|---|---|---|---|
| sglang | Qwen2.5-32B-Instruct | 4 | `launch_failed` | 80 | sglang exited with code -9 during startup. Log tail: |
| sglang | Qwen2.5-7B-Instruct | 1 | `launch_failed` | 6 | sglang exited with code -9 during startup. Log tail: |
| sglang | Qwen2.5-7B-Instruct | 2 | `failed` | 30 | no requests succeeded |
| sglang | Qwen2.5-7B-Instruct | 4 | `failed` | 36 | no requests succeeded |

