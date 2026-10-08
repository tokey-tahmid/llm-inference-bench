# llm-inference-bench

A backend-agnostic performance characterisation harness for LLM inference.

This is not a vLLM benchmark script. It treats the serving engine as a swappable
backend behind one interface, so that adding a third engine requires a new adapter
class and a registry entry, and nothing else. vLLM and SGLang are the two
implementations.

> **Status: measured.** The harness has produced 4,573 raw artifacts under
> `results/raw/` across eleven sweeps: phase-1 validation, full 7B and 32B
> matrices for both backends, TP-scaling, prefix caching, speculative decoding,
> length scaling, PagedAttention block-size / memory-utilisation, saturation,
> long context, and packing interference. See `results/RESULTS.md` for the
> current tables (generated, not transcribed). One coverage gap remains: the
> TP=8 multi-node point is unmeasured, because the pinned vLLM image ships
> without Ray (see Known limitations).

## What it measures

Per configuration: TTFT (p50/p95/p99), inter-token latency, end-to-end request
latency (p50/p95/p99), output tokens/sec, request throughput, GPU memory
occupancy, prefix cache hit rate, and speculative acceptance rate where the engine
exposes them.

Sweep axes: backend, model, tensor-parallel degree, concurrency and request rate,
input and output length, automatic prefix caching on/off, and speculative decoding
mode.

## Hardware

All results come from **MeluXina** (LuxProvide), `gpu` partition for ICHEC project:

- 4x NVIDIA A100-SXM4-40GB per node, **full NVLink mesh** (`NV4` between every
  pair, no intra-node PCIe path between GPUs)
- 128 logical cores, 480 GB RAM, 2x 200 Gb/s InfiniBand (Mellanox MT4123)
- Driver 595.71.05, CUDA 13.2
- Diskless nodes: `/` is tmpfs, `/dev/shm` is 428 GB

Because the intra-node fabric is uniform NVLink, TP∈{1,2,4} scaling results
cannot show an NVLink-versus-PCIe contrast; the only interconnect boundary
available is multi-node TP=8 over InfiniBand. See `docs/notes.md`.

Models are Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct, both bf16, pinned by
revision SHA. Llama was not used because it is gated and no HuggingFace token
exists on this system; Qwen2.5-32B replaces Llama-3.3-70B on the large-model axis
because 70B in bf16 needs ~141 GB of the 160 GB on a node, leaving almost no KV
cache and only a single TP point rather than a curve.

## Layout

```
src/llm_inference_bench/
  backends/         adapter interface + vLLM and SGLang implementations
  workload/         prompt generation with a controllable shared-prefix ratio
  client.py         open- and closed-loop load generation over the OpenAI API
  metrics.py        percentiles, spread, failure taxonomy
  provenance.py     the only writer of results/raw/
  sweep.py          YAML sweep expansion into server groups and phases
  runner.py         execution: server groups in, raw artifacts out
configs/            sweep definitions
slurm/              batch scripts
analysis/           reads results/raw/ only
results/raw/        real measurements only, append-only, never hand-written
docs/notes.md       findings and surprises
docs/budget.md      node-hours consumed, per job id
```

## Running it

Provisioning must run as a batch job, not on the login node: on this cluster the
login node has no `module` and no `apptainer`, while compute nodes have both and
also have outbound internet.

```bash
sbatch provision.sh                   # containers, python env, weights
sbatch provision.sh harness           # editable install only
sbatch slurm/run_tests.sh             # unit tests and lint (gpu partition, no GPU used)
sbatch slurm/probe_capabilities.sh    # verify adapter flags against the images
sbatch slurm/run_sweep.sh configs/phase1_validation.yaml
```

Plan and cost a sweep without spending any GPU time:

```bash
python -m llm_inference_bench.cli expand configs/phase1_validation.yaml
```

## Design decisions worth knowing

**Sweep axes are split into server axes and phase axes.** Launching an engine
costs minutes; concurrency and workload shape do not require a relaunch, while TP
degree and prefix caching do. Phases are grouped under one server launch, which
collapses a 4-concurrency x 3-repetition matrix from 12 model loads into 1. The
correctness cost is that phases share cache state, so the runner resets the
engine's prefix cache between phases and records whether the reset was
acknowledged.

**Prompts are sent as token IDs, not text.** Prefix caching keys on an exact token
prefix. Building prompts in token space, decoding to text and letting the server
re-tokenise is not a guaranteed round trip, so the "shared" region can shift and
silently stop being shared, producing a near-zero hit rate that reads as a finding
about the engine rather than a bug in the benchmark.

**Both open- and closed-loop load generation.** Closed loop cannot show queueing
collapse, because backpressure throttles the offered load. The latency knee past
saturation only appears under open-loop Poisson arrivals. The client records its
own scheduling drift, so a run where the load generator was the bottleneck is
identifiable in the data rather than mistaken for server saturation.

**Failed configurations are recorded, not dropped.** An OOM at a given
batch/context is a real boundary. Launch failures, timeouts and configurations a
backend cannot support each produce an artifact with a status, so the sweep matrix
has no silent holes.

## Results

The numbers live in [`results/RESULTS.md`](results/RESULTS.md) and the figures
in `results/figures/`. Both are generated from `results/raw/` and are never
edited by hand; regeneration is the single command in the "Reproducing"
section below. This README links to them rather than transcribing them because
"every number traces to a committed raw artifact" only stays true if no human
retypes the numbers.

The measurements come from the exact software and hardware described in the
provenance block of every raw artifact:

- **Hardware:** 4x NVIDIA A100-SXM4-40GB, driver **595.71.05**, CUDA 13.2.
  Full NVLink mesh (`NV4` between every intra-node GPU pair); no PCIe path
  between GPUs on a node. Multi-node crosses 2x 200 Gb/s InfiniBand.
- **Backends:** vLLM **0.27.1** and SGLang **0.5.18**, both pinned by image
  digest in `provision.lock.json`.
- **Models:** Qwen2.5-7B-Instruct at revision
  `a09a35458c702b33eeacc393d103063234e8bc28` and Qwen2.5-32B-Instruct at
  revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, both bf16.
- **Methodology:** N>=3 per configuration (N=5 for the loaded 7B points where
  spread was ~11%), one warmup repetition per config recorded and excluded,
  median with min/max reported, spread shown on every plot.

### Headline findings

Each of these is a section in `docs/notes.md`, with the raw job IDs that
observed it. They are summarised here so the README stands on its own; the
tables and figures live in `results/RESULTS.md` and `results/figures/`.

- **Strong-scaling parallel efficiency falls to ~0.69 at TP=4** for vLLM on
  Qwen2.5-7B at c=1024 (2.75x speedup on 4 GPUs). SGLang lands at 0.65 at
  c=128. Every intra-node GPU pair here is NVLink `NV4`, so the falloff is
  *not* an interconnect effect: it is kernel efficiency, per-replica memory
  bandwidth, or scheduler overhead. A TP=8 multi-node point would be the
  *only* configuration where the interconnect is observable; it is not
  measured (see Known limitations).
- **Prefix caching reaches the theoretical bound on both backends.** The
  medians sit at or just under the no-eviction bound at every shared-prefix
  ratio (table in `results/RESULTS.md`). SGLang exposes no hits/queries
  counter and its `cache_hit_rate` gauge reads 0 even while caching works, so
  its rate is derived from `prompt_tokens_total` and
  `uncached_prompt_tokens_histogram_sum`, taken as a per-phase delta.
- **Speculative decoding stops helping throughput between c=64 and c=128** on
  a single vLLM replica, at n-gram acceptance ~0.9. The mechanism is
  saturation, not prediction quality: acceptance barely drops while the
  latency win rises. See docs/notes.md.
- **Packed placement is not measurement-safe.** Running four TP=1 replicas
  NUMA-pinned on one billed node degrades TTFT p50 by 150% at c=1 and output
  throughput by 34% at c=8, versus running one replica alone with three GPUs
  idle. Headline numbers therefore use unpacked runs (three-quarters of each
  billed hour is spent on idle GPUs), and only a documented trend-vs-absolute
  distinction saves any packed data.
- **Prefill dominates TTFT and scales with input length** (see the length
  sweep in `results/RESULTS.md`); **decode dominates end-to-end latency and
  is memory-bandwidth-bound at low concurrency**, transitioning to
  compute-bound at high concurrency. This is the axis the P1 spec asked
  the writeup to address.

### Known limitations

Recorded here rather than in a footnote, per the "failed configurations are
data" rule:

- **Dirty git tree.** 2,794 of 4,573 raw artifacts were written from a
  `-dirty` tree during the intensive collection window. Provenance is exact
  (git SHA is stamped with the `-dirty` suffix, image digest is pinned), but
  a clean-checkout replay of *those specific artifacts* would not
  bit-reproduce. Sweeps produced after 2026-08-31 are on a clean tree.
- **SGLang 32B runs at TP=4 all launched-failed** (SIGKILL during startup).
  Recorded in `results/RESULTS.md`'s "Configurations that did not run"
  section. The 32B scaling curve on SGLang is single-point at TP=2 because
  of this.
- **TP=8 multi-node is not measured.** Probe job 5174440 found no Ray in the
  pinned vLLM image, and Ray is what vLLM's multi-node backend needs.
  Rebuilding the image would change the pinned digest that every other sweep
  was measured under, which breaks cross-sweep comparability, so the axis is
  left empty rather than filled from a different software stack.
  `slurm/run_tp8_ray.sh` and `configs/tp8_multinode.yaml` are committed for a
  Ray-equipped image.

## Reproducing

Every figure and table is regenerable end-to-end from `results/raw/` by one
documented command:

```bash
sbatch slurm/run_analysis.sh
```

That job reads only `results/raw/`, writes only `results/figures/` and
`results/RESULTS.md`, and never invents a value. It runs on the `gpu`
partition because `p201466` has no CPU allocation, but uses no GPU.

Raw artifacts are append-only, written read-only, and carry full provenance:
git SHA (with a `-dirty` suffix when the tree was not clean), hostname,
Slurm job ID, GPU model and count, driver and CUDA version, `nvidia-smi topo -m`
verbatim, backend version and image digest, model revision SHA, the complete
config, and the exact command line.

To capture or verify the performance regression baseline:

```bash
sbatch slurm/run_regression_check.sh --sweep-name full-7b --update-baseline
sbatch slurm/run_regression_check.sh --sweep-name full-7b
```

The check compares median metrics per (backend, model, TP, prefix caching,
phase) against a stored baseline in `baselines/p1.json`. The tolerance is
derived from the baseline's own measured spread, not picked from the air; a
missing configuration is a failure, not a pass; only degradation is flagged.
