# llm-inference-bench

A backend-agnostic performance characterisation harness for LLM inference.

This is not a vLLM benchmark script. It treats the serving engine as a swappable
backend behind one interface, so that adding a third engine requires a new adapter
class and a registry entry, and nothing else. vLLM and SGLang are the two
implementations.

> **Status: no measurements yet.** The harness is built and the environment is
> characterised, but no benchmark has been run. This README will carry numbers only
> once they trace to a committed artifact under `results/raw/`. There are
> deliberately no placeholder figures, example results, or illustrative numbers
> anywhere in this repository.

## What it measures

Per configuration: TTFT (p50/p95/p99), inter-token latency, end-to-end request
latency (p50/p95/p99), output tokens/sec, request throughput, GPU memory
occupancy, prefix cache hit rate, and speculative acceptance rate where the engine
exposes them.

Sweep axes: backend, model, tensor-parallel degree, concurrency and request rate,
input and output length, automatic prefix caching on/off, and speculative decoding
mode.

## Hardware

All results come from **MeluXina** (LuxProvide), `gpu` partition:

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
sbatch slurm/run_tests.sh             # unit tests and lint (cpu partition)
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

## Reproducing

Every figure and table is regenerable from `results/raw/` by one documented
command, which will be stated here alongside the first results. Raw artifacts are
append-only, written read-only, and carry full provenance: git SHA (with a
`-dirty` suffix when the tree was not clean), hostname, Slurm job ID, GPU model
and count, driver and CUDA version, `nvidia-smi topo -m` verbatim, backend version
and image digest, model revision SHA, the complete config, and the exact command
line.
