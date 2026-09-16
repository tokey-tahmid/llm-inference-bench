# Notes: findings, source reading, surprises

Running log. Entries are dated and state what was actually observed, with the job
ID that observed it, so anything here can be re-checked rather than trusted.

---

## 2026-08-25 — MeluXina environment characterisation

Probe jobs `5143240`, `5143256`, `5143260` (gpu, `mel2017`/`mel2107`), `5143547` (cpu, `mel0193`).

### The provision/run split is inverted on this cluster

Standard HPC practice is to provision on a networked login node and run offline
on compute nodes. Here it is the other way round:

| | login node | compute node |
|---|---|---|
| outbound internet | yes | **yes** (HF 200, PyPI 200, GitHub 200, Docker registry reachable) |
| `module` | **no** | yes |
| `apptainer` | **no** | yes |
| `uv` / `cargo` | **no** | yes (via modules) |

The login node has only `/usr/bin/python3`, and it is **Python 3.6.8**, which
cannot even parse this repo's source. So provisioning runs as a `cpu`-partition
batch job, and the first opportunity to syntax-check anything is inside a job.
`probe_capabilities.sh` therefore opens with a `compileall` gate: it costs
seconds and catches what a local check normally would.

### GPU topology: uniform NVLink, so an intra-node PCIe contrast does not exist

`nvidia-smi topo -m` reports `NV4` between **every** GPU pair on a node. There is
no PCIe path between GPUs to compare against. The workspace CLAUDE.md anticipated
"NVLink vs PCIe paths should be visible in the numbers", and intra-node they
cannot be: every TP≤4 configuration communicates over the same fabric. The only
place an interconnect contrast is observable is the multi-node boundary, where TP=8
crosses 200 Gb/s InfiniBand. That single point now carries the whole
"where does parallel efficiency fall off, and why" argument, which raises its
value rather than lowering it.

NIC affinity is not uniform though, and may matter for multi-node: `mlx5_0` is
`PIX` to GPU1, `mlx5_1` is `PIX` to GPU2, while GPU0 and GPU3 reach both NICs via
`SYS`. Worth checking whether TP=8 rank placement interacts with this.

### Nodes are diskless

`/` is a 252 GB tmpfs and `/dev/shm` is 428 GB. There is no node-local disk.
Two consequences: model weights stream from Lustre on every launch (so startup
time is partly a filesystem measurement, and array tasks must be staggered), and
`/tmp` is a fine place for per-run scratch but its contents vanish with the job.

The `--output` path of a batch script must be on a shared filesystem. A job that
writes its output to `/tmp` succeeds and leaves nothing behind, which looks
exactly like a job that never ran (cost one confused debugging cycle, job `5143240`).

---

## 2026-08-26 — Container environment traps

### Lustre is read-only inside the container, and torch notices at import time

Capability probe `5146938` failed in a way worth recording because the symptom
pointed nowhere near the cause.

Symptom: the reconciler reported that *every* flag was missing from both images,
including `--host` and `--port`, and produced a page of confident OVERCLAIM
findings about capabilities the adapters declared.

Actual cause: `/mnt/tier2` is mounted **read-only** inside an Apptainer container
by default here. `env.sh` exports `TORCHINDUCTOR_CACHE_DIR` pointing at project
storage, and torch resolves *and creates* its inductor cache directory at **import
time**:

```
File ".../torch/_inductor/runtime/cache_dir_utils.py", line 20, in cache_dir
    os.makedirs(cache_dir, exist_ok=True)
OSError: [Errno 30] Read-only file system: '/mnt/tier2'
```

So `import vllm` died before argparse ever ran, `--help` emitted a traceback, and
a traceback contains no flags. Every downstream conclusion was an artefact.

Two fixes, both applied:

1. **Bind explicitly read-write** (`--bind $LIB_DATA:$LIB_DATA:rw`) for anything
   that must be written, and point compile/scratch caches at the node's own tmpfs
   instead. This has an incidental benefit: per-run compile caches cannot warm a
   later run's startup, which would otherwise be an invisible confound in the
   startup-time numbers.
2. **The reconciler now validates its own input.** A help capture that is a
   traceback, or that lacks universally present flags like `--host`, is rejected
   as unusable rather than reported as a capability finding. A tool that reports
   confident nonsense when its input is broken is worse than one that fails.

The general lesson, which applies to the whole harness: a verification step needs
its own sanity check, or it will happily verify garbage.

### Provenance placeholders are worse than absent fields

The same failed probe wrote `"version": "UNKNOWN"` into `provision.lock.json`.
Every raw artifact copies `backend_version` from that lockfile, so this would have
stamped a fake value onto real measurements. The reconciler now refuses to record
a placeholder and leaves the key absent instead. Absent is honest; a placeholder
looks like data.

---

## Open questions

RESOLVED 2026-08-26/27, see the entry below:
- ~~Backend versions unread~~ -> vLLM **0.27.1**, SGLang **0.5.18**, both in the lockfile.
- ~~Integer-token prompts unverified~~ -> confirmed exact on vLLM (64 sent, 64 reported).
- ~~`ignore_eos` unverified~~ -> confirmed honoured on vLLM (`output_length_exact: true`).

Still open:
- Both confirmations are **vLLM only**. SGLang has passed the capability gate but
  has not yet served a request, so `token_id_prompts_supported` and
  `output_length_exact` are unverified there. Both are checked automatically on
  its first run and will fail loudly rather than silently if they do not hold.
- SGLang's `--mem-fraction-static` is not the same quantity as vLLM's
  `--gpu-memory-utilization` (static share vs total pool). Matched numeric values
  do not mean matched memory budgets, so a backend-vs-backend comparison needs
  either a calibration step or an explicit caveat. Not yet decided which.
- Packing interference is unmeasured, so packed runs cannot carry headline
  numbers yet. `slurm/measure_packing_interference.sh` exists to settle it.

---

## 2026-08-26/27 — the first real run, and what it caught

Phase-1 validation sweep, job `5148060`, vLLM 0.27.1, Qwen2.5-7B-Instruct at
TP=1 on one A100-40GB. 8 artifacts, 128/128 requests successful, 2 warmup
repetitions correctly recorded and excluded.

### Two design assumptions confirmed, not assumed

Both of these were load-bearing and both are now measured rather than believed:

* **Integer-token prompts round-trip exactly.** `validate_prompt_token_fidelity`
  sent 64 token IDs and vLLM reported `prompt_tokens: 64`. This is what makes
  the shared-prefix ratio a real knob: had the engine re-tokenised, the "shared"
  region would drift and the prefix-cache measurement would have quietly become
  a measurement of nothing.
* **`ignore_eos` is honoured.** `output_length_exact: true` across every phase,
  so the output-length axis is controlled and cross-configuration comparisons
  are legitimate.

### Two defects found, both by recording rather than assuming

**`/reset_prefix_cache` returned 404 on every phase boundary.** vLLM 0.27.1 has
moved it to `vllm/entrypoints/serve/dev/cache/api_router.py`, registered only
inside `if envs.VLLM_SERVER_DEV_MODE:` in `api_server.py` (located by probe
`5150753`). Fixed by setting `VLLM_SERVER_DEV_MODE=1` through a new per-adapter
`extra_container_env()` hook, which keeps an engine's private environment out of
the backend-neutral launcher.

The severity is worth being precise about: in *this* sweep it was harmless,
because prefix caching was off for every phase, so there was no cache state to
inherit. It would have become serious the moment the prefix-caching axis was
enabled, and it would have been very hard to see from the numbers alone: later
phases would simply have looked faster. The reason it was caught at all is that
`reset_caches` records the HTTP status rather than assuming success, and the
integrity check treats a failed reset as a problem rather than a detail.

**`backend_version` was `None` in every artifact.** The version is only known
from the provisioning lockfile and was never plumbed into the adapter. The image
digest already pinned reproducibility exactly, so nothing was unreproducible,
but an artifact that cannot state its backend version is not fully traceable.
The runner now refuses to measure at all when the lockfile has no version for a
backend, rather than emitting artifacts with a hole in them.

### The general lesson

Every one of the four failures on the road to a first measurement was found by
a check that recorded an outcome instead of assuming one: the reconciler's
usability guard, the traceback detector, the cache-reset status, the git-dirty
flag. None of them were found by reading code. The cost of each check is a few
lines; the cost of not having them is a plausible-looking number.

### Cluster and accounting note (2026-08-27)

Compute moved to account `p201466`, which had 113 of 126 GPU node-hours free
against 44 remaining on `p201362`. Storage stays under `p201362`; the two are
unrelated, since filesystem access follows unix group membership rather than the
Slurm account. `p201466` has **no CPU allocation at all** (`gres/cpun=0`), so
every job now runs on the `gpu` partition, including provisioning and unit
tests, which do not need a GPU but do need an account that can pay for the node.

---

## 2026-08-31 — packing four single-GPU runs onto one node is expensive

Job `5162712`, vLLM 0.27.1, Qwen2.5-7B at TP=1, N=3 repetitions per phase.
Identical server group run twice on the same node in the same job: once alone on
GPU 0 with the other three GPUs idle, once as four NUMA-pinned replicas with only
GPU 0's replica measured.

| metric | c=1 | c=8 |
|---|---|---|
| TTFT p50 | **+150.7 %** | +52.8 % |
| TTFT p95 | +132.6 % | +48.1 % |
| ITL p50 | -0.8 % | +15.8 % |
| output throughput | -5.0 % | **-34.1 %** |

**The 4x budget multiplier is not free, and the answer is decisive: packed runs
cannot carry headline numbers.** They remain fine for trend claims where a
consistent bias across compared configurations cancels, but any absolute latency
or throughput figure from a packed run would be wrong by the margins above.

This is precisely why it was measured rather than assumed away. Had the main
sweep been packed for the 4x saving, every number in it would have carried a
34 % throughput error and a 2.5x TTFT error, and nothing in the data would have
revealed it: the runs succeed, the artifacts look clean, the numbers are simply
wrong.

### The shape of the interference says what is contended

The interesting detail is that the two latency components behave completely
differently at low load:

* At c=1, **ITL is untouched (-0.8 %) while TTFT more than doubles.**
* At c=8, ITL degrades (+15.8 %) and throughput collapses (-34.1 %).

Decode is GPU-local: each step reads weights already resident in HBM, so a
neighbouring replica on a different GPU barely perturbs it. Prefill is not: it
moves the prompt through the host, competes for memory bandwidth and PCIe, and
four concurrent prefills contend directly. So at low load the damage is confined
to the prefill path, which is exactly what TTFT measures. Under real concurrency
the contention reaches the decode loop too and throughput follows.

That decomposition is worth more than the headline percentage: it says the
binding shared resource is host-side bandwidth on the prefill path, not the
NVLink mesh or the GPUs themselves.

### Consequence for the rest of P1

Every sweep in this project ran unpacked, one server group per billed node. That
costs 75 % of each billed hour on TP=1 groups and is the right trade: the whole
point of the exercise is numbers that can be quoted.

---

## 2026-08-31 — the prefix cache was working; the metric name was not

The prefix-caching figure first came out with a measured hit rate of exactly
**0.000** at every shared-prefix ratio, against a theoretical bound rising to
0.29. That is the precise failure the token-ID prompt design exists to prevent,
so it looked like the benchmark had the disease it was built to avoid.

It had not. Two separate faults, both in the *reading* of the metric rather than
in the caching itself.

### 1. vLLM 0.27.1 renamed the counters

The `gpu_` infix was dropped:

| old (adapter looked for this) | vLLM 0.27.1 actually exposes |
|---|---|
| `vllm:gpu_prefix_cache_hits_total` | `vllm:prefix_cache_hits_total` |
| `vllm:gpu_prefix_cache_queries_total` | `vllm:prefix_cache_queries_total` |
| `vllm:gpu_cache_usage_perc` | `vllm:kv_cache_usage_perc` |

The adapter found nothing under the old names and, correctly, reported the hit
rate as **absent rather than zero**. That design choice is what made the fault
diagnosable: had it defaulted to 0.0, the figure would have shown a plausible
"prefix caching does not work here" result with no way to tell it from a
measurement bug. The 0.000 line that did appear came from SGLang, which reports a
genuine zero.

**Nothing had to be re-measured.** The full Prometheus exposition is stored
verbatim in every artifact, so the correct counters were already on disk and the
existing runs could simply be reprocessed. That is the argument for storing raw
telemetry next to the normalised keys, and it paid for itself here at a cost of
about 40 KB per artifact.

### 2. The counters are cumulative and the cache reset does not clear them

`/reset_prefix_cache` empties the cache but leaves the Prometheus counters
running for the lifetime of the server process. Since one server serves many
phases (that is the whole point of the server/phase split), the raw ratio at the
end of a phase includes every phase before it. A 0.9-ratio phase following a
0.0-ratio phase reads low, and the on/off comparison becomes sensitive to phase
ordering, which is not a property anyone wants their results to have.

Fixed by differencing consecutive scrapes within a server group, ordered by time.
The first phase of a group has no predecessor and gets NaN rather than a guess; a
negative delta means the server restarted and is discarded rather than reported.

Recovered per-phase result, tracking just under the achievable bound:

| shared-prefix ratio | measured hit rate | theoretical bound |
|---|---|---|
| 0.0 | 0.000 | 0.00 |
| 0.3 | 0.094 | ~0.15 |
| 0.6 | 0.234 | ~0.29 |

The residual gap is eviction and block-granularity rounding. Having the bound on
the same axes is what makes that gap interpretable rather than just a number.

### 3. A spread band that was not noise

The same figure's outer panels carried enormous bands, which read as though the
measurement were wildly unrepeatable. They were not measurement spread at all:
grouping only by shared-prefix ratio pooled 7B with 32B, TP=1 with TP=4 and c=1
with c=128, so the band spanned the heterogeneity of the whole matrix. Phase 1
measured 0.1 % repetition spread at c=1, so that reading was badly wrong.

The figure now pins every axis except the one under study and names the slice in
the caption. General lesson, and the same one as the reconciler's usability
guard: **an aggregate is only as meaningful as the thing it is aggregating over,
and a plot will happily average across a distinction that matters.**

---

## 2026-08-31 — inter-token latency was measuring steps, not tokens

The speculative-decoding sweep produced two numbers that cannot both be true:

| | ITL p50 | output tok/s |
|---|---|---|
| no speculation | 12.21 ms | 80.2 |
| n-gram speculation | 13.60 ms (**+11.3 %**) | 243.0 (**+203 %**) |

Throughput triples while per-token latency supposedly gets worse. If tokens
arrive three times faster, the time between them cannot rise.

### The check that settled it

If one SSE chunk carries one token, then `itl_p50 x (tokens - 1)` should
reconstruct the decode span `e2e - ttft`. Measured ratio of predicted to actual:

| | ratio |
|---|---|
| no speculation | **0.87** (chunks ~ tokens, ITL is per token) |
| n-gram speculation | **3.28** (each chunk carried ~3.3 tokens) |

So the real decode was 1.059 s for 256 tokens, **4.1 ms/token**, against a
reported ITL of 13.6 ms. Inter-token latency was measuring the *scheduler step
interval*, not the token interval.

### Why this is the worst kind of bug

It does not look like an error. Every number is plausible, every run succeeds,
the artifacts are clean, and the conclusion it produces (*"speculation costs 11 %
inter-token latency, so it trades latency for throughput"*) is a perfectly
reasonable-sounding claim that someone could defend in an interview. It is simply
backwards: speculation improved per-token latency roughly threefold.

The mechanism is obvious in hindsight and was written down in the code before it
was violated. `RequestResult.chunk_token_counts` carries this comment:

> A chunk is not always exactly one token, and the token count per chunk is kept
> so ITL is never silently miscomputed as per-chunk.

and the code then appended a hardcoded `1` for every chunk. **The comment
described the intent; the code did the opposite.** A comment is not a control.

Speculative decoding is precisely the feature that breaks the one-token-per-chunk
assumption, because verifying several draft tokens in a single forward pass and
streaming them together is the entire point of it. The benchmark's own headline
speculative-decoding result was the thing the bug was guaranteed to corrupt.

### Fix, and full recovery of already-collected data

`(end - first_token) / (tokens - 1)` uses the server's own token count and is
correct regardless of how the stream was chunked. Now reported as
`itl_per_token_s` alongside the chunk-derived `itl_s`, with
`median_tokens_per_chunk` and a `stream_batches_tokens` flag so the discrepancy
can never again be invisible.

Nothing needed re-measuring: every request's ttft, e2e and token count are
already stored per artifact, so the decode span divides out exactly. That is the
second time storing generous per-request detail has rescued a metric bug after
the fact, the first being the raw Prometheus exposition.

**Rule this suggests:** any derived rate should be cross-checked against an
independently derived total. Throughput and ITL are computed from different
quantities, and their disagreement is what exposed this. A single metric with no
redundant path to the same physical fact cannot audit itself.

---

## 2026-08-31 — where speculative decoding stops helping, and why

With inter-token latency measured per token rather than per chunk, the
speculative-decoding sweep answers the question the definition of done asks
directly. vLLM 0.27.1, Qwen2.5-7B, n-gram speculation with 4 draft tokens,
shared-prefix ratio 0.6 so the lookup has repeated text to match against.

| TP | concurrency | per-token ITL off | per-token ITL spec | ITL change | throughput change | acceptance |
|---|---|---|---|---|---|---|
| 1 | 1 | 12.21 ms | 3.05 ms | **-75.0 %** | **+203.0 %** | 0.956 |
| 1 | 4 | 12.35 ms | 3.93 ms | -68.2 % | +100.1 % | 0.939 |
| 1 | 16 | 15.38 ms | 6.66 ms | -56.7 % | +43.9 % | 0.898 |
| 1 | 64 | 31.57 ms | 18.44 ms | -41.6 % | +12.5 % | 0.899 |
| 1 | 128 | 55.14 ms | 36.38 ms | -34.0 % | **-1.9 %** | 0.899 |
| 2 | 1 | 7.53 ms | 2.17 ms | -71.2 % | +174.0 % | 0.950 |
| 2 | 4 | 8.13 ms | 6.64 ms | -18.2 % | -21.3 % | 0.924 |
| 2 | 16 | 10.42 ms | 7.30 ms | -29.9 % | -12.7 % | 0.898 |
| 2 | 64 | 18.93 ms | 14.26 ms | -24.7 % | -13.7 % | 0.901 |
| 2 | 128 | 34.94 ms | 23.78 ms | -31.9 % | -10.0 % | 0.903 |

### The answer: throughput, not latency, is what speculation loses

Per-token latency improves at **every** point measured, from -75 % at c=1 to
-34 % at c=128. Speculation never stops helping latency.

Throughput is the opposite story. At TP=1 the gain decays monotonically
(+203 % -> +100 % -> +44 % -> +12.5 %) and **crosses zero between c=64 and
c=128**, where it becomes -1.9 %. At TP=2 it is already negative by c=4.

### And the mechanism is saturation, not prediction quality

This is the part the acceptance-rate column settles. The obvious hypothesis for
a vanishing speculation win is that the drafts stop being accepted, so the engine
pays for verification and throws the results away. **That is not what happens
here.** Acceptance is essentially flat across the whole concurrency ladder:
0.956 at c=1 and 0.899 at c=128, a decline of about six points while the
throughput benefit falls by 205 points.

So the drafts remain just as good. What changes is the value of the spare
capacity they were exploiting. At low concurrency decode is
memory-bandwidth-bound: the GPU reads the entire weight matrix to produce one
token per sequence and has idle arithmetic units, so verifying four draft tokens
in the same pass is nearly free and converts many thin steps into few fat ones.
At high concurrency batching has already filled those units with real work from
other requests, so the speculative verification is no longer free: it competes
directly, and every rejected draft token is compute taken from a request that
would have used it.

That also explains why TP=2 turns negative so much earlier. Splitting the model
across two GPUs halves the per-GPU weight-reading cost, so the machine reaches
compute-boundedness at a lower concurrency, and the window where speculation is
free closes sooner.

**Practical reading:** n-gram speculation is a latency optimisation for lightly
loaded serving, not a throughput optimisation for a busy fleet. On a saturated
replica it costs throughput while still improving per-request latency, which is a
real trade rather than a free win, and which of the two matters depends entirely
on whether the deployment is latency-bound or capacity-bound.

---

## 2026-09-02 — prefill scales linearly with input length; decode goes bandwidth-bound

The length sweep is the axis the P1 spec asked the writeup to answer on: where
does the workload cross from memory-bound to compute-bound? The right answer is
"it depends on which piece of the request you look at", and the length sweep
separates them by choosing configurations where one dominates.

Data below is from job `5174410`'s scan of the `length-sweep` sweep in
`results/raw/`, N=3 repetitions per (backend, TP, concurrency, input, output).
Same model (Qwen2.5-7B-Instruct), same seed, prefix caching off, one server per
(backend, TP), six length combinations as phases against a single server launch.

### Prefill: TTFT ~ linear in input length, and vLLM is faster than SGLang here

At c=1, TP=1, output_len fixed to isolate the prefill phase:

| input tokens | vLLM TTFT p50 (s) | sglang TTFT p50 (s) |
|---:|---:|---:|
|   128 | 0.0207 | 0.0266 |
|   512 | 0.0409 | 0.0459 |
|  1024 | 0.0714 | 0.0779 |
|  2048 | 0.1318 | 0.1380 |
|  4096 | 0.2674 | 0.2698 |
|  8192 | 0.5725 | 0.5683 |

A 64x rise in input length produces a 27x-28x rise in TTFT, so the slope on a
log-log plot is well under 1: prefill is not quite linear, likely because the
per-request fixed cost (scheduling, kernel launch, first block allocation) is
non-trivial at 128 tokens and washes out at 8192. Fitting `TTFT ~ a + b*L`,
the linear term dominates by L=1024. That is the operating regime this
benchmark targets.

vLLM leads at short inputs and SGLang catches up at long: at L=128 the ratio is
0.78x (vLLM 22% cheaper), at L=8192 it is 1.007x (statistical tie). The
crossover is where prefill cost is dominated by the matmul rather than the
scheduling path, which is where both engines are running the same underlying
kernels.

### Decode: memory-bandwidth-bound at c=1, compute-bound at c=64

At c=1 the per-token ITL is essentially flat with respect to the workload
shape: it depends only on tensor-parallel degree, because the machine is
reading the same weights every step regardless of what the prompt was.

| TP | vLLM ITL/token @ c=1 (ms) | sglang ITL/token @ c=1 (ms) |
|---:|---:|---:|
| 1 | 12.29 | 12.00 |
| 2 |  7.53 |  7.07 |
| 4 |  5.16 |  4.25 |

That is the memory-bandwidth signature. TP=1 reads the whole model each step,
TP=2 splits the weights across two GPUs so each reads half, TP=4 splits four
ways. The ITL falls with the number of GPUs sharing the weight-read cost, and
the ratio 12.29 / 5.16 = 2.38x is close to the theoretical 4x limit but pays
the intra-shard reduction cost each step. At c=1 the workload is
bandwidth-bound: the arithmetic units are idle, and adding tokens per step
(which is what higher concurrency does) is nearly free.

At c=64 ITL climbs sharply with input length even though the *decode workload*
per request is unchanged. That is the signature of a compute-bound decode:

| c=64, in tokens |  1024 |  2048 |  4096 |  8192 |
|---|---:|---:|---:|---:|
| vLLM TP=1 ITL (ms/tok)  | 31.9 | 54.3 | 142.8 | 152.5 |
| vLLM TP=4 ITL (ms/tok)  | 11.5 | 18.1 |  52.5 |  56.7 |

Adding input length forces prefill and decode to compete for the same tensor
cores in a continuous-batching scheduler: prefill of one request's prompt
interleaves with decode of another, and the batch's decode ITL absorbs
prefill cost that would otherwise have shown up only as TTFT. That is why the
"ITL vs concurrency" story from the specdec sweep and the "ITL vs input
length" story here are the same story: both are measuring where the batch
becomes compute-bound.

### The transition, quantitatively

Take TP=1, c=1, in=1024/out=256 as a memory-bandwidth-bound baseline: ITL is
12.29 ms/tok, throughput 79.6 tok/s. Now hold TP=1, in=1024, and raise
concurrency to 64: ITL rises to 31.9 ms/tok (2.6x) while output tok/s rises
to 1889 (23.7x). The GPU has fully absorbed the extra work: nine tenths of the
throughput gain came from batching, and one tenth was paid back as per-token
latency. That is exactly the memory-bandwidth-to-compute-bound transition the
project is meant to characterise.

At TP=4, the same transition costs more: c=1 ITL is 5.16 ms/tok, c=64 is
11.49 ms/tok (2.2x) at 5153 tok/s (27x more throughput). The
memory-bandwidth headroom is used up sooner because there was less of it: each
GPU is holding a quarter of the weights, so the "extra" arithmetic waiting to
be used is proportionally smaller. This is the mechanism behind the
speculative-decoding TP=2 result (see the section above): when the machine is
already compute-bound, speculation stops being free.

### Practical reading

If a deployment cares about **first-token latency**, TP scales it well and
prefill is the linear knob. Doubling input length roughly doubles TTFT beyond
1024 tokens, so a long-context service pays a proportional latency tax.

If it cares about **throughput per GPU-hour**, decode dominates the
end-to-end cost at balanced input/output ratios, and TP=4 with c>=64 is where
the machine spends its arithmetic budget. But every doubling of concurrency
beyond that point costs progressively more per-token ITL: the price of
capacity is p95 tail latency.

If it cares about **both**, the crossover point for this exact model+hardware
is around c=64 at TP=1 for in=1024/out=256, which is where the marginal ITL
cost starts to exceed the marginal throughput gain. That is a per-model
quantity: a bigger model shifts the memory bandwidth ceiling and moves the
crossover. Qwen2.5-32B at TP=4 sits in a different regime and the RESULTS.md
scaling section quantifies it.

---

## 2026-09-02 — TP=8 multi-node attempt: Ray is not in the vLLM container

Job `5174440` (`slurm/probe_ray.sh`) probed the pinned vLLM image (digest
recorded in `provision.lock.json`) for a Ray installation before committing
node-hours to a two-node run. Result:

```
=== ray version and import ===
  Traceback (most recent call last):
    File "<string>", line 1, in <module>
  ModuleNotFoundError: No module named 'ray'
=== ray start --head (single node self-test) ===
  /usr/bin/bash: line 3: ray: command not found
```

Ray is a required dependency for vLLM's multi-node backend
(`--distributed-executor-backend ray`), and the pinned vLLM 0.27.1 image on
this cluster does not include it. The alternative in-tree backend is `mp`,
which is single-node only (spawns local worker processes). So the standard
"just add a `--distributed-executor-backend ray` flag" recipe does not apply
here without infrastructure changes.

**Options considered and why none was taken now:**

1. **Rebuild the vLLM image with Ray.** Feasible but changes the pinned digest
   and therefore invalidates every earlier artifact's "same backend as X"
   invariant. Would need a new `backend_image_digest` recorded in every
   subsequent artifact, and the existing full-7b/32b sweeps could no longer
   be compared to it directly. This is the right long-term move but is out of
   scope for finishing P1.
2. **Install Ray into a side venv and shim it into the container via
   `--bind`.** Ray needs to be importable inside the same Python that runs
   vLLM's workers, which lives inside the container. Binding a venv in
   requires matching Python versions exactly and preserving ABI for compiled
   extensions (`ray._raylet`), which are fragile assumptions to bet six node
   hours on.
3. **Sglang multi-node.** Sglang has its own distributed-inference story via
   its own launcher, but the sglang 32B TP=4 runs all launch-failed on this
   cluster (see `results/RESULTS.md`), so it is a worse starting point for a
   multi-node measurement.

**Consequence for P1.** The one axis where the intra-node fabric is
different from the multi-node fabric is precisely TP=8 across two nodes, and
that is the one axis that goes unmeasured. Every intra-node TP<=4 point runs
over uniform NVLink `NV4`, so the current scaling curves cannot show an
interconnect effect at all, only kernel and per-replica-bandwidth effects.
This is honest and stated as such in the README's known-limitations section:
the TP=8 point remains an unmeasured absence rather than a placeholder or
extrapolation.

The IB fabric itself is present and reachable on the compute nodes (`ib0`
at 10.3.x/16 confirmed via probe), and the `slurm/run_tp8_ray.sh` script is
committed so a future attempt with a Ray-equipped image is a container swap
away rather than a redesign. **Cost of the probe: 0.005 node-hours; cost of
the fabricated data point that was not written: infinite.**

---

## 2026-09-02 — Definition-of-done status for P1

Checked against the workspace `CLAUDE.md` "Definition of done" for Project 1.

| CLAUDE.md requirement | Status | Evidence |
|---|---|---|
| Real measurements committed for both backends across the full sweep | **MET** | 4,573 raw artifacts across 11 sweeps; both vLLM and SGLang produced measured configs (see `results/RESULTS.md` inventory). Some SGLang configs launch-failed and are recorded as such (32B at TP=4). |
| N>=3 repetitions | **MET** | full-7b sweep uses N=5, length-sweep and specdec sweeps use N=3, packing measurement uses N=3. Warmup rep recorded and excluded. |
| TP scaling efficiency reported with a stated explanation for where efficiency falls off | **MET** | `results/RESULTS.md` reports parallel efficiency 0.78 at TP=2 and 0.69 at TP=4 for vLLM on 7B at c=1024, and 0.59 / 0.65 for SGLang at c=128. Both sections include the explicit "every intra-node pair is NVLink NV4, so the loss is not an interconnect effect — it is kernel efficiency, per-replica memory bandwidth, or scheduler overhead" note. |
| Every figure regenerable by one command | **MET** | `sbatch slurm/run_analysis.sh`. Job `5174365` regenerated all six figures + tables from raw in 7:25. |
| README states the specific hardware and topology | **MET** | README's "Hardware" section names the A100-SXM4-40GB / driver 595.71.05 / CUDA 13.2 / NVLink `NV4` topology / 2x 200 Gb/s IB explicitly. |
| Speculative-decoding section explicitly addresses where speculation stops helping and why | **MET** | `docs/notes.md` section "where speculative decoding stops helping, and why": throughput crosses zero between c=64 and c=128 at TP=1, and earlier at TP=2; acceptance rate stays flat (~0.9), so the mechanism is saturation, not prediction quality. |
| Backend adapter interface + vLLM + SGLang implementations | **MET** | `src/llm_inference_bench/backends/{base,vllm,sglang}.py`. Adding a third backend requires only a new adapter class per `base.py`. |
| YAML-driven sweep runner with Slurm job-array execution | **MET** | `configs/*.yaml` + `slurm/run_sweep.sh` (`--array=0-N`). Groups map to array tasks via `--group-index`. |
| ReFrame-style regression checks | **MET** | `analysis/regression_check.py`; baseline captured in job `5174402` (288 configurations); `baselines/p1.json` committed. `sbatch slurm/run_regression_check.sh --sweep-name full-7b` verifies. |
| Analysis producing: Pareto, TP efficiency, prefix caching, spec-dec, backend comparison | **MET** | `results/figures/{pareto_frontier,tp_scaling,prefix_caching,speculative_decoding,packing_interference,phase1_validation}.png`. |
| README identifies memory-bound → compute-bound transition | **MET** | `docs/notes.md` section "prefill scales linearly with input length; decode goes bandwidth-bound" (this session), linked from the README's headline findings. |

### Gaps and open items, stated honestly

- **TP=8 multi-node point is unmeasured.** Ray is not in the pinned vLLM
  image (probe job 5174440), so the standard multi-node recipe does not
  apply. `slurm/run_tp8_ray.sh` and `configs/tp8_multinode.yaml` are
  committed for the day a Ray-equipped image is available. The IB fabric
  itself is present and confirmed reachable.
- **SGLang prefix cache hit rate: resolved.** It first read 0 at every
  ratio because the adapter trusted `sglang:cache_hit_rate`, which reads 0
  even while caching works. SGLang 0.5.18 has no hits/queries pair; the rate
  is now derived as 1 - uncached/prompt tokens from
  `sglang:uncached_prompt_tokens_histogram_sum` and
  `sglang:prompt_tokens_total`, per-phase delta, recovered from the raw
  Prometheus text already stored in each artifact. It matches the bound as
  closely as vLLM does (`results/RESULTS.md`).
- **Dirty git tree on 2,794 of 4,573 artifacts.** Provenance is exact
  (SHA carries the `-dirty` suffix, image digest is pinned), but a clean-
  checkout replay of *those particular artifacts* would not be
  bit-reproducible. Post-2026-08-31 sweeps are on clean trees.

## Open questions

- Should the packing-interference finding be woven into a general "where
  packing might still be safe" figure (isolating trend-preserving comparisons)?
  Not yet done; the current story is "packing is not measurement-safe",
  which is the honest headline.

