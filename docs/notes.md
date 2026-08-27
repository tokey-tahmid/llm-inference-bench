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
