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

- Backend versions for both images are still **unread**. Until the fixed probe
  runs, the lockfile has image digests (which are exact and sufficient for
  reproducibility) but no human-readable version string.
- Whether both engines accept integer-token prompts on `/v1/completions` is
  asserted by design but not yet verified on hardware. `validate_prompt_token_fidelity`
  runs before any measurement and records the answer; if either engine fails it,
  the input-length and prefix-sharing axes need rethinking for that backend.
- Whether `ignore_eos` is honoured by both engines at the top level of a
  completions request. `compute_metrics` flags any mismatch via
  `output_length_exact`, so this cannot pass silently.
