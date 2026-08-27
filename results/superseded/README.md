# Superseded raw artifacts

Real runs, moved out of `results/raw/` because they are known-defective in ways
their own provenance records. Nothing here was edited or deleted: the files are
byte-identical to what the runner wrote, and they are kept so the defects stay
auditable.

## 2026-08-26, phase-1 validation, 8 artifacts (sweep job 5148060)

The first end-to-end pipeline validation. It succeeded as a validation (128/128
requests ok, warmup correctly excluded, `output_length_exact` and
`prompt_length_exact` both true) and is superseded as data for two reasons it
recorded about itself:

1. `git_sha` ends in `-dirty`. The tree had uncommitted changes, so these runs
   are not reproducible from a clean checkout.
2. `cache_reset_ok: false`, HTTP 404. vLLM 0.27.1 only mounts
   `/reset_prefix_cache` under `VLLM_SERVER_DEV_MODE`, so the reset between
   phases silently did nothing and the second phase inherited the first phase's
   cache state. Harmless here (prefix caching was off for the whole sweep) but
   invalid as a template.

Both are fixed; the replacement run is in `results/raw/`.

## 2026-08-27, phase-1 re-run, 8 artifacts (sweep job 5150758)

Both defects from the previous batch were confirmed fixed in these artifacts
(`cache_reset_ok: true` with HTTP 200, `backend_version: 0.27.1`), and
`output_length_exact` / `prompt_length_exact` were true throughout.

Superseded for one reason, and it was a process error rather than a code defect:
the repo was edited *while the sweep was running*, so `git_sha` again ends in
`-dirty` even though the code that actually ran was the clean tree at `a715573`.

That exposed a genuine provenance flaw, now fixed: `git_sha` was re-checked at
every artifact write rather than sampled once. A long sweep could therefore be
stamped dirty by an edit to a completely unrelated file. It is now cached at
process start, which is the honest answer to "what produced this", since Python
imported the harness once at launch.

The discipline that follows: do not touch the repo while a sweep is in flight.
