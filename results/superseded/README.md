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
