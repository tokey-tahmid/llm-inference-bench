#!/usr/bin/env python3
"""ReFrame-style performance regression check against a stored baseline.

    python analysis/regression_check.py --update-baseline   # capture current state
    python analysis/regression_check.py                     # check, non-zero on regression

Baselines live in ``baselines/<name>.json``, are committed, and are updated only
deliberately. Exit codes: 0 pass, 1 regression, 2 baseline missing or unusable.

The hard part of a performance regression check is not comparing numbers, it is
**not firing on noise**. A check that cries wolf gets disabled, at which point it
protects nothing. Three things keep this one honest:

1. **The tolerance is derived from the baseline's own measured spread**, not
   picked out of the air. A configuration whose repetitions ranged +/-11% cannot
   support a 5% threshold, and one that ranged +/-0.1% should not need a 20% one.
   The threshold is ``max(relative_spread * SPREAD_MULTIPLE, MIN_TOLERANCE)``.
2. **Only degradation is flagged.** An improvement is reported for information
   but never fails the check.
3. **A missing configuration is a failure, not a pass.** Silently skipping a
   configuration that vanished from the sweep is how a regression check quietly
   stops covering the thing that broke.

Metrics are compared per (backend, model, TP, phase), with direction handled
explicitly so the sign of a regression is never ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS_RAW = REPO / "results" / "raw"
BASELINES = REPO / "baselines"

# (column, human name, "lower"|"higher" is better)
METRICS: list[tuple[str, str, str]] = [
    ("output_tokens_per_s", "output throughput", "higher"),
    ("ttft_s_p50", "TTFT p50", "lower"),
    ("ttft_s_p95", "TTFT p95", "lower"),
    ("itl_s_p50", "ITL p50", "lower"),
    ("e2e_latency_s_p95", "e2e p95", "lower"),
]

KEY_COLUMNS = [
    "backend",
    "model_id",
    "tensor_parallel_size",
    "enable_prefix_caching",
    "phase_label",
]

# A regression must clear the configuration's own measured noise by this factor
# before it counts. 1.5x the observed spread is deliberately forgiving: the cost
# of a false alarm (the check gets ignored) exceeds the cost of catching a small
# regression one commit later.
SPREAD_MULTIPLE = 1.5
# Floor, for configurations whose spread is tiny. Nothing below this is treated
# as a regression regardless of how quiet the baseline was.
MIN_TOLERANCE = 0.05


def _key(row: dict[str, Any]) -> str:
    return "|".join(str(row.get(c)) for c in KEY_COLUMNS)


def build_baseline(sweep_name: str | None) -> dict[str, Any]:
    frame, report = load(RESULTS_RAW, include_warmup=False, include_failed=False)
    print(report.summary())
    ok = frame[frame["status"] == "ok"]
    if sweep_name:
        ok = ok[ok["sweep_name"] == sweep_name]
    if ok.empty:
        raise SystemExit("no successful runs to baseline")

    entries: dict[str, Any] = {}
    for keys, grp in ok.groupby(KEY_COLUMNS, dropna=False):
        row = dict(zip(KEY_COLUMNS, keys, strict=True))
        entry: dict[str, Any] = {"key_fields": row, "n_repetitions": int(len(grp))}
        for col, _name, _dir in METRICS:
            vals = grp[col].dropna()
            if vals.empty:
                continue
            median = float(vals.median())
            lo, hi = float(vals.min()), float(vals.max())
            # Relative spread of the baseline itself, which sets the tolerance.
            spread = (hi - lo) / median if median else 0.0
            entry[col] = {"median": median, "min": lo, "max": hi, "rel_spread": spread}
        entries[_key(row)] = entry

    first = ok.iloc[0]
    return {
        "captured_utc": str(first.get("utc_timestamp")),
        "git_sha": str(first.get("git_sha")),
        "gpu_model": str(first.get("gpu_model")),
        "driver_version": str(first.get("driver_version")),
        "backend_versions": sorted(
            {f"{b}:{v}" for b, v in zip(ok["backend"], ok["backend_version"], strict=True)}
        ),
        "sweep_name": sweep_name,
        "entries": entries,
    }


def check(baseline: dict[str, Any], sweep_name: str | None) -> int:
    frame, _ = load(RESULTS_RAW, include_warmup=False, include_failed=False)
    ok = frame[frame["status"] == "ok"]
    if sweep_name:
        ok = ok[ok["sweep_name"] == sweep_name]

    # Environment drift makes a comparison meaningless rather than merely noisy,
    # so it is reported prominently instead of being folded into the tolerance.
    if not ok.empty:
        cur_gpu = str(ok.iloc[0].get("gpu_model"))
        cur_drv = str(ok.iloc[0].get("driver_version"))
        if cur_gpu != baseline.get("gpu_model"):
            print(f"WARNING hardware changed: {baseline.get('gpu_model')} -> {cur_gpu}")
        if cur_drv != baseline.get("driver_version"):
            print(f"WARNING driver changed: {baseline.get('driver_version')} -> {cur_drv}")

    current: dict[str, Any] = {}
    for keys, grp in ok.groupby(KEY_COLUMNS, dropna=False):
        row = dict(zip(KEY_COLUMNS, keys, strict=True))
        current[_key(row)] = grp

    regressions: list[str] = []
    improvements: list[str] = []
    missing: list[str] = []
    compared = 0

    for key, entry in baseline["entries"].items():
        grp = current.get(key)
        if grp is None:
            missing.append(key)
            continue
        for col, name, direction in METRICS:
            base = entry.get(col)
            if not base:
                continue
            vals = grp[col].dropna()
            if vals.empty:
                continue
            compared += 1
            now = float(vals.median())
            ref = base["median"]
            if ref == 0:
                continue
            tol = max(base.get("rel_spread", 0.0) * SPREAD_MULTIPLE, MIN_TOLERANCE)
            delta = (now - ref) / ref
            worse = -delta if direction == "higher" else delta
            if worse > tol:
                regressions.append(
                    f"{key} :: {name}  {ref:.4g} -> {now:.4g}  "
                    f"({delta * 100:+.1f}%, tolerance {tol * 100:.1f}%)"
                )
            elif -worse > tol:
                improvements.append(
                    f"{key} :: {name}  {ref:.4g} -> {now:.4g} ({delta * 100:+.1f}%)"
                )

    print(f"\ncompared {compared} metric(s) across {len(baseline['entries'])} configurations")
    if improvements:
        print(f"\nIMPROVED ({len(improvements)}), informational only:")
        for line in improvements[:20]:
            print(f"  {line}")
    if missing:
        print(f"\nMISSING from current results ({len(missing)}):")
        for key in missing[:20]:
            print(f"  {key}")
        print("  A configuration that vanished is a failure: the check no longer covers it.")
    if regressions:
        print(f"\nREGRESSIONS ({len(regressions)}):")
        for line in regressions:
            print(f"  {line}")

    if regressions or missing:
        print("\nFAIL")
        return 1
    print("\nPASS: no configuration regressed beyond its measured noise")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=BASELINES / "p1.json")
    ap.add_argument("--sweep-name", type=str, default=None,
                    help="restrict to one sweep, e.g. full-7b")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()

    if args.update_baseline:
        data = build_baseline(args.sweep_name)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"wrote baseline with {len(data['entries'])} configurations to {args.baseline}")
        return 0

    if not args.baseline.exists():
        print(f"no baseline at {args.baseline}; create one with --update-baseline")
        return 2
    return check(json.loads(args.baseline.read_text()), args.sweep_name)


if __name__ == "__main__":
    raise SystemExit(main())
