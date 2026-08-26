#!/usr/bin/env python3
"""Phase-1 validation figure: proves the pipeline, does not report a result.

    python analysis/plot_phase1.py

Reads ``results/raw/`` and writes ``results/figures/phase1_validation.png``.

This script is deliberately **untested until real data exists**. Per the
non-negotiable rules, no sample or seeded results file is created to exercise it;
it is written against the documented artifact schema and run for the first time
against the output of a real sweep. Expect to fix something the first time.

What phase 1 is checking, in order of what would invalidate everything downstream:

1. artifacts parse and carry complete provenance
2. warmup repetitions are present and are excluded from analysis
3. output lengths came back exactly as requested (``ignore_eos`` honoured)
4. prompt token counts round-tripped exactly (integer-token prompts honoured)
5. TTFT and throughput move in the expected directions between the two
   concurrency points, which is the weakest check here and only a smoke test
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402
from style import (  # noqa: E402
    STATUS_CRITICAL,
    apply_style,
    backend_string,
    hardware_string,
    label_line_ends,
    plot_with_spread,
    provenance_caption,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS_RAW = REPO / "results" / "raw"
FIGURES = REPO / "results" / "figures"


def check_integrity(frame, report) -> list[str]:
    """Conditions that would make the measurements untrustworthy.

    Returned as a list of complaints rather than raised, so every problem is
    reported in one pass instead of one per run.
    """
    problems: list[str] = []

    if report.warmup_excluded == 0:
        problems.append(
            "no warmup repetitions found: either warmup_repetitions was 0, or the "
            "first repetition of each config is being reported as a measurement"
        )
    if report.dirty_git_runs:
        problems.append(
            f"{report.dirty_git_runs} run(s) came from a dirty git tree; these are "
            "not reproducible from a clean checkout"
        )
    if report.unparseable:
        problems.append(f"{len(report.unparseable)} artifact(s) failed to parse")

    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        problems.append("no successful runs at all")
        return problems

    if not ok["output_length_exact"].fillna(False).all():
        problems.append(
            "output lengths did not match what was requested: ignore_eos was not "
            "honoured, so the output-length axis is NOT controlled and no "
            "cross-configuration comparison is valid"
        )
    if "prompt_length_exact" in ok and not ok["prompt_length_exact"].fillna(True).all():
        problems.append(
            "prompt token counts did not round-trip: integer-token prompts are being "
            "re-interpreted, so input-length and prefix-sharing axes are invalid"
        )
    if not ok["cache_reset_ok"].fillna(False).all():
        problems.append(
            "prefix cache reset failed between phases: later phases may have "
            "inherited a warm cache from earlier ones"
        )
    if ok["client_may_be_bottleneck"].fillna(False).any():
        problems.append("at least one phase may have been limited by the load client")

    failed = frame[frame["status"] != "ok"]
    if not failed.empty:
        counts = failed["status"].value_counts().to_dict()
        problems.append(f"non-ok runs present (recorded, not dropped): {counts}")

    return problems


def main() -> int:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not RESULTS_RAW.exists() or not any(RESULTS_RAW.glob("*.json")):
        print(
            f"no raw artifacts in {RESULTS_RAW}.\n"
            "Nothing has been measured yet, so there is nothing to plot. "
            "Run: sbatch slurm/run_sweep.sh configs/phase1_validation.yaml"
        )
        return 1

    frame, report = load(RESULTS_RAW, include_warmup=False, include_failed=True)
    print(report.summary())
    print()

    problems = check_integrity(frame, report)
    if problems:
        print("INTEGRITY PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        print()

    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        print("no successful runs; not plotting.")
        return 1

    # Aggregate across repetitions per concurrency point. Median with min/max and
    # the individual repetitions retained, never a bare mean.
    points = []
    for conc, sub in ok.groupby("concurrency", dropna=False):
        points.append(
            {
                "concurrency": conc,
                "n": len(sub),
                "ttft_reps": sub["ttft_s_p50"].dropna().tolist(),
                "tput_reps": sub["output_tokens_per_s"].dropna().tolist(),
                "itl_reps": sub["itl_s_p50"].dropna().tolist(),
            }
        )
    points.sort(key=lambda p: p["concurrency"])
    x = [p["concurrency"] for p in points]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    panels = [
        ("ttft_reps", "TTFT p50 (s)", "Time to first token"),
        ("itl_reps", "ITL p50 (s/token)", "Inter-token latency"),
        ("tput_reps", "Output throughput (tokens/s)", "Throughput"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels, strict=True):
        median = [sorted(p[key])[len(p[key]) // 2] if p[key] else float("nan") for p in points]
        lo = [min(p[key]) if p[key] else float("nan") for p in points]
        hi = [max(p[key]) if p[key] else float("nan") for p in points]
        plot_with_spread(
            ax, x, median, lo, hi,
            index=0,
            label="vLLM TP=1",
            reps=[p[key] for p in points],
        )
        ax.set_xlabel("Concurrency (in-flight requests)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        if median and median[-1] == median[-1]:  # not NaN
            label_line_ends(ax, x[-1], median[-1], "TP=1", 0)

    # A single series needs no legend box: the title and direct label name it.

    first = ok.iloc[0]
    n_reps = int(min(p["n"] for p in points)) if points else 0
    provenance_caption(
        fig,
        hardware=hardware_string(first),
        model=f"{first.get('model_id')} @ {str(first.get('model_revision'))[:12]}",
        backend=backend_string(first),
        repetitions=n_reps,
        extra=f"input {first.get('input_len')} tok, output {first.get('output_len')} tok, "
              "warmup excluded",
    )
    fig.suptitle(
        "Phase-1 pipeline validation (not a performance result)",
        x=0.0, ha="left", fontsize=11,
    )

    if problems:
        fig.text(
            0.0, -0.13,
            "INTEGRITY WARNINGS: " + "; ".join(problems),
            ha="left", va="top", fontsize=6.5, color=STATUS_CRITICAL, wrap=True,
        )

    out = FIGURES / "phase1_validation.png"
    fig.savefig(out)
    print(f"wrote {out}")

    # A machine-readable companion, so the README can quote numbers that trace
    # back to artifacts without anyone transcribing them by hand.
    table = FIGURES / "phase1_validation.csv"
    ok.to_csv(table, index=False)
    print(f"wrote {table}")
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
