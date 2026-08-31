#!/usr/bin/env python3
"""What co-locating four single-GPU runs on one node actually costs.

    python analysis/plot_packing_interference.py

Reads ``results/raw/``, writes ``results/figures/packing_interference.png``.

Pairs runs of an identical server group that differ only in placement
(``placement=isolated`` vs ``placement=packed``), and reports the delta per
metric. Only the ``packing_role=measured`` replica of the packed phase is
compared; the other three exist to create contention and are excluded.

The number this produces decides how the rest of the packed data may be used.
A small delta means packed runs can carry headline numbers and the sweep gets a
4x budget multiplier. A large one means packed runs support trend claims only,
and headline configurations have to be re-run isolated. Either answer is
reportable; assuming the first without measuring is not.

Untested until real data exists, per the non-negotiable rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402
from style import (  # noqa: E402
    SERIES,
    SURFACE,
    TEXT_SECONDARY,
    apply_style,
    backend_string,
    hardware_string,
    provenance_caption,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS_RAW = REPO / "results" / "raw"
FIGURES = REPO / "results" / "figures"

# Lower is better for the first three, higher is better for throughput. Kept
# explicit so the sign of the reported delta is never ambiguous.
METRICS = [
    ("ttft_s_p50", "TTFT p50 (s)", "lower"),
    ("ttft_s_p95", "TTFT p95 (s)", "lower"),
    ("itl_s_p50", "ITL p50 (s/token)", "lower"),
    ("output_tokens_per_s", "Output throughput (tok/s)", "higher"),
]


def main() -> int:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not RESULTS_RAW.exists() or not any(RESULTS_RAW.glob("*.json")):
        print(f"no raw artifacts in {RESULTS_RAW}; run the packing measurement first.")
        return 1

    frame, report = load(RESULTS_RAW, include_warmup=False, include_failed=True)
    print(report.summary())

    ok = frame[frame["status"] == "ok"]
    isolated = ok[ok["placement"] == "isolated"]
    packed = ok[(ok["placement"] == "packed") & (ok["packing_role"] == "measured")]

    if isolated.empty or packed.empty:
        print(
            "\nneed both placement=isolated and placement=packed "
            f"(measured role) runs; found {len(isolated)} isolated, {len(packed)} packed.\n"
            "Run: sbatch slurm/measure_packing_interference.sh <config> <group-index>"
        )
        return 1

    # Pair on the configuration, so only placement differs between the two sides.
    keys = ["group_label", "phase_label"]
    shared = sorted(
        set(map(tuple, isolated[keys].values)) & set(map(tuple, packed[keys].values))
    )
    if not shared:
        print("no configuration was run in both placements; nothing comparable.")
        return 1

    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.0 * len(METRICS), 3.6))
    summary_lines: list[str] = []

    for ax, (metric, ylabel, better) in zip(axes, METRICS, strict=True):
        iso_vals, pk_vals, labels = [], [], []
        for group_label, phase_label in shared:
            sel = (isolated["group_label"] == group_label) & (
                isolated["phase_label"] == phase_label
            )
            sel2 = (packed["group_label"] == group_label) & (
                packed["phase_label"] == phase_label
            )
            a = isolated.loc[sel, metric].dropna()
            b = packed.loc[sel2, metric].dropna()
            if a.empty or b.empty:
                continue
            iso_vals.append(a.tolist())
            pk_vals.append(b.tolist())
            labels.append(phase_label)

        if not iso_vals:
            ax.set_visible(False)
            continue

        x = np.arange(len(labels))
        width = 0.36
        for offset, values, color, name in (
            (-width / 2, iso_vals, SERIES[0], "isolated"),
            (+width / 2, pk_vals, SERIES[1], "packed"),
        ):
            medians = [float(np.median(v)) for v in values]
            lows = [min(v) for v in values]
            highs = [max(v) for v in values]
            ax.bar(
                x + offset,
                medians,
                width,
                label=name,
                color=color,
                edgecolor=SURFACE,
                linewidth=2,  # 2px surface gap between adjacent bars
            )
            ax.errorbar(
                x + offset, medians,
                yerr=[np.array(medians) - np.array(lows), np.array(highs) - np.array(medians)],
                fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3,
            )
            # Individual repetitions, so N=3 is shown rather than summarised away.
            for xi, v in zip(x + offset, values, strict=True):
                ax.plot([xi] * len(v), v, linestyle="none", marker="_",
                        markersize=7, color=TEXT_SECONDARY, alpha=0.7)

        for i, (a, b) in enumerate(zip(iso_vals, pk_vals, strict=True)):
            ma, mb = float(np.median(a)), float(np.median(b))
            if ma == 0:
                continue
            delta_pct = (mb - ma) / ma * 100.0
            worse = delta_pct > 0 if better == "lower" else delta_pct < 0
            summary_lines.append(
                f"{metric} @ {labels[i]}: {delta_pct:+.1f}% "
                f"({'worse' if worse else 'better'} when packed)"
            )
            ax.annotate(
                f"{delta_pct:+.1f}%",
                xy=(i, max(max(a), max(b))),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=7.5, color=TEXT_SECONDARY,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split(' (')[0]} ({better} is better)", loc="left")

    axes[0].legend(loc="upper left")

    first = ok.iloc[0]
    # Repetitions PER CONFIGURATION, not total rows. Counting rows conflates
    # repetitions with phases and overstates N in the caption, which is a
    # provenance error however small it looks: a reader sizing confidence off
    # "N=6" would be reading 3 repetitions across 2 phases.
    n_reps = int(
        min(
            isolated.groupby(["group_label", "phase_label"]).size().min(),
            packed.groupby(["group_label", "phase_label"]).size().min(),
        )
    )
    provenance_caption(
        fig,
        hardware=hardware_string(first),
        model=f"{first.get('model_id')} @ {str(first.get('model_revision'))[:12]}",
        backend=backend_string(first),
        repetitions=n_reps,
        extra="isolated = 1 replica on GPU 0, 3 GPUs idle; "
              "packed = 4 NUMA-pinned replicas, GPU 0 measured",
    )
    fig.suptitle(
        "Cost of packing four single-GPU runs onto one billed node",
        x=0.0, ha="left", fontsize=11,
    )

    out = FIGURES / "packing_interference.png"
    fig.savefig(out)
    print(f"\nwrote {out}")
    print("\ndeltas (packed relative to isolated):")
    for line in summary_lines:
        print(f"  {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
