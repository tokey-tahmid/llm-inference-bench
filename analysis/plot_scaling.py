#!/usr/bin/env python3
"""Tensor-parallel scaling: speedup, parallel efficiency, strong and weak.

    python analysis/plot_scaling.py

Reads ``results/raw/``, writes ``results/figures/tp_scaling.png`` and
``tp_scaling.csv``.

Treated with the same rigour as a node-scaling study, because it is one.
Efficiency curves, not raw throughput bars.

Definitions used, stated explicitly because "efficiency" is overloaded:

* **Strong scaling.** Fixed total offered load, increasing TP degree.
  ``speedup(N) = throughput(N) / throughput(1)`` and
  ``efficiency(N) = speedup(N) / N``. Efficiency of 1.0 means N GPUs did N times
  the work; below 1.0 the extra GPUs are not paying for themselves.
* **Weak scaling.** Load grows with TP so per-GPU work is constant.
  ``efficiency(N) = throughput(N) / (N * throughput(1))`` on the per-GPU points.
  This asks whether N GPUs serve N times as many concurrent users at the same
  latency, which is the capacity-planning question rather than the latency one.

One caveat is stamped on the figure rather than left to the reader: on this
hardware every GPU pair is joined by NVLink (`NV4`), with no intra-node PCIe
path. So a fall in efficiency at TP<=4 is *not* an interconnect-topology effect.
It is kernel efficiency, memory bandwidth per replica, or scheduler overhead.
Only the multi-node TP=8 point can show an interconnect effect at all.

Untested until real data exists, per the non-negotiable rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402
from style import (  # noqa: E402
    GRID,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_style,
    backend_string,
    hardware_string,
    label_line_ends,
    plot_with_spread,
    provenance_caption,
    series_style,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS_RAW = REPO / "results" / "raw"
FIGURES = REPO / "results" / "figures"


def _median_by(frame: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    """Median with min/max across repetitions, plus the repetition count."""
    g = frame.groupby(keys, dropna=False)[metric]
    out = g.agg(["median", "min", "max", "count"]).reset_index()
    return out.rename(columns={"count": "n_reps"})


def main() -> int:
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)

    if not RESULTS_RAW.exists() or not any(RESULTS_RAW.glob("*.json")):
        print(f"no raw artifacts in {RESULTS_RAW}")
        return 1

    frame, report = load(RESULTS_RAW, include_warmup=False, include_failed=True)
    print(report.summary())
    ok = frame[frame["status"] == "ok"].copy()
    if ok.empty:
        print("no successful runs")
        return 1

    # Prefix caching off only: mixing it in would confound a cache effect with a
    # parallelism effect, and this figure is about parallelism.
    ok = ok[~ok["enable_prefix_caching"].fillna(False)]
    if ok.empty:
        print("no prefix-caching-off runs to build a scaling curve from")
        return 1

    strong = ok[~ok["phase_label"].astype(str).str.startswith("weak_")]
    weak = ok[ok["phase_label"].astype(str).str.startswith("weak_")]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    caveats: list[str] = []

    # ---- panel 1: strong scaling speedup -----------------------------------
    ax = axes[0]
    plotted = 0
    for i, (backend, sub) in enumerate(sorted(strong.groupby("backend"))):
        med = _median_by(sub, ["tensor_parallel_size", "concurrency"], "output_tokens_per_s")
        # Use the highest concurrency present for every TP degree, so the
        # comparison is at matched, saturating load rather than at a point that
        # only some TP degrees reached.
        common = set.intersection(
            *[
                set(med[med["tensor_parallel_size"] == tp]["concurrency"])
                for tp in med["tensor_parallel_size"].unique()
            ]
        ) if med["tensor_parallel_size"].nunique() else set()
        if not common:
            continue
        conc = max(common)
        at = med[med["concurrency"] == conc].sort_values("tensor_parallel_size")
        if at.empty or 1 not in set(at["tensor_parallel_size"]):
            caveats.append(f"{backend}: no TP=1 baseline, speedup not computable")
            continue
        base = float(at[at["tensor_parallel_size"] == 1]["median"].iloc[0])
        tps = at["tensor_parallel_size"].tolist()
        speedup = (at["median"] / base).tolist()
        lo = (at["min"] / base).tolist()
        hi = (at["max"] / base).tolist()
        plot_with_spread(ax, tps, speedup, lo, hi, index=i, label=f"{backend} (c={conc:g})")
        label_line_ends(ax, tps[-1], speedup[-1], backend, i)
        plotted += 1

    ideal = [1, 2, 4]
    ax.plot(ideal, ideal, linestyle=":", linewidth=1.2, color=TEXT_MUTED, zorder=0)
    ax.annotate("ideal", xy=(ideal[-1], ideal[-1]), xytext=(-2, 6),
                textcoords="offset points", fontsize=7.5, color=TEXT_MUTED, ha="right")
    ax.set_xlabel("Tensor-parallel degree")
    ax.set_ylabel("Speedup vs TP=1")
    ax.set_title("Strong scaling: speedup", loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ideal)
    ax.set_xticklabels([str(t) for t in ideal])
    if plotted > 1:
        ax.legend(loc="upper left")

    # ---- panel 2: strong scaling parallel efficiency ------------------------
    ax = axes[1]
    for i, (backend, sub) in enumerate(sorted(strong.groupby("backend"))):
        med = _median_by(sub, ["tensor_parallel_size", "concurrency"], "output_tokens_per_s")
        common = set.intersection(
            *[
                set(med[med["tensor_parallel_size"] == tp]["concurrency"])
                for tp in med["tensor_parallel_size"].unique()
            ]
        ) if med["tensor_parallel_size"].nunique() else set()
        if not common:
            continue
        conc = max(common)
        at = med[med["concurrency"] == conc].sort_values("tensor_parallel_size")
        if at.empty or 1 not in set(at["tensor_parallel_size"]):
            continue
        base = float(at[at["tensor_parallel_size"] == 1]["median"].iloc[0])
        tps = np.array(at["tensor_parallel_size"], dtype=float)
        eff = (at["median"].to_numpy() / base) / tps
        lo = (at["min"].to_numpy() / base) / tps
        hi = (at["max"].to_numpy() / base) / tps
        plot_with_spread(ax, tps.tolist(), eff.tolist(), lo.tolist(), hi.tolist(),
                         index=i, label=backend)
        label_line_ends(ax, tps[-1], eff[-1], backend, i)

    ax.axhline(1.0, linestyle=":", linewidth=1.2, color=TEXT_MUTED, zorder=0)
    ax.set_xlabel("Tensor-parallel degree")
    ax.set_ylabel("Parallel efficiency (speedup / N)")
    ax.set_title("Strong scaling: efficiency", loc="left")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ideal)
    ax.set_xticklabels([str(t) for t in ideal])
    ax.set_ylim(0, 1.25)

    # ---- panel 3: weak scaling efficiency -----------------------------------
    ax = axes[2]
    if weak.empty:
        ax.text(0.5, 0.5, "no weak-scaling points yet", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY)
        ax.set_axis_off()
    else:
        for i, (backend, sub) in enumerate(sorted(weak.groupby("backend"))):
            med = _median_by(sub, ["tensor_parallel_size"], "output_tokens_per_s")
            med = med.sort_values("tensor_parallel_size")
            if 1 not in set(med["tensor_parallel_size"]):
                caveats.append(f"{backend}: no TP=1 weak-scaling baseline")
                continue
            base = float(med[med["tensor_parallel_size"] == 1]["median"].iloc[0])
            tps = np.array(med["tensor_parallel_size"], dtype=float)
            eff = med["median"].to_numpy() / (base * tps)
            lo = med["min"].to_numpy() / (base * tps)
            hi = med["max"].to_numpy() / (base * tps)
            plot_with_spread(ax, tps.tolist(), eff.tolist(), lo.tolist(), hi.tolist(),
                             index=i, label=backend)
            label_line_ends(ax, tps[-1], eff[-1], backend, i)
        ax.axhline(1.0, linestyle=":", linewidth=1.2, color=TEXT_MUTED, zorder=0)
        ax.set_xlabel("Tensor-parallel degree")
        ax.set_ylabel("Weak-scaling efficiency")
        ax.set_title("Weak scaling: constant load per GPU", loc="left")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ideal)
        ax.set_xticklabels([str(t) for t in ideal])
        ax.set_ylim(0, 1.25)

    first = ok.iloc[0]
    n_reps = int(ok.groupby(["group_label", "phase_label"]).size().min())
    provenance_caption(
        fig,
        hardware=hardware_string(first),
        model=f"{first.get('model_id')} @ {str(first.get('model_revision'))[:12]}",
        backend=backend_string(first),
        repetitions=n_reps,
        extra="prefix caching OFF · all GPU pairs NVLink (NV4), so efficiency loss "
              "at TP<=4 is NOT an interconnect-topology effect",
    )
    fig.suptitle("Tensor-parallel scaling", x=0.0, ha="left", fontsize=11)

    if caveats:
        fig.text(0.0, -0.14, "CAVEATS: " + "; ".join(sorted(set(caveats))),
                 ha="left", va="top", fontsize=6.5, color=TEXT_SECONDARY, wrap=True)

    out = FIGURES / "tp_scaling.png"
    fig.savefig(out)
    print(f"wrote {out}")

    table = _median_by(
        ok, ["backend", "model_id", "tensor_parallel_size", "phase_label", "concurrency"],
        "output_tokens_per_s",
    )
    table.to_csv(FIGURES / "tp_scaling.csv", index=False)
    print(f"wrote {FIGURES / 'tp_scaling.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
