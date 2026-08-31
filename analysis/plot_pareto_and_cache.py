#!/usr/bin/env python3
"""Throughput/latency frontier, prefix-caching effect, and speculative decoding.

    python analysis/plot_pareto_and_cache.py

Reads ``results/raw/``, writes three figures into ``results/figures/``:

* ``pareto_frontier.png`` -- throughput against p95 latency across the
  concurrency ladder. Each curve is one configuration swept over concurrency, so
  the frontier is traced rather than asserted. Drawn as connected lines because
  the connection *is* the sweep, and because that keeps the palette on its
  validated adjacent-pair footing.
* ``prefix_caching.png`` -- TTFT and measured cache hit rate against
  shared-prefix ratio, caching on vs off. The theoretical hit-rate bound is drawn
  alongside the measured rate: the gap between them is eviction and
  block-granularity loss, and without the bound there is nothing to attribute the
  measured number against.
* ``speculative_decoding.png`` -- the question the definition of done asks
  directly: where does speculation stop helping, and why? Acceptance rate is
  plotted beside the latency effect so a vanishing win can be attributed to
  falling acceptance or to batch saturation rather than guessed at.

Untested until real data exists, per the non-negotiable rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402
from style import (  # noqa: E402
    TEXT_MUTED,
    TEXT_SECONDARY,
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


def _agg(frame: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    g = frame.groupby(keys, dropna=False)[metric]
    return g.agg(["median", "min", "max", "count"]).reset_index()


def _caption(fig, ok, extra: str) -> None:
    first = ok.iloc[0]
    n = int(ok.groupby(["group_label", "phase_label"]).size().min())
    provenance_caption(
        fig,
        hardware=hardware_string(first),
        model=f"{first.get('model_id')} @ {str(first.get('model_revision'))[:12]}",
        backend=backend_string(first),
        repetitions=n,
        extra=extra,
    )


# --------------------------------------------------------------------------
def figure_pareto(ok: pd.DataFrame) -> bool:
    sub = ok[(~ok["enable_prefix_caching"].fillna(False))
             & (ok["load_mode"] == "closed_loop")
             & (~ok["phase_label"].astype(str).str.startswith("weak_"))]
    sub = sub[sub["shared_prefix_ratio"].fillna(0.0) == 0.0]
    if sub.empty:
        print("pareto: no data")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for ax, (metric, ylabel, title) in zip(
        axes,
        [("e2e_latency_s_p95", "End-to-end p95 latency (s)", "Request latency"),
         ("ttft_s_p95", "TTFT p95 (s)", "Time to first token")],
        strict=True,
    ):
        idx = 0
        for (backend, tp), grp in sorted(sub.groupby(["backend", "tensor_parallel_size"])):
            med = _agg(grp, ["concurrency"], metric).sort_values("concurrency")
            tput = _agg(grp, ["concurrency"], "output_tokens_per_s").sort_values("concurrency")
            if med.empty:
                continue
            if idx > 3:
                break  # palette is four slots; facet rather than cycle hues
            plot_with_spread(
                ax,
                tput["median"].tolist(), med["median"].tolist(),
                med["min"].tolist(), med["max"].tolist(),
                index=idx, label=f"{backend} TP={tp:g}",
            )
            label_line_ends(ax, tput["median"].iloc[-1], med["median"].iloc[-1],
                            f"TP{tp:g}", idx)
            # Annotate the concurrency at each end, so the sweep direction is
            # readable without a third encoding.
            for pos in (0, -1):
                ax.annotate(f"c={med['concurrency'].iloc[pos]:g}",
                            xy=(tput["median"].iloc[pos], med["median"].iloc[pos]),
                            xytext=(0, -11), textcoords="offset points",
                            fontsize=6.5, color=TEXT_MUTED, ha="center")
            idx += 1
        ax.set_xlabel("Output throughput (tokens/s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}: up and left is better", loc="left")
        ax.legend(loc="upper left")

    _caption(fig, sub, "prefix caching OFF, no shared prefix · each curve swept over "
                       "concurrency 1->128")
    fig.suptitle("Throughput / latency frontier", x=0.0, ha="left", fontsize=11)
    out = FIGURES / "pareto_frontier.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return True


# --------------------------------------------------------------------------
def figure_prefix_caching(ok: pd.DataFrame) -> bool:
    sub = ok[ok["shared_prefix_ratio"].notna()]
    if sub.empty or sub["enable_prefix_caching"].nunique() < 2:
        print("prefix caching: need both on and off arms")
        return False

    # Pin every axis except the one under study.
    #
    # Without this the spread band is not measurement noise at all: grouping only
    # by shared-prefix ratio pools 7B with 32B, TP=1 with TP=4, and c=1 with
    # c=128, so the band spans the heterogeneity of the whole matrix and reads as
    # though the measurement were wildly unrepeatable. It is not. The bands in
    # phase 1 were 0.1% at c=1.
    #
    # Chosen scope: the most-measured (backend, model, TP, concurrency)
    # combination that has BOTH caching arms across the most ratios, so the
    # comparison is like for like and the figure states which slice it is.
    best, best_key = None, None
    for key, grp in sub.groupby(
        ["backend", "model_id", "tensor_parallel_size", "concurrency"], dropna=False
    ):
        if grp["enable_prefix_caching"].nunique() < 2:
            continue
        score = (grp["shared_prefix_ratio"].nunique(), len(grp))
        if best is None or score > best:
            best, best_key = score, key
    if best_key is None:
        print("prefix caching: no configuration has both arms at matched settings")
        return False
    backend, model, tp, conc = best_key
    sub = sub[
        (sub["backend"] == backend)
        & (sub["model_id"] == model)
        & (sub["tensor_parallel_size"] == tp)
        & (sub["concurrency"] == conc)
    ]
    scope = f"{backend} · {str(model).split('/')[-1]} · TP={tp:g} · c={conc:g}"
    print(f"prefix caching: scoped to {scope} ({len(sub)} rows)")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    # TTFT vs shared-prefix ratio, caching on vs off.
    ax = axes[0]
    for i, (on, grp) in enumerate(sorted(sub.groupby("enable_prefix_caching"))):
        med = _agg(grp, ["shared_prefix_ratio"], "ttft_s_p50").sort_values("shared_prefix_ratio")
        if med.empty:
            continue
        label = "caching on" if on else "caching off"
        plot_with_spread(ax, med["shared_prefix_ratio"].tolist(), med["median"].tolist(),
                         med["min"].tolist(), med["max"].tolist(), index=i, label=label)
        label_line_ends(ax, med["shared_prefix_ratio"].iloc[-1], med["median"].iloc[-1],
                        "on" if on else "off", i)
    ax.set_xlabel("Shared-prefix ratio")
    ax.set_ylabel("TTFT p50 (s)")
    ax.set_title("Prefix caching effect on TTFT", loc="left")
    ax.legend(loc="upper right")

    # Measured hit rate against the theoretical bound.
    ax = axes[1]
    on_only = sub[sub["enable_prefix_caching"].fillna(False)]
    if not on_only.empty and on_only["prefix_cache_hit_rate"].notna().any():
        med = _agg(on_only, ["shared_prefix_ratio"], "prefix_cache_hit_rate")
        med = med.sort_values("shared_prefix_ratio")
        plot_with_spread(ax, med["shared_prefix_ratio"].tolist(), med["median"].tolist(),
                         med["min"].tolist(), med["max"].tolist(), index=0, label="measured")
        bound = _agg(on_only, ["shared_prefix_ratio"], "theoretical_prefix_hit_rate_bound")
        bound = bound.sort_values("shared_prefix_ratio")
        ax.plot(bound["shared_prefix_ratio"], bound["median"], linestyle=":",
                linewidth=1.4, color=TEXT_MUTED, label="theoretical bound")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "engine reported no hit rate", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Shared-prefix ratio")
    ax.set_ylabel("Prefix cache hit rate")
    ax.set_title("Measured vs achievable: the gap is eviction", loc="left")

    # Throughput effect.
    ax = axes[2]
    for i, (on, grp) in enumerate(sorted(sub.groupby("enable_prefix_caching"))):
        med = _agg(grp, ["shared_prefix_ratio"], "output_tokens_per_s")
        med = med.sort_values("shared_prefix_ratio")
        if med.empty:
            continue
        plot_with_spread(ax, med["shared_prefix_ratio"].tolist(), med["median"].tolist(),
                         med["min"].tolist(), med["max"].tolist(), index=i,
                         label="caching on" if on else "caching off")
    ax.set_xlabel("Shared-prefix ratio")
    ax.set_ylabel("Output throughput (tokens/s)")
    ax.set_title("Throughput effect", loc="left")
    ax.legend(loc="upper left")

    _caption(fig, sub, f"{scope} · shared prefix = 512 tokens across 4 prefix groups · "
                       "all other axes pinned, so the band is repetition spread")
    fig.suptitle("Automatic prefix caching", x=0.0, ha="left", fontsize=11)
    out = FIGURES / "prefix_caching.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return True


# --------------------------------------------------------------------------
def figure_speculative(ok: pd.DataFrame) -> bool:
    sub = ok[ok["sweep_name"].astype(str).str.contains("specdec", na=False)]
    if sub.empty:
        print("speculative decoding: no data")
        return False

    sub = sub.copy()
    sub["spec_on"] = sub["group_label"].astype(str).str.contains("spec-")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    for ax, (metric, ylabel, title) in zip(
        axes[:2],
        [("itl_s_p50", "ITL p50 (s/token)", "Inter-token latency"),
         ("output_tokens_per_s", "Output throughput (tokens/s)", "Throughput")],
        strict=True,
    ):
        for i, (on, grp) in enumerate(sorted(sub.groupby("spec_on"))):
            med = _agg(grp, ["concurrency"], metric).sort_values("concurrency")
            if med.empty:
                continue
            label = "n-gram speculation" if on else "no speculation"
            plot_with_spread(ax, med["concurrency"].tolist(), med["median"].tolist(),
                             med["min"].tolist(), med["max"].tolist(), index=i, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrency (in-flight requests)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")

    ax = axes[2]
    spec = sub[sub["spec_on"] & sub["spec_acceptance_rate"].notna()]
    if not spec.empty:
        med = _agg(spec, ["concurrency"], "spec_acceptance_rate").sort_values("concurrency")
        plot_with_spread(ax, med["concurrency"].tolist(), med["median"].tolist(),
                         med["min"].tolist(), med["max"].tolist(), index=0,
                         label="acceptance rate")
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "engine reported no acceptance rate", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Speculative acceptance rate")
    ax.set_title("Why the win changes with load", loc="left")

    _caption(fig, sub, "vLLM n-gram speculation, 4 draft tokens · shared-prefix ratio 0.6 "
                       "so lookup has repeated text to match")
    fig.suptitle("Speculative decoding: where it stops helping", x=0.0, ha="left", fontsize=11)
    out = FIGURES / "speculative_decoding.png"
    fig.savefig(out)
    print(f"wrote {out}")
    return True


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

    made = [figure_pareto(ok), figure_prefix_caching(ok), figure_speculative(ok)]
    return 0 if any(made) else 1


if __name__ == "__main__":
    raise SystemExit(main())
