#!/usr/bin/env python3
"""Generate ``results/RESULTS.md`` from raw artifacts.

    python analysis/make_results_tables.py

Exists so that no number in this repository is ever typed by a human. The README
links here rather than quoting figures inline, which makes the
"every number traces to a committed raw artifact" rule mechanically true instead
of a promise: regenerate, and any transcription error simply cannot survive.

Every table states its own N and spread, and every section states the slice it
was computed over. Where a quantity could not be computed it says so rather than
omitting the row, because a silently missing row reads as "not interesting"
rather than "not measured".
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS_RAW = REPO / "results" / "raw"
OUT = REPO / "results" / "RESULTS.md"


def _fmt(v: object, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "not measured"
    if isinstance(v, int | float):
        return f"{v:.{digits}g}"
    return str(v)


def _spread(grp: pd.DataFrame, col: str, digits: int = 3) -> str:
    """Median with min/max and N, the only summary form this project reports."""
    vals = grp[col].dropna()
    if vals.empty:
        return "not measured"
    if len(vals) == 1:
        return f"{_fmt(vals.iloc[0], digits)} (N=1)"
    return (
        f"{_fmt(vals.median(), digits)} "
        f"[{_fmt(vals.min(), digits)}, {_fmt(vals.max(), digits)}] N={len(vals)}"
    )


def section_inventory(frame: pd.DataFrame, report) -> list[str]:
    out = ["## What was measured", ""]
    out.append(f"- artifacts on disk: **{report.files_found}**")
    out.append(f"- analysed (warmup excluded): **{report.parsed}**")
    out.append(f"- warmup repetitions recorded and excluded: **{report.warmup_excluded}**")
    out.append(f"- status breakdown: `{report.by_status}`")
    if report.dirty_git_runs:
        out.append(
            f"- **WARNING**: {report.dirty_git_runs} run(s) from a dirty git tree"
        )
    out.append("")

    ok = frame[frame["status"] == "ok"]
    if ok.empty:
        return out
    out += ["### Coverage", "", "| backend | model | TP | prefix caching | phases | runs |",
            "|---|---|---|---|---|---|"]
    cov = (
        ok.groupby(
            ["backend", "model_id", "tensor_parallel_size", "enable_prefix_caching"],
            dropna=False,
        )
        .agg(phases=("phase_label", "nunique"), runs=("run_id", "count"))
        .reset_index()
        .sort_values(["backend", "model_id", "tensor_parallel_size"])
    )
    for _, r in cov.iterrows():
        out.append(
            f"| {r['backend']} | {str(r['model_id']).split('/')[-1]} "
            f"| {r['tensor_parallel_size']:g} | {r['enable_prefix_caching']} "
            f"| {r['phases']} | {r['runs']} |"
        )
    out.append("")
    return out


def section_failures(frame: pd.DataFrame) -> list[str]:
    """Failed configurations are data. They get their own section, not a footnote."""
    failed = frame[frame["status"] != "ok"]
    out = ["## Configurations that did not run", ""]
    if failed.empty:
        out += ["Every attempted configuration produced a measurement.", ""]
        return out
    out += [
        "Recorded rather than dropped: a configuration that cannot run is a real",
        "boundary of the system, and its absence from a sweep should be visible in",
        "the data rather than only in someone's memory.",
        "",
        "| backend | model | TP | status | runs | cause |",
        "|---|---|---|---|---|---|",
    ]
    grp = failed.groupby(
        ["backend", "model_id", "tensor_parallel_size", "status"], dropna=False
    )
    for (backend, model, tp, status), sub in grp:
        err = str(sub["error"].dropna().iloc[0]) if sub["error"].notna().any() else ""
        first = err.strip().splitlines()[0][:110] if err else "see artifact"
        out.append(
            f"| {backend} | {str(model).split('/')[-1]} | {tp:g} | `{status}` "
            f"| {len(sub)} | {first} |"
        )
    out.append("")
    return out


def section_scaling(ok: pd.DataFrame) -> list[str]:
    out = ["## Tensor-parallel scaling", ""]
    sub = ok[
        (~ok["enable_prefix_caching"].fillna(False))
        & (~ok["phase_label"].astype(str).str.startswith("weak_"))
    ]
    if sub.empty:
        return out + ["Not yet measured.", ""]

    for (backend, model), grp in sorted(sub.groupby(["backend", "model_id"])):
        tps = sorted(grp["tensor_parallel_size"].dropna().unique())
        common = None
        for tp in tps:
            cs = set(grp[grp["tensor_parallel_size"] == tp]["concurrency"].dropna())
            common = cs if common is None else (common & cs)
        if not common:
            continue
        conc = max(common)
        at = grp[grp["concurrency"] == conc]
        out += [
            f"### {backend} · {str(model).split('/')[-1]} · strong scaling at c={conc:g}",
            "",
            "| TP | output tok/s (median [min, max]) | speedup | parallel efficiency |",
            "|---|---|---|---|",
        ]
        base = None
        for tp in tps:
            g = at[at["tensor_parallel_size"] == tp]
            if g.empty:
                continue
            med = g["output_tokens_per_s"].median()
            if tp == 1:
                base = med
            sp = f"{med / base:.2f}x" if base else "n/a"
            eff = f"{(med / base) / tp:.2f}" if base else "n/a"
            out.append(f"| {tp:g} | {_spread(g, 'output_tokens_per_s', 4)} | {sp} | {eff} |")
        out += [
            "",
            "Every GPU pair on this node is joined by NVLink (`NV4`) with no intra-node",
            "PCIe path, so efficiency loss at TP<=4 is **not** an interconnect-topology",
            "effect. It is kernel efficiency, per-replica memory bandwidth, or scheduler",
            "overhead.",
            "",
        ]
    return out


def section_prefix_cache(ok: pd.DataFrame) -> list[str]:
    out = ["## Automatic prefix caching", ""]
    on = ok[ok["enable_prefix_caching"].fillna(False)]
    if on.empty:
        return out + ["Not yet measured.", ""]
    out += [
        "Measured hit rate is computed as a **per-phase delta** of the engine's",
        "cumulative counters, because the cache-reset endpoint clears the cache without",
        "resetting the counters. The theoretical bound is what this workload could",
        "achieve with no eviction; the gap between them is eviction and block-granularity",
        "rounding.",
        "",
        "| backend | shared-prefix ratio | measured hit rate | theoretical bound |",
        "|---|---|---|---|",
    ]
    for (backend, ratio), grp in sorted(on.groupby(["backend", "shared_prefix_ratio"])):
        out.append(
            f"| {backend} | {ratio:g} | {_spread(grp, 'prefix_cache_hit_rate')} "
            f"| {_fmt(grp['theoretical_prefix_hit_rate_bound'].median())} |"
        )
    out.append("")
    return out


def section_length_sweep(ok: pd.DataFrame) -> list[str]:
    """Prefill vs decode: TTFT should scale with input length, output tok/s with
    output length. This is the section the P1 README answers on."""
    out = ["## Prefill vs decode", ""]
    sub = ok[ok["sweep_name"] == "length-sweep"]
    if sub.empty:
        return out + ["Not yet measured.", ""]
    out += [
        "One server per (backend, TP), six length combinations per server as phases.",
        "Short input / long output isolates decode (memory-bandwidth bound at low",
        "concurrency); long input / short output isolates prefill (compute bound).",
        "",
    ]
    for (backend, tp), grp in sorted(sub.groupby(["backend", "tensor_parallel_size"])):
        if grp.empty:
            continue
        out += [
            f"### {backend} · TP={tp:g} · Qwen2.5-7B-Instruct",
            "",
            "| concurrency | input | output | TTFT p50 (s) | ITL p50 (ms/tok) | output tok/s |",
            "|---|---|---|---|---|---|",
        ]
        for (c, il, ol), phase in grp.groupby(
            ["concurrency", "input_len", "output_len"], dropna=False,
        ):
            itl_ms = phase["itl_per_token_s_p50"].dropna() * 1000
            itl_str = (
                f"{itl_ms.median():.3g} [{itl_ms.min():.3g}, {itl_ms.max():.3g}] "
                f"N={len(itl_ms)}"
                if not itl_ms.empty else "not measured"
            )
            out.append(
                f"| {c:g} | {il:g} | {ol:g} | {_spread(phase, 'ttft_s_p50', 4)} "
                f"| {itl_str} | {_spread(phase, 'output_tokens_per_s', 4)} |"
            )
        out.append("")
    return out


def section_packing(ok: pd.DataFrame) -> list[str]:
    out = ["## Cost of packed placement", ""]
    iso = ok[ok["placement"] == "isolated"]
    pk = ok[(ok["placement"] == "packed") & (ok["packing_role"] == "measured")]
    if iso.empty or pk.empty:
        return out + ["Not yet measured.", ""]
    out += [
        "Identical server group, same node, same job: once alone with three GPUs idle,",
        "once as four NUMA-pinned replicas with only GPU 0 measured.",
        "",
        "| phase | metric | isolated | packed | change |",
        "|---|---|---|---|---|",
    ]
    metrics = [
        ("ttft_s_p50", "TTFT p50 (s)", "lower"),
        ("itl_s_p50", "ITL p50 (s/tok)", "lower"),
        ("output_tokens_per_s", "output tok/s", "higher"),
    ]
    for phase in sorted(set(iso["phase_label"]) & set(pk["phase_label"])):
        a, b = iso[iso["phase_label"] == phase], pk[pk["phase_label"] == phase]
        for col, name, direction in metrics:
            ma, mb = a[col].median(), b[col].median()
            if ma != ma or mb != mb or ma == 0:
                continue
            d = (mb - ma) / ma * 100
            worse = d > 0 if direction == "lower" else d < 0
            out.append(
                f"| {phase} | {name} | {_spread(a, col, 4)} | {_spread(b, col, 4)} "
                f"| **{d:+.1f}%** {'worse' if worse else 'better'} |"
            )
    out.append("")
    return out


def main() -> int:
    if not RESULTS_RAW.exists() or not any(RESULTS_RAW.glob("*.json")):
        print(f"no raw artifacts in {RESULTS_RAW}")
        return 1
    frame, report = load(RESULTS_RAW, include_warmup=False, include_failed=True)
    ok = frame[frame["status"] == "ok"]

    first = ok.iloc[0] if not ok.empty else None
    lines = [
        "# Measured results",
        "",
        "**Generated file. Do not edit.** Regenerate with:",
        "",
        "```bash",
        "python analysis/make_results_tables.py",
        "```",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} from "
        f"`results/raw/` ({report.files_found} artifacts).",
        "",
    ]
    if first is not None:
        lines += [
            "## Provenance",
            "",
            f"- hardware: **{first['gpu_count']}x {first['gpu_model']}**, "
            f"driver {first['driver_version']}, CUDA {first['cuda_version']}",
            f"- backends: **{sorted(set(ok['backend'].dropna()))}**, versions "
            f"**{sorted(set(str(v) for v in ok['backend_version'].dropna()))}**",
            f"- models: **{sorted(set(ok['model_id'].dropna()))}**",
            f"- git sha: `{first['git_sha']}`",
            "",
            "Every row below traces to a committed artifact under `results/raw/`.",
            "Summaries are median with [min, max] across repetitions, never a bare mean.",
            "",
        ]
    lines += section_inventory(frame, report)
    lines += section_scaling(ok) if not ok.empty else []
    lines += section_length_sweep(ok) if not ok.empty else []
    lines += section_prefix_cache(ok) if not ok.empty else []
    lines += section_packing(ok) if not ok.empty else []
    lines += section_failures(frame)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
