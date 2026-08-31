"""Load raw artifacts into a tidy frame.

Reads ``results/raw/`` and nothing else. This module never writes there, never
repairs a malformed artifact, and never fills a missing value: an artifact that
cannot be parsed is reported and skipped, not patched.

Warmup handling is the load-bearing part. Per the measurement methodology, the
first repetition of any configuration is excluded from analysis because it pays
CUDA graph capture, autotuning and cache population. It is *recorded* though, so
the exclusion happens here, visibly and countably, rather than at write time
where it would be invisible.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class LoadReport:
    """What the loader saw, so a thin dataset cannot be mistaken for a full one."""

    files_found: int
    parsed: int
    unparseable: list[tuple[str, str]]
    by_status: dict[str, int]
    warmup_excluded: int
    dirty_git_runs: int

    def summary(self) -> str:
        lines = [
            f"artifacts found:     {self.files_found}",
            f"parsed:              {self.parsed}",
            f"warmup excluded:     {self.warmup_excluded}",
            f"status breakdown:    {self.by_status}",
        ]
        if self.dirty_git_runs:
            lines.append(
                f"WARNING: {self.dirty_git_runs} run(s) came from a dirty git tree "
                "and are not reproducible from a clean checkout"
            )
        if self.unparseable:
            lines.append(f"UNPARSEABLE ({len(self.unparseable)}):")
            lines.extend(f"  {name}: {err}" for name, err in self.unparseable[:10])
        return "\n".join(lines)


def iter_artifacts(results_raw: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(results_raw.glob("*.json")):
        with path.open() as fh:
            yield path, json.load(fh)


def load(
    results_raw: Path,
    *,
    include_warmup: bool = False,
    include_failed: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """Flatten raw artifacts into one row per measurement.

    ``include_failed`` defaults to True. Failed configurations are data: an OOM
    boundary is a real result and dropping it by default would quietly turn a
    measured limit into an absence.
    """
    rows: list[dict[str, Any]] = []
    unparseable: list[tuple[str, str]] = []
    by_status: dict[str, int] = {}
    warmup_excluded = 0
    dirty = 0
    found = 0

    for path, payload in _safe_iter(results_raw, unparseable):
        found += 1
        prov = payload.get("provenance", {})
        meas = payload.get("measurements", {})

        status = prov.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        if prov.get("warmup"):
            warmup_excluded += 1
            if not include_warmup:
                continue
        if status != "ok" and not include_failed:
            continue
        if str(prov.get("git_sha", "")).endswith("-dirty"):
            dirty += 1

        cfg = prov.get("full_config_dict", {})
        engine = cfg.get("engine", {})
        load_cfg = cfg.get("load", {})
        workload = cfg.get("workload", {})
        telemetry = meas.get("engine_telemetry", {}) or {}

        row: dict[str, Any] = {
            "artifact": path.name,
            "run_id": prov.get("run_id"),
            "utc_timestamp": prov.get("utc_timestamp"),
            "status": status,
            "warmup": prov.get("warmup"),
            "repetition": prov.get("repetition_index"),
            "git_sha": prov.get("git_sha"),
            "hostname": prov.get("hostname"),
            "slurm_job_id": prov.get("slurm_job_id"),
            # Hardware, carried onto every row so a figure caption can name it
            # without a second lookup.
            "gpu_model": prov.get("gpu_model"),
            "gpu_count": prov.get("gpu_count"),
            "gpu_memory_gb": prov.get("gpu_memory_gb"),
            "driver_version": prov.get("driver_version"),
            "cuda_version": prov.get("cuda_version"),
            # Software identity
            "backend": prov.get("backend_name"),
            "backend_version": prov.get("backend_version"),
            "image_digest": prov.get("backend_image_digest"),
            "model_id": prov.get("model_id"),
            "model_revision": prov.get("model_revision"),
            # Configuration
            "sweep_name": cfg.get("sweep_name"),
            "group_label": cfg.get("group_label"),
            "phase_label": cfg.get("phase_label"),
            "tensor_parallel_size": engine.get("tensor_parallel_size"),
            "enable_prefix_caching": engine.get("enable_prefix_caching"),
            "block_size": engine.get("block_size"),
            "kv_cache_dtype": engine.get("kv_cache_dtype"),
            "max_num_seqs": engine.get("max_num_seqs"),
            "gpu_memory_utilization": engine.get("gpu_memory_utilization"),
            "load_mode": load_cfg.get("mode"),
            "concurrency": load_cfg.get("concurrency"),
            "request_rate": load_cfg.get("request_rate"),
            "input_len": _length_mean(workload.get("input_len")),
            "output_len": _length_mean(workload.get("output_len")),
            "shared_prefix_ratio": workload.get("shared_prefix_ratio"),
            "num_requests": workload.get("num_requests"),
            "workload_fingerprint": workload.get("fingerprint"),
            # Measurements
            "requests_ok": meas.get("requests_ok"),
            "requests_failed": meas.get("requests_failed"),
            "output_tokens_per_s": meas.get("output_tokens_per_s"),
            "requests_per_s": meas.get("requests_per_s"),
            "measurement_window_s": meas.get("measurement_window_s"),
            "engine_startup_s": meas.get("engine_startup_s"),
            "output_length_exact": meas.get("output_length_exact"),
            "prompt_length_exact": meas.get("prompt_length_exact"),
            "cache_reset_ok": meas.get("cache_reset_ok"),
            "client_may_be_bottleneck": meas.get("client_may_be_bottleneck"),
            "theoretical_prefix_hit_rate_bound": meas.get(
                "theoretical_prefix_hit_rate_bound"
            ),
            # Operator tags: how the run was executed, as opposed to how it was
            # configured. `placement` distinguishes packed from isolated runs of
            # an otherwise identical group, which is the only thing that makes
            # the packing-interference delta computable.
            "placement": (prov.get("notes") or {}).get("placement"),
            "packing_role": (prov.get("notes") or {}).get("packing_role"),
            "packing_gpu": (prov.get("notes") or {}).get("packing_gpu"),
            "packing_job": (prov.get("notes") or {}).get("packing_job"),
            # Engine-reported telemetry, already normalised by the adapters
            "prefix_cache_hit_rate": telemetry.get("prefix_cache_hit_rate"),
            # Raw counters retained so the rate can be recomputed as a per-phase
            # delta; see add_prefix_cache_deltas.
            "raw_hits": _counter(telemetry.get("raw_prometheus") or {}, _HIT_KEYS),
            "raw_queries": _counter(telemetry.get("raw_prometheus") or {}, _QUERY_KEYS),
            "spec_acceptance_rate": telemetry.get("spec_acceptance_rate"),
            "kv_cache_usage_frac": telemetry.get("kv_cache_usage_frac"),
            "preemptions_total": telemetry.get("preemptions_total"),
            "error": prov.get("error"),
        }
        # Recover per-token ITL for artifacts written before the metric existed.
        # Every request's ttft, e2e and token count are stored, so the decode
        # span divides out exactly; nothing needs re-measuring.
        per_tok = []
        for r in meas.get("per_request", []) or []:
            if not r.get("success"):
                continue
            n, e2e, ttft = r.get("output_tokens"), r.get("e2e_s"), r.get("ttft_s")
            if n and e2e is not None and ttft is not None and n > 1:
                per_tok.append((e2e - ttft) / (n - 1))
        if per_tok:
            per_tok.sort()
            row["itl_per_token_s_p50_recovered"] = per_tok[len(per_tok) // 2]
            row["itl_per_token_s_p95_recovered"] = per_tok[int(len(per_tok) * 0.95) - 1]

        for metric in ("ttft_s", "itl_s", "e2e_latency_s", "itl_per_token_s"):
            block = meas.get(metric) or {}
            for stat in ("p50", "p95", "p99", "mean", "min", "max", "n"):
                row[f"{metric}_{stat}"] = block.get(stat)

        rows.append(row)

    frame = pd.DataFrame(rows)
    # Prefer the live metric, fall back to the value recovered from per-request
    # timings, so old and new artifacts are directly comparable.
    if "itl_per_token_s_p50" in frame.columns:
        frame["itl_per_token_s_p50"] = frame["itl_per_token_s_p50"].combine_first(
            frame.get("itl_per_token_s_p50_recovered")
        )
    elif "itl_per_token_s_p50_recovered" in frame.columns:
        frame["itl_per_token_s_p50"] = frame["itl_per_token_s_p50_recovered"]
    frame = add_prefix_cache_deltas(frame)
    report = LoadReport(
        files_found=found,
        parsed=len(rows),
        unparseable=unparseable,
        by_status=by_status,
        warmup_excluded=warmup_excluded,
        dirty_git_runs=dirty,
    )
    return frame, report


def _safe_iter(
    results_raw: Path, unparseable: list[tuple[str, str]]
) -> Iterator[tuple[Path, dict[str, Any]]]:
    for path in sorted(results_raw.glob("*.json")):
        try:
            with path.open() as fh:
                yield path, json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            unparseable.append((path.name, str(exc)))


# Prometheus counter families whose value we recover from stored raw telemetry.
# Both vLLM spellings are listed: 0.27.1 dropped the `gpu_` infix.
_HIT_KEYS = ("vllm:gpu_prefix_cache_hits_total", "vllm:prefix_cache_hits_total")
_QUERY_KEYS = ("vllm:gpu_prefix_cache_queries_total", "vllm:prefix_cache_queries_total")


def _counter(raw: dict[str, Any], names: tuple[str, ...]) -> float | None:
    """Sum a Prometheus counter family, trying each accepted spelling."""
    for name in names:
        vals = [v for k, v in raw.items() if k.startswith(name)]
        if vals:
            return float(sum(vals))
    return None


def add_prefix_cache_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute prefix cache hit rate per phase, from stored raw telemetry.

    Two problems are solved here, and neither needed a re-measurement because the
    full Prometheus exposition is stored in every artifact.

    **Metric renaming.** vLLM 0.27.1 dropped the `gpu_` infix, so artifacts
    collected while the adapter looked only for the old spelling carry no
    normalised hit rate. The raw exposition still has the numbers.

    **Cumulative counters.** ``/reset_prefix_cache`` clears the cache but does
    NOT reset the Prometheus counters, which run for the lifetime of the server.
    Since one server serves many phases, the raw ratio at the end of a phase is
    contaminated by every phase before it: a 0.9-shared-prefix phase following a
    0.0 phase reads low, and the on/off comparison is skewed by ordering.

    Taking the difference between consecutive scrapes within a server group,
    ordered by time, recovers the per-phase rate. The first artifact of each
    group has no predecessor and gets NaN rather than a guess.
    """
    if frame.empty or "raw_hits" not in frame.columns:
        return frame
    frame = frame.sort_values(["group_label", "utc_timestamp", "repetition"]).copy()
    frame["prefix_cache_hit_rate_delta"] = float("nan")
    frame["prefix_cache_queries_delta"] = float("nan")

    for _, idx in frame.groupby("group_label", dropna=False).groups.items():
        sub = frame.loc[idx]
        dh = sub["raw_hits"].diff()
        dq = sub["raw_queries"].diff()
        # A negative delta means the server restarted and the counters reset;
        # treat it as unusable rather than reporting a nonsensical rate.
        valid = (dq > 0) & (dh >= 0)
        frame.loc[idx, "prefix_cache_queries_delta"] = dq.where(valid)
        frame.loc[idx, "prefix_cache_hit_rate_delta"] = (dh / dq).where(valid)

    # Prefer the delta; fall back to whatever the adapter normalised live.
    frame["prefix_cache_hit_rate"] = frame["prefix_cache_hit_rate_delta"].combine_first(
        frame["prefix_cache_hit_rate"]
    )
    return frame


def _length_mean(spec: Any) -> Any:
    """Pull a scalar length out of a serialised LengthSpec."""
    if isinstance(spec, dict):
        return spec.get("mean")
    return spec


def per_request_frame(results_raw: Path) -> pd.DataFrame:
    """One row per individual request, across all artifacts.

    The runner stores every request's timing, so distributional questions asked
    later (does the TTFT tail come from prefix-group requests? did latency drift
    during the run?) are answerable without spending another allocation.
    """
    rows: list[dict[str, Any]] = []
    for path, payload in iter_artifacts(results_raw):
        prov = payload.get("provenance", {})
        if prov.get("warmup"):
            continue
        cfg = prov.get("full_config_dict", {})
        for req in payload.get("measurements", {}).get("per_request", []) or []:
            rows.append(
                {
                    "artifact": path.name,
                    "run_id": prov.get("run_id"),
                    "backend": prov.get("backend_name"),
                    "group_label": cfg.get("group_label"),
                    "phase_label": cfg.get("phase_label"),
                    "repetition": prov.get("repetition_index"),
                    **req,
                }
            )
    return pd.DataFrame(rows)


def aggregate_over_repetitions(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Median across repetitions, with min/max, per the reporting rule.

    Never a bare mean, and the repetition count travels with every row so a
    figure can state N rather than imply it.
    """
    metrics = [
        "output_tokens_per_s",
        "requests_per_s",
        "ttft_s_p50",
        "ttft_s_p95",
        "ttft_s_p99",
        "itl_s_p50",
        "itl_s_p95",
        "e2e_latency_s_p50",
        "e2e_latency_s_p95",
        "e2e_latency_s_p99",
        "prefix_cache_hit_rate",
        "spec_acceptance_rate",
    ]
    present = [m for m in metrics if m in frame.columns]
    ok = frame[frame["status"] == "ok"]
    agg = ok.groupby(group_cols, dropna=False)[present].agg(["median", "min", "max", "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()
