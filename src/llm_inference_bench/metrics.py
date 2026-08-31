"""Metric computation from raw request timings.

Two rules from the workspace CLAUDE.md drive every choice here:

* **Report median with spread, never a bare mean of one run.** Point estimates
  without an error indication do not go in a figure, so every aggregate this
  module produces carries its dispersion alongside it.
* **Never invent data.** A metric that cannot be computed is ``None``, never zero
  and never an interpolation. A percentile from too few samples is reported with
  its sample count so the reader can discount it.

The percentile convention is stated explicitly rather than left to a library
default: ``numpy``'s linear interpolation is used throughout, and ``n`` is always
reported next to the percentile so a p99 drawn from 30 requests is visibly a p99
drawn from 30 requests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .client import LoadResult, RequestResult

PERCENTILES = (50.0, 95.0, 99.0)


def _percentiles(values: Sequence[float]) -> dict[str, float | None]:
    """Percentiles plus enough context to judge whether they mean anything."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"n": 0, "p50": None, "p95": None, "p99": None, "mean": None,
                "min": None, "max": None, "std": None}
    out: dict[str, Any] = {"n": int(arr.size)}
    for p in PERCENTILES:
        out[f"p{int(p)}"] = float(np.percentile(arr, p))
    out["mean"] = float(arr.mean())
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["std"] = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return out


@dataclass
class RunMetrics:
    """Everything derived from one load phase."""

    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.payload


def compute_metrics(load: LoadResult, *, warmup_excluded: int = 0) -> dict[str, Any]:
    """Reduce a load phase to the metric set P1 reports.

    Throughput is computed over the **measurement window** (first send to last
    completion) rather than over wall-clock job time, so container startup and
    model loading do not silently deflate it.
    """
    ok = load.successful
    failed = [r for r in load.results if not r.success]

    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e_latency for r in ok if r.e2e_latency is not None]

    # Inter-token latency is pooled across requests. Per-request means would
    # weight a 16-token request the same as a 512-token one, which flatters
    # configurations that produce short outputs.
    itls: list[float] = []
    for r in ok:
        itls.extend(r.inter_token_latencies)

    output_tokens = sum(r.output_tokens_received for r in ok)
    prompt_tokens = sum(r.prompt_tokens_sent for r in ok)

    if ok:
        window_start = min(r.send_time for r in ok)
        window_end = max(r.end_time for r in ok if r.end_time is not None)
        window = max(window_end - window_start, 1e-9)
    else:
        window = None

    # Per-token decode latency derived from the decode span. Preferred over the
    # chunk-derived value whenever the stream batches tokens.
    per_token_itls = [v for v in (r.mean_itl_per_token for r in ok) if v is not None]
    tpc = [v for v in (r.tokens_per_chunk for r in ok) if v is not None]

    metrics: dict[str, Any] = {
        "mode": str(load.mode),
        "concurrency": load.concurrency,
        "request_rate": load.request_rate,
        "requests_total": len(load.results),
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "warmup_excluded": warmup_excluded,
        "measurement_window_s": window,
        "ttft_s": _percentiles(ttfts),
        # Chunk-derived: correct only when one chunk carries one token.
        "itl_s": _percentiles(itls),
        # Decode-span derived: correct regardless of chunking. This is the one to
        # quote for speculative decoding.
        "itl_per_token_s": _percentiles(per_token_itls),
        "tokens_per_chunk": _percentiles(tpc),
        "e2e_latency_s": _percentiles(e2es),
        "prompt_tokens_total": prompt_tokens,
        "output_tokens_total": output_tokens,
    }

    if window:
        metrics["output_tokens_per_s"] = output_tokens / window
        metrics["prompt_tokens_per_s"] = prompt_tokens / window
        metrics["requests_per_s"] = len(ok) / window
    else:
        # No successful requests: throughput is undefined, not zero. Zero would
        # read as "the system served nothing slowly" rather than "nothing ran".
        metrics["output_tokens_per_s"] = None
        metrics["prompt_tokens_per_s"] = None
        metrics["requests_per_s"] = None

    # Failure taxonomy. Failed configurations are data and the writeup needs to
    # distinguish an OOM boundary from a timeout from a 429.
    if failed:
        taxonomy: dict[str, int] = {}
        for r in failed:
            key = _classify_failure(r)
            taxonomy[key] = taxonomy.get(key, 0) + 1
        metrics["failure_taxonomy"] = taxonomy
        metrics["failure_examples"] = [
            {"index": r.index, "status": r.http_status, "error": (r.error or "")[:500]}
            for r in failed[:5]
        ]

    # Client-side honesty check for open loop. If the load generator could not
    # keep to the arrival schedule, the run measured the client, not the server.
    drifts = [r.scheduling_drift for r in load.results if r.scheduling_drift is not None]
    if drifts:
        drift_stats = _percentiles(drifts)
        metrics["client_scheduling_drift_s"] = drift_stats
        p95 = drift_stats.get("p95")
        metrics["client_may_be_bottleneck"] = bool(p95 is not None and p95 > 0.05)

    # Output-length fidelity. ignore_eos should make received == requested; a
    # mismatch means the engine ignored it and the output-length axis is not
    # actually controlled, which would invalidate cross-config comparison.
    mismatched = [
        r for r in ok if r.output_tokens_received != r.output_tokens_requested
    ]
    metrics["output_length_exact"] = not mismatched
    if mismatched:
        metrics["output_length_mismatch_count"] = len(mismatched)
        metrics["output_length_mismatch_examples"] = [
            {
                "index": r.index,
                "requested": r.output_tokens_requested,
                "received": r.output_tokens_received,
            }
            for r in mismatched[:5]
        ]

    # If the stream batched tokens, say so loudly: `itl_s` is then a per-STEP
    # interval, not a per-token one, and quoting it would invert the conclusion
    # about speculative decoding.
    if tpc:
        median_tpc = float(np.median(tpc))
        metrics["stream_batches_tokens"] = bool(median_tpc > 1.15)
        metrics["median_tokens_per_chunk"] = median_tpc
        if median_tpc > 1.15:
            metrics["itl_s_is_per_chunk_not_per_token"] = True

    # Prompt-token fidelity: did the engine see the token count we sent?
    reported = [
        (r.prompt_tokens_sent, r.prompt_tokens_reported)
        for r in ok
        if r.prompt_tokens_reported is not None
    ]
    if reported:
        bad = [(s, g) for s, g in reported if s != g]
        metrics["prompt_length_exact"] = not bad
        if bad:
            metrics["prompt_length_mismatch_count"] = len(bad)
            metrics["prompt_length_mismatch_examples"] = bad[:5]

    return metrics


def _classify_failure(r: RequestResult) -> str:
    """Bucket a failure by its cause, for the failure taxonomy.

    Semantic markers in the error text are checked **before** falling back to the
    HTTP status, and the ordering is load-bearing. An engine that OOMs while
    serving reports it as a 500 with the OOM in the body, so a status-first
    classifier files it as a generic ``http_500`` and the memory boundary
    disappears from the taxonomy. That boundary is the single most interesting
    failure in this project (and the headline result of P2), so it must survive
    classification.

    The one status checked first is 429, which is an explicit overload signal
    rather than an error whose cause has to be read out of a message.
    """
    err = (r.error or "").lower()
    if r.http_status == 429:
        return "http_429_overload"
    if "out of memory" in err or "oom" in err:
        return "oom"
    if "timeout" in err or "timedout" in err:
        return "timeout"
    if "connect" in err:
        return "connection_error"
    if r.http_status and r.http_status >= 400:
        return f"http_{r.http_status}"
    return "other"


def aggregate_repetitions(per_rep: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine N repetitions of one configuration.

    Per the measurement methodology: report the median across repetitions with
    min/max, never a bare mean. The per-repetition values are retained so a
    figure can show the spread rather than assert it.

    Repetitions marked as warmup must already have been filtered out by the
    caller; this function does not silently drop anything.
    """
    if not per_rep:
        return {}

    scalar_keys = [
        "output_tokens_per_s",
        "prompt_tokens_per_s",
        "requests_per_s",
        "measurement_window_s",
    ]
    nested_keys = ["ttft_s", "itl_s", "e2e_latency_s"]

    out: dict[str, Any] = {"n_repetitions": len(per_rep)}

    for key in scalar_keys:
        vals = [m.get(key) for m in per_rep if m.get(key) is not None]
        out[key] = _across_reps(vals)

    for key in nested_keys:
        out[key] = {}
        for stat in ("p50", "p95", "p99", "mean"):
            vals = [
                m.get(key, {}).get(stat)
                for m in per_rep
                if m.get(key, {}).get(stat) is not None
            ]
            out[key][stat] = _across_reps(vals)

    out["requests_ok_total"] = sum(m.get("requests_ok", 0) for m in per_rep)
    out["requests_failed_total"] = sum(m.get("requests_failed", 0) for m in per_rep)
    out["all_output_lengths_exact"] = all(m.get("output_length_exact", False) for m in per_rep)
    return out


def _across_reps(values: Sequence[float]) -> dict[str, Any]:
    """Median plus full range across repetitions, and the raw values themselves.

    The raw list is kept so plots can show individual repetitions rather than
    only an error bar, and so an outlier repetition remains visible rather than
    being absorbed into a summary statistic.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"median": None, "min": None, "max": None, "values": []}
    return {
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        # Interquartile range is meaningless below 4 points; report it only when
        # there are enough repetitions for it to mean something.
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)) if arr.size >= 4 else None,
        "values": [float(v) for v in arr],
    }
