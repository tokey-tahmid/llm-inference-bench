"""Metric computation tests.

The recurring theme: a metric that could not be computed must be ``None``, never
zero. Zero reads as a measurement; None reads as an absence. Conflating them is
how fabricated numbers get into a writeup without anyone deciding to put them there.
"""

from __future__ import annotations

import pytest

from llm_inference_bench.client import LoadMode, LoadResult, RequestResult
from llm_inference_bench.metrics import aggregate_repetitions, compute_metrics


def _request(
    index: int = 0,
    *,
    success: bool = True,
    send: float = 0.0,
    first: float | None = 0.1,
    end: float | None = 1.1,
    out_requested: int = 10,
    out_received: int = 10,
    chunks: int = 10,
    status: int | None = 200,
    error: str | None = None,
) -> RequestResult:
    times = [first + 0.1 * i for i in range(chunks)] if first is not None else []
    return RequestResult(
        index=index,
        success=success,
        prefix_group=None,
        send_time=send,
        first_token_time=first,
        end_time=end,
        prompt_tokens_sent=100,
        prompt_tokens_reported=100,
        output_tokens_requested=out_requested,
        output_tokens_received=out_received,
        chunk_times=times,
        chunk_token_counts=[1] * len(times),
        http_status=status,
        error=error,
    )


def _load(results: list[RequestResult], **kw) -> LoadResult:
    return LoadResult(
        results=results,
        mode=kw.get("mode", LoadMode.CLOSED_LOOP),
        concurrency=kw.get("concurrency", 1),
        request_rate=kw.get("request_rate"),
        wall_start=0.0,
        wall_end=10.0,
    )


def test_basic_metrics() -> None:
    m = compute_metrics(_load([_request(i) for i in range(5)]))
    assert m["requests_ok"] == 5
    assert m["requests_failed"] == 0
    assert m["ttft_s"]["p50"] == pytest.approx(0.1)
    assert m["output_tokens_total"] == 50


def test_throughput_is_none_when_nothing_succeeded() -> None:
    """Not zero. Zero would read as 'served nothing slowly' rather than 'nothing ran'."""
    failed = [_request(i, success=False, first=None, end=0.5, status=500) for i in range(3)]
    m = compute_metrics(_load(failed))
    assert m["requests_ok"] == 0
    assert m["output_tokens_per_s"] is None
    assert m["measurement_window_s"] is None
    assert m["ttft_s"]["n"] == 0
    assert m["ttft_s"]["p50"] is None


def test_failure_taxonomy_distinguishes_causes() -> None:
    """An OOM boundary and an overload 429 are different findings."""
    results = [
        _request(0, success=False, first=None, status=429, error="Too many requests"),
        _request(1, success=False, first=None, status=500, error="CUDA out of memory"),
        _request(2, success=False, first=None, status=None, error="ReadTimeout: timed out"),
        _request(3),
    ]
    m = compute_metrics(_load(results))
    tax = m["failure_taxonomy"]
    assert tax["http_429_overload"] == 1
    assert tax["oom"] == 1
    assert tax["timeout"] == 1


def test_output_length_mismatch_is_flagged() -> None:
    """ignore_eos should make received == requested; if not, the axis is uncontrolled."""
    ok = _request(0, out_requested=10, out_received=10)
    bad = _request(1, out_requested=10, out_received=4)
    m = compute_metrics(_load([ok, bad]))
    assert m["output_length_exact"] is False
    assert m["output_length_mismatch_count"] == 1


def test_itl_excludes_the_prefill_gap() -> None:
    """TTFT and ITL are different quantities and must not be pooled."""
    r = _request(0, first=5.0, chunks=3)  # chunks at 5.0, 5.1, 5.2
    itls = r.inter_token_latencies
    assert len(itls) == 2
    assert all(v == pytest.approx(0.1) for v in itls)
    assert 5.0 not in itls


def test_itl_normalises_multi_token_chunks() -> None:
    r = _request(0, first=1.0, chunks=2)
    r.chunk_token_counts = [1, 4]  # second chunk carried 4 tokens over 0.1s
    itls = r.inter_token_latencies
    assert len(itls) == 4
    assert all(v == pytest.approx(0.025) for v in itls)


def test_client_bottleneck_is_surfaced() -> None:
    """A run where the load generator lagged measured the client, not the server."""
    results = [_request(i) for i in range(4)]
    for r in results:
        r.scheduling_drift = 0.5
    m = compute_metrics(_load(results, mode=LoadMode.OPEN_LOOP, request_rate=100.0))
    assert m["client_may_be_bottleneck"] is True


def test_aggregate_reports_median_and_spread_not_bare_mean() -> None:
    reps = [
        {"output_tokens_per_s": 100.0, "ttft_s": {"p50": 0.1}, "requests_ok": 10,
         "requests_failed": 0, "output_length_exact": True},
        {"output_tokens_per_s": 110.0, "ttft_s": {"p50": 0.12}, "requests_ok": 10,
         "requests_failed": 0, "output_length_exact": True},
        {"output_tokens_per_s": 105.0, "ttft_s": {"p50": 0.11}, "requests_ok": 10,
         "requests_failed": 0, "output_length_exact": True},
    ]
    agg = aggregate_repetitions(reps)
    assert agg["n_repetitions"] == 3
    tput = agg["output_tokens_per_s"]
    assert tput["median"] == pytest.approx(105.0)
    assert tput["min"] == pytest.approx(100.0)
    assert tput["max"] == pytest.approx(110.0)
    # Raw values retained so a figure can show repetitions, not just an error bar.
    assert tput["values"] == [100.0, 110.0, 105.0]


def test_iqr_withheld_when_too_few_repetitions() -> None:
    """An IQR from 3 points is not meaningful and should not be offered."""
    reps = [{"output_tokens_per_s": float(v)} for v in (100, 110, 105)]
    agg = aggregate_repetitions(reps)
    assert agg["output_tokens_per_s"]["iqr"] is None


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate_repetitions([]) == {}
