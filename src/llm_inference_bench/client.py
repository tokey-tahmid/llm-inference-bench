"""Backend-agnostic load client.

Nothing engine-specific may live here. Both target engines serve an
OpenAI-compatible HTTP API, so every externally observable metric is measured
identically against every backend; the moment a backend special case leaks in,
the backend-vs-backend comparison stops being apples to apples.

Three choices that decide whether the numbers mean anything:

**``/v1/completions``, not ``/v1/chat/completions``.** Chat templates differ
between models and between engine versions, so a chat request with the same text
becomes a different token sequence on each backend. The completions endpoint with
integer-token prompts sends byte-identical work to both engines.

**``ignore_eos``.** Without it, output length is whatever the model decides, so a
config that happens to emit short answers looks faster. With it, the output-length
axis is exact and controlled. It is requested explicitly and its acceptance is
verified, not assumed.

**Open loop and closed loop are both provided, and they answer different
questions.** Closed loop (fixed concurrency) measures the system at a held
occupancy and cannot show queueing collapse, because backpressure propagates to
the client and throttles the offered load. Open loop (Poisson arrivals at a fixed
rate) is what exposes the latency knee past saturation. A throughput/latency
Pareto frontier drawn from closed-loop data alone would miss exactly the regime
that matters. The client records its own scheduling drift under open loop, so a
run where the *client* was the bottleneck is identifiable in the data rather than
misread as server saturation.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
import numpy as np
import structlog

from .workload.generator import GeneratedRequest

log = structlog.get_logger(__name__)


class LoadMode(StrEnum):
    CLOSED_LOOP = "closed_loop"
    OPEN_LOOP = "open_loop"


@dataclass
class RequestResult:
    """One request's timing, in seconds relative to the run's start."""

    index: int
    success: bool
    prefix_group: int | None

    send_time: float
    first_token_time: float | None
    end_time: float | None

    prompt_tokens_sent: int
    prompt_tokens_reported: int | None
    output_tokens_requested: int
    output_tokens_received: int

    # Per-chunk arrival times, so inter-token latency can be recomputed later
    # without rerunning. A chunk is not always exactly one token, and the token
    # count per chunk is kept so ITL is never silently miscomputed as per-chunk.
    chunk_times: list[float] = field(default_factory=list)
    chunk_token_counts: list[int] = field(default_factory=list)

    # Open loop only: intended minus actual send time. Consistently large values
    # mean the load generator, not the server, was the limiting factor.
    scheduling_drift: float | None = None

    error: str | None = None
    http_status: int | None = None

    @property
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.send_time

    @property
    def e2e_latency(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.send_time

    @property
    def inter_token_latencies(self) -> list[float]:
        """Gaps between successive decode chunks, normalised per token.

        Excludes the prefill gap (send to first token), which is TTFT and a
        different quantity. Chunks carrying multiple tokens have their gap divided
        by the token count so the result is per-token throughout.
        """
        if len(self.chunk_times) < 2:
            return []
        out: list[float] = []
        for i in range(1, len(self.chunk_times)):
            gap = self.chunk_times[i] - self.chunk_times[i - 1]
            ntok = max(1, self.chunk_token_counts[i])
            out.extend([gap / ntok] * ntok)
        return out


@dataclass
class LoadResult:
    """Everything one load phase produced."""

    results: list[RequestResult]
    mode: LoadMode
    concurrency: int | None
    request_rate: float | None
    wall_start: float
    wall_end: float

    @property
    def duration(self) -> float:
        return self.wall_end - self.wall_start

    @property
    def successful(self) -> list[RequestResult]:
        return [r for r in self.results if r.success]


class LoadClient:
    """Drives a running engine over its OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        *,
        request_timeout: float = 600.0,
        max_connections: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.request_timeout = request_timeout
        self._limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
        )

    def _payload(self, req: GeneratedRequest) -> dict[str, Any]:
        prompt: Any = req.prompt_token_ids if req.prompt_text is None else req.prompt_text
        return {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": req.output_tokens,
            "min_tokens": req.output_tokens,
            # Greedy: sampling variance would show up as run-to-run noise in
            # output length and cache behaviour, confounding the repetitions.
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
            # Force the exact output length. Both engines accept this at the top
            # level of the completions request.
            "ignore_eos": True,
        }

    async def _one_request(
        self,
        client: httpx.AsyncClient,
        req: GeneratedRequest,
        t0: float,
        scheduling_drift: float | None = None,
    ) -> RequestResult:
        result = RequestResult(
            index=req.index,
            success=False,
            prefix_group=req.prefix_group,
            send_time=time.perf_counter() - t0,
            first_token_time=None,
            end_time=None,
            prompt_tokens_sent=req.input_tokens,
            prompt_tokens_reported=None,
            output_tokens_requested=req.output_tokens,
            output_tokens_received=0,
            scheduling_drift=scheduling_drift,
        )
        try:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/completions",
                json=self._payload(req),
                timeout=self.request_timeout,
            ) as resp:
                result.http_status = resp.status_code
                if resp.status_code >= 300:
                    body = (await resp.aread()).decode(errors="replace")[:2000]
                    result.error = f"HTTP {resp.status_code}: {body}"
                    result.end_time = time.perf_counter() - t0
                    return result

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    now = time.perf_counter() - t0
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    # The final usage-only chunk carries no choices; it must not
                    # be counted as a decode step or it inflates the last ITL.
                    usage = chunk.get("usage")
                    choices = chunk.get("choices") or []
                    text = "".join(c.get("text", "") for c in choices)

                    if choices and text:
                        if result.first_token_time is None:
                            result.first_token_time = now
                        result.chunk_times.append(now)
                        result.chunk_token_counts.append(1)
                        result.output_tokens_received += 1

                    if usage:
                        result.prompt_tokens_reported = usage.get("prompt_tokens")
                        if usage.get("completion_tokens"):
                            # Trust the server's count over our chunk count: a
                            # chunk can carry more than one token.
                            result.output_tokens_received = usage["completion_tokens"]

                result.end_time = time.perf_counter() - t0
                result.success = result.first_token_time is not None
                if not result.success:
                    result.error = result.error or "stream produced no tokens"
        except (TimeoutError, httpx.HTTPError) as exc:
            result.end_time = time.perf_counter() - t0
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    async def run_closed_loop(
        self, requests: Sequence[GeneratedRequest], concurrency: int
    ) -> LoadResult:
        """Hold ``concurrency`` requests in flight until the request list is done."""
        sem = asyncio.Semaphore(concurrency)
        t0 = time.perf_counter()
        wall_start = time.time()

        async with httpx.AsyncClient(limits=self._limits, http2=False) as client:

            async def guarded(req: GeneratedRequest) -> RequestResult:
                async with sem:
                    return await self._one_request(client, req, t0)

            results = await asyncio.gather(*(guarded(r) for r in requests))

        return LoadResult(
            results=list(results),
            mode=LoadMode.CLOSED_LOOP,
            concurrency=concurrency,
            request_rate=None,
            wall_start=wall_start,
            wall_end=time.time(),
        )

    async def run_open_loop(
        self,
        requests: Sequence[GeneratedRequest],
        request_rate: float,
        *,
        burstiness: float = 1.0,
    ) -> LoadResult:
        """Poisson arrivals at ``request_rate`` req/s, independent of completions.

        ``burstiness`` is the shape parameter of a gamma inter-arrival
        distribution: 1.0 is exactly Poisson, below 1.0 is burstier, above 1.0 is
        more regular. Held at 1.0 unless a config says otherwise, since a memoryless
        arrival process is the standard assumption a serving system is designed for.

        Arrivals are *not* gated on completions. If the server saturates, the
        backlog grows and latency diverges, which is the phenomenon being measured.
        """
        rng = np.random.default_rng(0xC0FFEE)
        if request_rate <= 0:
            raise ValueError("request_rate must be > 0 for open loop")
        gaps = rng.gamma(
            shape=burstiness, scale=1.0 / (request_rate * burstiness), size=len(requests)
        )
        arrival_offsets = np.cumsum(gaps)

        t0 = time.perf_counter()
        wall_start = time.time()
        tasks: list[asyncio.Task[RequestResult]] = []

        async with httpx.AsyncClient(limits=self._limits, http2=False) as client:
            for req, offset in zip(requests, arrival_offsets, strict=True):
                now = time.perf_counter() - t0
                delay = float(offset) - now
                if delay > 0:
                    await asyncio.sleep(delay)
                    drift = 0.0
                else:
                    # Negative delay means the client fell behind the intended
                    # arrival schedule. Recorded, not hidden: a run with large
                    # drift measured the client, not the server.
                    drift = -delay
                tasks.append(
                    asyncio.create_task(self._one_request(client, req, t0, scheduling_drift=drift))
                )
            results = await asyncio.gather(*tasks)

        drifts = [r.scheduling_drift or 0.0 for r in results]
        if drifts and float(np.percentile(drifts, 95)) > 0.05:
            log.warning(
                "open_loop_client_lagging",
                p95_drift_s=float(np.percentile(drifts, 95)),
                request_rate=request_rate,
                hint="the load generator may be the bottleneck, not the server",
            )

        return LoadResult(
            results=list(results),
            mode=LoadMode.OPEN_LOOP,
            concurrency=None,
            request_rate=request_rate,
            wall_start=wall_start,
            wall_end=time.time(),
        )
