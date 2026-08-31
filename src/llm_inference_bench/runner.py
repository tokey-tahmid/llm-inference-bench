"""Sweep execution: server groups in, raw artifacts out.

The runner is the only component that both launches engines and writes results,
so it is where the non-negotiable rules are enforced in practice:

* Every phase produces exactly one artifact, including phases that failed. An
  OOM at a given batch/context is a real boundary and is recorded as such.
* A configuration the backend cannot serve is recorded as ``unsupported`` with
  the adapter's stated reasons, so the gap is visible in the data rather than
  appearing as a hole someone has to explain from memory later.
* Nothing is estimated. If a measurement did not happen, no number is written.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import structlog

from .backends import BackendLaunchError, get_adapter
from .backends.base import BackendAdapter, ServerHandle
from .client import LoadClient, LoadMode
from .metrics import compute_metrics
from .provenance import RunStatus, build_provenance, git_sha, write_artifact
from .sweep import Phase, ServerGroup
from .workload.generator import (
    WorkloadGenerator,
    theoretical_prefix_hit_rate,
    validate_prompt_token_fidelity,
)

log = structlog.get_logger(__name__)


@dataclass
class RunnerConfig:
    repo_root: Path
    results_raw: Path
    log_dir: Path
    repetitions: int = 3
    warmup_repetitions: int = 1
    gpu_ids: list[int] | None = None
    startup_timeout: float = 900.0
    sweep_name: str = "unnamed"
    # Free-form tags recorded in every artifact's provenance. Used to
    # distinguish otherwise-identical runs that differ in how they were
    # executed rather than in what was configured, e.g. packed vs isolated
    # placement on a node. Without this the two are indistinguishable in
    # results/raw and the interference delta cannot be computed at all.
    notes: dict[str, Any] = field(default_factory=dict)


def _load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


class SweepRunner:
    """Executes server groups and writes one raw artifact per measurement."""

    def __init__(
        self,
        cfg: RunnerConfig,
        image_paths: dict[str, Path],
        image_digests: dict[str, str | None],
        backend_versions: dict[str, str | None] | None = None,
    ) -> None:
        # Sample the git state EAGERLY, here, before any measurement runs.
        #
        # git_sha is cached for the process, but the cache populates on first
        # call, which was the first artifact write, minutes into the sweep. If
        # anyone touched the working tree in that window, every artifact from a
        # multi-hour run got stamped `-dirty` even though the code under test was
        # the clean tree loaded at launch. That happened here: analysis fixes
        # committed while sweeps were in flight marked 1027 otherwise-valid
        # artifacts as unreproducible.
        #
        # Sampling at construction time makes the recorded sha mean what it
        # claims: the state of the tree when this run started. It does not
        # license editing mid-sweep, it just stops an unrelated edit from
        # corrupting the record.
        sha = git_sha(cfg.repo_root)
        log.info("provenance_git_sha_sampled", git_sha=sha)
        self.cfg = cfg
        self.image_paths = image_paths
        self.image_digests = image_digests
        self.backend_versions = backend_versions or {}

    # -- artifact helpers ----------------------------------------------------

    def _write(
        self,
        group: ServerGroup,
        phase: Phase,
        rep: int,
        warmup: bool,
        status: RunStatus,
        measurements: dict[str, Any],
        backend_version: str | None = None,
        error: str | None = None,
    ) -> Path:
        config = {
            "sweep_name": self.cfg.sweep_name,
            "group_id": group.group_id(),
            "group_label": group.label,
            "phase_id": phase.phase_id(),
            "phase_label": phase.label,
            "engine": group.engine.as_dict(),
            "load": {
                "mode": str(phase.load.mode),
                "concurrency": phase.load.concurrency,
                "request_rate": phase.load.request_rate,
                "burstiness": phase.load.burstiness,
            },
            "workload": {
                **{
                    k: v
                    for k, v in phase.workload.__dict__.items()
                    if not k.startswith("_")
                },
                "fingerprint": phase.workload.fingerprint(),
            },
        }
        prov = build_provenance(
            repo_root=self.cfg.repo_root,
            backend_name=group.backend,
            backend_version=backend_version,
            backend_image_digest=self.image_digests.get(group.backend),
            model_id=group.model_repo,
            model_revision=group.model_revision,
            config=config,
            repetition_index=rep,
            warmup=warmup,
            status=status,
            error=error,
            notes=self.cfg.notes,
        )
        return write_artifact(self.cfg.results_raw, prov, measurements)

    # -- execution -----------------------------------------------------------

    def run_group(self, group: ServerGroup) -> list[Path]:
        """Launch one engine and measure every phase against it."""
        adapter = get_adapter(
            group.backend,
            sif_path=self.image_paths[group.backend],
            image_digest=self.image_digests.get(group.backend),
            backend_version=self.backend_versions.get(group.backend),
        )

        # Capability gate, before any GPU time is spent on this group.
        reasons = adapter.unsupported_reasons(group.engine)
        if reasons:
            log.warning("group_unsupported", group=group.label, reasons=reasons)
            return [
                self._write(
                    group, phase, rep, warmup,
                    status="unsupported",
                    measurements={"unsupported_reasons": reasons},
                    backend_version=adapter.backend_version,
                    error="; ".join(reasons),
                )
                for phase in group.phases
                for rep, warmup in self._rep_plan()
            ]

        log_path = self.cfg.log_dir / f"engine_{group.label}_{group.group_id()}.log"
        try:
            with adapter.serve(
                group.engine,
                log_path=log_path,
                gpu_ids=self.cfg.gpu_ids,
                startup_timeout=self.cfg.startup_timeout,
            ) as handle:
                return asyncio.run(self._measure_group(adapter, group, handle))
        except BackendLaunchError as exc:
            # A launch failure is a real result about a real configuration, most
            # often the memory boundary. It is recorded for every phase in the
            # group so the sweep matrix has no silent holes.
            log.error("group_launch_failed", group=group.label, error=str(exc))
            status: RunStatus = "oom" if _looks_like_oom(str(exc)) else "launch_failed"
            return [
                self._write(
                    group, phase, rep, warmup,
                    status=status,
                    measurements={"engine_log_path": str(log_path)},
                    backend_version=adapter.backend_version,
                    error=str(exc)[:8000],
                )
                for phase in group.phases
                for rep, warmup in self._rep_plan()
            ]

    def _rep_plan(self) -> list[tuple[int, bool]]:
        total = self.cfg.warmup_repetitions + self.cfg.repetitions
        return [(rep, rep < self.cfg.warmup_repetitions) for rep in range(total)]

    async def _measure_group(
        self, adapter: BackendAdapter, group: ServerGroup, handle: ServerHandle
    ) -> list[Path]:
        written: list[Path] = []
        tokenizer = _load_tokenizer(group.engine.model_path)

        async with httpx.AsyncClient() as ctl:
            # Verify the engine consumes the exact token IDs we send, before any
            # measurement. If it does not, every input-length and prefix-sharing
            # number from this backend would be wrong, so it is checked rather
            # than assumed.
            fidelity = await validate_prompt_token_fidelity(
                ctl, handle.base_url, group.engine.model_id, list(range(100, 164))
            )
            if not fidelity["token_id_prompts_supported"]:
                log.warning("token_id_prompts_unsupported", backend=group.backend, **fidelity)

            for phase in group.phases:
                generator = WorkloadGenerator(tokenizer, phase.workload)
                requests = generator.generate()
                bound = theoretical_prefix_hit_rate(
                    requests, phase.workload.shared_prefix_tokens
                )

                for rep, warmup in self._rep_plan():
                    reset_info = await adapter.reset_caches(ctl, handle.base_url)
                    # Let the engine settle after a cache flush so the reset cost
                    # is not attributed to the first request's TTFT.
                    await asyncio.sleep(2.0)

                    client = LoadClient(handle.base_url, group.engine.model_id)
                    started = time.time()
                    try:
                        if phase.load.mode is LoadMode.CLOSED_LOOP:
                            load = await client.run_closed_loop(
                                requests, phase.load.concurrency or 1
                            )
                        else:
                            load = await client.run_open_loop(
                                requests,
                                phase.load.request_rate or 1.0,
                                burstiness=phase.load.burstiness,
                            )
                    except Exception as exc:  # recorded as data, never swallowed
                        log.error(
                            "phase_failed",
                            group=group.label,
                            phase=phase.label,
                            error=str(exc),
                        )
                        written.append(
                            self._write(
                                group, phase, rep, warmup,
                                status="failed",
                                measurements={
                                    "elapsed_s": time.time() - started,
                                    **reset_info,
                                },
                                backend_version=handle.backend_version,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue

                    metrics = compute_metrics(load)
                    telemetry = await adapter.engine_telemetry(ctl, handle.base_url)

                    measurements: dict[str, Any] = {
                        **metrics,
                        "engine_telemetry": telemetry,
                        "theoretical_prefix_hit_rate_bound": bound,
                        "prompt_token_fidelity": fidelity,
                        "engine_startup_s": handle.startup_seconds,
                        "engine_launch_command": handle.launch_command,
                        "engine_log_path": str(handle.log_path) if handle.log_path else None,
                        **reset_info,
                        # Per-request timings are kept in full. Percentiles can be
                        # recomputed from raw artifacts without rerunning, and a
                        # figure that needs a different statistic later does not
                        # cost another allocation.
                        "per_request": [
                            {
                                "index": r.index,
                                "success": r.success,
                                "prefix_group": r.prefix_group,
                                "ttft_s": r.ttft,
                                "e2e_s": r.e2e_latency,
                                "send_time_s": r.send_time,
                                "prompt_tokens": r.prompt_tokens_sent,
                                "output_tokens": r.output_tokens_received,
                                "http_status": r.http_status,
                                "error": r.error,
                            }
                            for r in load.results
                        ],
                    }

                    status: RunStatus = "ok" if metrics["requests_ok"] else "failed"
                    written.append(
                        self._write(
                            group, phase, rep, warmup,
                            status=status,
                            measurements=measurements,
                            backend_version=handle.backend_version,
                            error=None if status == "ok" else "no requests succeeded",
                        )
                    )
                    log.info(
                        "phase_complete",
                        group=group.label,
                        phase=phase.label,
                        rep=rep,
                        warmup=warmup,
                        ok=metrics["requests_ok"],
                        failed=metrics["requests_failed"],
                        tput=metrics.get("output_tokens_per_s"),
                    )
        return written


def _looks_like_oom(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda out of memory",
            "no available memory for the cache blocks",
            "torch.outofmemoryerror",
        )
    )
