"""Backend adapter interface.

The framing that matters for this project: this is not a vLLM benchmark script,
it is a portable characterisation harness that treats the serving engine as a
swappable backend. Adding a third engine must require only a new subclass of
:class:`BackendAdapter` plus a registry entry, with no change to the workload
generator, the load client, the sweep runner or the analysis.

The split of responsibilities is the whole design:

* The **adapter** owns everything engine-specific: process lifecycle, translating
  a neutral :class:`EngineConfig` into that engine's CLI flags, declaring which
  knobs the engine actually supports, and scraping engine-internal telemetry
  (prefix cache hit rate, speculative acceptance rate, KV cache utilisation)
  that is exposed differently by every engine.
* The **client** (``client.py``) is engine-agnostic. Both vLLM and SGLang serve
  an OpenAI-compatible HTTP API, so all externally observable latency and
  throughput metrics are measured identically against every backend. Nothing
  engine-specific may leak into it, or a backend-vs-backend comparison stops
  being apples to apples.

Capability declaration is not decoration. When a sweep asks for a knob an engine
does not support, the runner records an ``unsupported`` artifact rather than
silently dropping the point or, far worse, substituting the other engine's
number. See the non-negotiable rules in the workspace CLAUDE.md.
"""

from __future__ import annotations

import abc
import contextlib
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)


class Capability(StrEnum):
    """Knobs a backend may or may not support.

    Declared per adapter so the sweep runner can decide, before burning an
    allocation, whether a configuration is measurable on this engine at all.
    """

    PREFIX_CACHING = "prefix_caching"
    PREFIX_CACHE_METRICS = "prefix_cache_metrics"
    SPEC_DECODE_NGRAM = "spec_decode_ngram"
    SPEC_DECODE_DRAFT_MODEL = "spec_decode_draft_model"
    SPEC_DECODE_METRICS = "spec_decode_metrics"
    BLOCK_SIZE = "block_size"
    KV_CACHE_DTYPE_FP8 = "kv_cache_dtype_fp8"
    TENSOR_PARALLEL = "tensor_parallel"
    MULTI_NODE = "multi_node"
    PROMETHEUS_METRICS = "prometheus_metrics"
    IGNORE_EOS = "ignore_eos"


class SpecDecodeMode(StrEnum):
    OFF = "off"
    NGRAM = "ngram"
    DRAFT_MODEL = "draft_model"


@dataclass(frozen=True)
class SpecDecodeConfig:
    mode: SpecDecodeMode = SpecDecodeMode.OFF
    num_speculative_tokens: int = 0
    draft_model: str | None = None
    ngram_prompt_lookup_min: int = 1
    ngram_prompt_lookup_max: int = 4


@dataclass(frozen=True)
class EngineConfig:
    """Engine-neutral serving configuration.

    Every field here must be expressible on every supported backend, or be
    guarded by a :class:`Capability`. Resist adding engine-specific fields:
    that is what ``extra_args`` is for, and anything passed through it is
    recorded in the raw artifact so the deviation stays visible.
    """

    model_path: str
    model_id: str
    model_revision: str | None = None

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    max_num_seqs: int = 256

    gpu_memory_utilization: float = 0.90
    block_size: int | None = None
    kv_cache_dtype: str = "auto"
    dtype: str = "bfloat16"

    enable_prefix_caching: bool = False
    spec_decode: SpecDecodeConfig = field(default_factory=SpecDecodeConfig)

    seed: int = 0
    port: int = 8000
    host: str = "127.0.0.1"

    extra_args: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class ServerHandle:
    """A running engine, plus everything needed to attribute its measurements."""

    base_url: str
    process: subprocess.Popen[bytes] | None
    backend_name: str
    backend_version: str | None
    image_digest: str | None
    launch_command: str
    startup_seconds: float
    log_path: Path | None = None


class BackendLaunchError(RuntimeError):
    """The engine failed to come up. Recorded as a ``launch_failed`` artifact."""


class BackendAdapter(abc.ABC):
    """One serving engine, behind a uniform lifecycle.

    Subclasses implement flag translation, capability declaration and telemetry
    scraping. They do not implement load generation or metric computation.
    """

    name: str
    capabilities: frozenset[Capability]

    def __init__(self, sif_path: Path, image_digest: str | None = None) -> None:
        self.sif_path = Path(sif_path)
        self.image_digest = image_digest

    # -- capability gating ---------------------------------------------------

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def unsupported_reasons(self, cfg: EngineConfig) -> list[str]:
        """Why this engine cannot measure this configuration, if it cannot.

        An empty list means the configuration is measurable. A non-empty list is
        recorded verbatim in an ``unsupported`` artifact so the gap in the sweep
        matrix is visible in the data rather than only in someone's memory.
        """
        reasons: list[str] = []
        if cfg.enable_prefix_caching and not self.supports(Capability.PREFIX_CACHING):
            reasons.append(f"{self.name} does not support prefix caching")
        if cfg.tensor_parallel_size > 1 and not self.supports(Capability.TENSOR_PARALLEL):
            reasons.append(f"{self.name} does not support tensor parallelism")
        if cfg.block_size is not None and not self.supports(Capability.BLOCK_SIZE):
            reasons.append(f"{self.name} does not expose a configurable block size")
        if cfg.kv_cache_dtype.startswith("fp8") and not self.supports(
            Capability.KV_CACHE_DTYPE_FP8
        ):
            reasons.append(f"{self.name} does not support fp8 KV cache")
        mode = cfg.spec_decode.mode
        if mode is SpecDecodeMode.NGRAM and not self.supports(Capability.SPEC_DECODE_NGRAM):
            reasons.append(f"{self.name} does not support n-gram speculative decoding")
        if mode is SpecDecodeMode.DRAFT_MODEL and not self.supports(
            Capability.SPEC_DECODE_DRAFT_MODEL
        ):
            reasons.append(f"{self.name} does not support draft-model speculative decoding")
        return reasons

    # -- engine-specific, implemented per adapter ----------------------------

    @abc.abstractmethod
    def server_argv(self, cfg: EngineConfig) -> list[str]:
        """Translate a neutral config into this engine's server command line."""

    @abc.abstractmethod
    def health_path(self) -> str:
        """Path that returns 2xx once the engine is ready to accept requests."""

    @abc.abstractmethod
    async def engine_telemetry(self, client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
        """Scrape engine-internal counters.

        Returns a dict of *normalised* keys where the concept exists on this
        engine, so analysis code does not branch per backend. Keys absent from
        the dict mean "this engine does not report it", which is different from
        zero and must never be coerced to zero.
        """

    def version_probe_argv(self) -> list[str]:
        """Command that prints the backend version from inside the container."""
        return ["python3", "-c", f"import {self.name}; print({self.name}.__version__)"]

    def reset_cache_endpoint(self) -> str | None:
        """Endpoint that clears prefix/KV cache state, or None if unavailable."""
        return None

    async def reset_caches(self, client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
        """Clear cache state between measurement phases sharing one server.

        Server grouping (see ``sweep.py``) amortises the minutes-long model load
        across many phases. The hazard it creates is that a later phase inherits a
        warm prefix cache from an earlier one and posts a flattering TTFT, which
        would be invisible in the aggregate numbers.

        The outcome is recorded in the artifact rather than assumed. A phase whose
        reset failed is marked ``cache_reset_ok: false`` so analysis can exclude it
        instead of silently reporting a contaminated measurement.
        """
        endpoint = self.reset_cache_endpoint()
        if endpoint is None:
            return {
                "cache_reset_attempted": False,
                "cache_reset_ok": False,
                "cache_reset_note": f"{self.name} exposes no cache reset endpoint",
            }
        try:
            resp = await client.post(f"{base_url}{endpoint}", timeout=60.0)
            ok = resp.status_code < 300
            return {
                "cache_reset_attempted": True,
                "cache_reset_ok": ok,
                "cache_reset_endpoint": endpoint,
                "cache_reset_status": resp.status_code,
            }
        except httpx.HTTPError as exc:
            return {
                "cache_reset_attempted": True,
                "cache_reset_ok": False,
                "cache_reset_endpoint": endpoint,
                "cache_reset_error": str(exc),
            }

    # -- shared lifecycle ----------------------------------------------------

    def container_argv(self, cfg: EngineConfig, gpu_ids: list[int] | None = None) -> list[str]:
        """Wrap the server command in an unprivileged Apptainer invocation.

        No Docker, no root, no subuid on this cluster. Caches are bound to project
        storage explicitly because ``$HOME`` defaults would land on a small,
        contended filesystem.
        """
        data_root = os.environ.get("LIB_DATA", "")
        argv = ["apptainer", "exec", "--nv", "--cleanenv"]

        # Lustre is mounted READ-ONLY inside the container by default on this
        # cluster. Torch resolves an inductor cache directory at *import* time and
        # calls makedirs on it, so simply pointing a cache env var at project
        # storage makes `import vllm` die with
        #   OSError: [Errno 30] Read-only file system: '/mnt/tier2'
        # before argparse ever runs. Confirmed in capability probe job 5146938.
        #
        # Two consequences, both handled here:
        #   1. Anything under LIB_DATA that must be written has to be bound
        #      explicitly read-write.
        #   2. Caches that only matter within a single run go to the node's own
        #      tmpfs instead. These nodes are diskless with a 428 GB /dev/shm, so
        #      that is fast and costs no Lustre traffic. It also keeps compile
        #      caches from being shared across configurations, which would let one
        #      run's autotuning silently warm the next one's startup.
        if data_root:
            argv += ["--bind", f"{data_root}:{data_root}:rw"]

        passthrough = {
            # Read-only is fine and correct for weights.
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
            # Written at import/compile time: keep on node-local tmpfs.
            "TMPDIR": "/tmp",
            "HOME": "/tmp",
            "VLLM_CACHE_ROOT": "/tmp/vllm_cache",
            "TRITON_CACHE_DIR": "/tmp/triton_cache",
            "TORCHINDUCTOR_CACHE_DIR": "/tmp/inductor_cache",
            "TORCH_HOME": "/tmp/torch_home",
            "XDG_CACHE_HOME": "/tmp/xdg_cache",
            "OUTLINES_CACHE_DIR": "/tmp/outlines_cache",
            # Offline once provisioned: a sweep must never depend on network weather,
            # and an accidental re-download mid-sweep would contaminate startup timing.
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        if gpu_ids is not None:
            passthrough["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        for key, val in passthrough.items():
            if val:
                argv += ["--env", f"{key}={val}"]

        argv += [str(self.sif_path)]
        argv += self.server_argv(cfg)
        return argv

    @contextlib.contextmanager
    def serve(
        self,
        cfg: EngineConfig,
        *,
        log_path: Path,
        gpu_ids: list[int] | None = None,
        startup_timeout: float = 900.0,
    ) -> Iterator[ServerHandle]:
        """Launch the engine, yield a handle, and guarantee teardown.

        ``startup_timeout`` is generous by default: a 32B model at TP=4 loading
        from Lustre with CUDA graph capture is genuinely slow, and a too-tight
        timeout would record a spurious ``launch_failed`` for a healthy config.
        """
        argv = self.container_argv(cfg, gpu_ids=gpu_ids)
        cmdline = shlex.join(argv)
        log.info("engine_launch", backend=self.name, cmd=cmdline)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("wb") as logfh:
            proc = subprocess.Popen(
                argv,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                # Own process group, so teardown kills the engine's worker
                # children too. vLLM at TP>1 spawns workers that outlive a bare
                # SIGTERM to the parent and would hold the GPUs for the next run.
                start_new_session=True,
            )
            try:
                base_url = f"http://{cfg.host}:{cfg.port}"
                self._await_ready(proc, base_url, startup_timeout, log_path)
                yield ServerHandle(
                    base_url=base_url,
                    process=proc,
                    backend_name=self.name,
                    backend_version=None,
                    image_digest=self.image_digest,
                    launch_command=cmdline,
                    startup_seconds=time.monotonic() - started,
                    log_path=log_path,
                )
            finally:
                self._terminate(proc)

    def _await_ready(
        self,
        proc: subprocess.Popen[bytes],
        base_url: str,
        timeout: float,
        log_path: Path,
    ) -> None:
        url = base_url + self.health_path()
        deadline = time.monotonic() + timeout
        with httpx.Client(timeout=5.0) as probe:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    tail = _tail(log_path)
                    raise BackendLaunchError(
                        f"{self.name} exited with code {proc.returncode} during startup. "
                        f"Log tail:\n{tail}"
                    )
                try:
                    if probe.get(url).status_code < 300:
                        log.info("engine_ready", backend=self.name, url=base_url)
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(2.0)
        raise BackendLaunchError(
            f"{self.name} not ready within {timeout:.0f}s. Log tail:\n{_tail(log_path)}"
        )

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes], grace: float = 30.0) -> None:
        if proc.poll() is not None:
            return
        pgid = os.getpgid(proc.pid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.5)
        log.warning("engine_sigkill", pid=proc.pid)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGKILL)
        proc.wait(timeout=30)


def _tail(path: Path, lines: int = 40) -> str:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError:
        return "<log unavailable>"
    return "\n".join(content[-lines:])


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse a Prometheus text exposition into ``{metric{labels}: value}``.

    Deliberately minimal: both engines expose plain counters and gauges here, and
    pulling in a full client library for this would be a dependency for nothing.
    Histogram buckets are kept with their ``le`` label intact so percentile
    reconstruction stays possible if it is ever needed.
    """
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        if not name:
            continue
        try:
            out[name.strip()] = float(value)
        except ValueError:
            continue
    return out
