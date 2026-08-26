"""YAML sweep definition and expansion.

The central structural decision: sweep axes are split into **server axes** and
**phase axes**, and the split is what makes the node-hour budget survive contact
with a real matrix.

Launching an engine is expensive. A 32B model at TP=4 loading from Lustre with
CUDA graph capture costs minutes, and that cost is paid per *server*, not per
measurement. Concurrency, request rate and workload shape do not require a new
server; tensor-parallel degree, prefix caching, block size and speculative
decoding do. So a sweep is expanded into server groups, each launched once, with
every compatible measurement phase run against it. On a matrix of 4 concurrency
points x 3 repetitions that is 12 model loads collapsed into 1.

The correctness cost of that optimisation is real and is handled explicitly:
phases sharing a server also share KV and prefix cache state, so a later phase
would start warm and post a flattering TTFT. The runner resets the engine's
prefix cache between phases and records whether the reset was actually
acknowledged. A phase whose reset failed is marked, not quietly reported.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import structlog
import yaml

from .backends.base import EngineConfig, SpecDecodeConfig, SpecDecodeMode
from .client import LoadMode
from .workload.generator import (
    LengthDistribution,
    LengthSpec,
    PromptFormat,
    WorkloadSpec,
)

log = structlog.get_logger(__name__)

# Axes that change the engine and therefore require a fresh server launch.
SERVER_AXES = frozenset(
    {
        "backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "enable_prefix_caching",
        "block_size",
        "kv_cache_dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_model_len",
        "spec_decode",
        "model",
    }
)

# Axes that can be varied against an already-running server.
PHASE_AXES = frozenset({"load", "workload"})


@dataclass(frozen=True)
class LoadSpec:
    """How to offer load for one measurement phase."""

    mode: LoadMode = LoadMode.CLOSED_LOOP
    concurrency: int | None = None
    request_rate: float | None = None
    burstiness: float = 1.0

    def __post_init__(self) -> None:
        if self.mode is LoadMode.CLOSED_LOOP and not self.concurrency:
            raise ValueError("closed_loop load requires concurrency")
        if self.mode is LoadMode.OPEN_LOOP and not self.request_rate:
            raise ValueError("open_loop load requires request_rate")

    def label(self) -> str:
        if self.mode is LoadMode.CLOSED_LOOP:
            return f"closed_c{self.concurrency}"
        return f"open_r{self.request_rate:g}"


@dataclass(frozen=True)
class Phase:
    """One measurement against a running server."""

    load: LoadSpec
    workload: WorkloadSpec
    label: str

    def phase_id(self) -> str:
        payload = json.dumps(
            {"load": self.load.label(), "workload": self.workload.fingerprint()},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class ServerGroup:
    """One engine launch and every phase measured against it."""

    backend: str
    engine: EngineConfig
    phases: list[Phase]
    model_repo: str
    model_revision: str | None
    label: str

    def group_id(self) -> str:
        payload = json.dumps(
            {"backend": self.backend, "engine": _engine_fingerprint(self.engine)},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def estimated_gpu_seconds(self, per_phase_seconds: float = 120.0) -> float:
        """Rough cost, for the pre-flight budget check.

        Deliberately crude and deliberately stated as an estimate: it exists to
        catch a sweep that would blow the monthly allocation, not to be reported
        as a measurement. Startup is the dominant fixed cost at small phase counts.
        """
        startup = 300.0 if self.engine.tensor_parallel_size > 1 else 180.0
        return startup + per_phase_seconds * len(self.phases)


def _engine_fingerprint(cfg: EngineConfig) -> dict[str, Any]:
    """Engine fields that actually change the served configuration.

    Port and host are excluded: two servers differing only in port are the same
    configuration, and including them would defeat the grouping entirely.
    """
    d = cfg.as_dict()
    d.pop("port", None)
    d.pop("host", None)
    return d


# --- YAML parsing -----------------------------------------------------------


def _parse_length_spec(raw: Any, default_mean: int) -> LengthSpec:
    if raw is None:
        return LengthSpec(mean=default_mean)
    if isinstance(raw, int):
        return LengthSpec(distribution=LengthDistribution.FIXED, mean=raw)
    if not isinstance(raw, dict):
        raise ValueError(f"length spec must be an int or a mapping, got {type(raw).__name__}")
    return LengthSpec(
        distribution=LengthDistribution(raw.get("distribution", "fixed")),
        mean=int(raw.get("mean", default_mean)),
        sigma=float(raw.get("sigma", 0.0)),
        low=raw.get("low"),
        high=raw.get("high"),
    )


def _parse_workload(raw: dict[str, Any], defaults: dict[str, Any]) -> WorkloadSpec:
    merged = {**defaults, **raw}
    return WorkloadSpec(
        num_requests=int(merged.get("num_requests", 128)),
        input_len=_parse_length_spec(merged.get("input_len"), 1024),
        output_len=_parse_length_spec(merged.get("output_len"), 128),
        shared_prefix_ratio=float(merged.get("shared_prefix_ratio", 0.0)),
        shared_prefix_tokens=int(merged.get("shared_prefix_tokens", 512)),
        num_prefix_groups=int(merged.get("num_prefix_groups", 1)),
        seed=int(merged.get("seed", 1234)),
        prompt_format=PromptFormat(merged.get("prompt_format", "token_ids")),
    )


def _parse_load(raw: dict[str, Any]) -> LoadSpec:
    return LoadSpec(
        mode=LoadMode(raw.get("mode", "closed_loop")),
        concurrency=raw.get("concurrency"),
        request_rate=raw.get("request_rate"),
        burstiness=float(raw.get("burstiness", 1.0)),
    )


def _parse_spec_decode(raw: Any) -> SpecDecodeConfig:
    if raw in (None, "off", False):
        return SpecDecodeConfig()
    if isinstance(raw, str):
        return SpecDecodeConfig(mode=SpecDecodeMode(raw))
    return SpecDecodeConfig(
        mode=SpecDecodeMode(raw.get("mode", "off")),
        num_speculative_tokens=int(raw.get("num_speculative_tokens", 0)),
        draft_model=raw.get("draft_model"),
        ngram_prompt_lookup_min=int(raw.get("ngram_prompt_lookup_min", 1)),
        ngram_prompt_lookup_max=int(raw.get("ngram_prompt_lookup_max", 4)),
    )


@dataclass
class SweepDefinition:
    """A parsed sweep, before expansion."""

    name: str
    repetitions: int
    warmup_repetitions: int
    models: dict[str, dict[str, str]]
    engine_defaults: dict[str, Any]
    workload_defaults: dict[str, Any]
    server_axes: dict[str, list[Any]]
    phase_axes: dict[str, list[Any]]
    exclude: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> SweepDefinition:
        raw = yaml.safe_load(path.read_text())
        axes = raw.get("axes", {})

        unknown = set(axes) - SERVER_AXES - PHASE_AXES
        if unknown:
            raise ValueError(
                f"unknown sweep axes {sorted(unknown)}; "
                f"server axes are {sorted(SERVER_AXES)}, phase axes are {sorted(PHASE_AXES)}"
            )

        return cls(
            name=raw["name"],
            repetitions=int(raw.get("repetitions", 3)),
            warmup_repetitions=int(raw.get("warmup_repetitions", 1)),
            models=raw.get("models", {}),
            engine_defaults=raw.get("engine", {}),
            workload_defaults=raw.get("workload", {}),
            server_axes={k: v for k, v in axes.items() if k in SERVER_AXES},
            phase_axes={k: v for k, v in axes.items() if k in PHASE_AXES},
            exclude=raw.get("exclude", []),
            raw=raw,
        )

    def _excluded(self, combo: dict[str, Any]) -> bool:
        """Whether a server-axis combination is explicitly ruled out.

        Used for combinations that are known-impossible rather than merely
        untested, e.g. a 32B model at TP=1 which cannot fit in 40GB. Excluding it
        is honest bookkeeping; the README says why it is absent.
        """
        for rule in self.exclude:
            if all(combo.get(k) == v for k, v in rule.items()):
                return True
        return False

    def expand(self) -> list[ServerGroup]:
        """Cartesian product over server axes, each carrying all phase combinations."""
        server_keys = sorted(self.server_axes)
        server_values = [self.server_axes[k] for k in server_keys]

        phases = self._expand_phases()
        groups: list[ServerGroup] = []

        for combo_vals in itertools.product(*server_values) if server_keys else [()]:
            combo = dict(zip(server_keys, combo_vals, strict=True))
            if self._excluded(combo):
                log.info("sweep_combo_excluded", **combo)
                continue

            backend = combo.get("backend", self.raw.get("backend", "vllm"))
            model_key = combo.get("model", self.raw.get("model"))
            if model_key is None:
                raise ValueError("sweep must specify a model, as an axis or a top-level key")
            model_info = self.models.get(model_key, {})
            model_path = model_info.get("path")
            if not model_path:
                raise ValueError(
                    f"model {model_key!r} has no 'path' in the sweep's models: block; "
                    "paths must be explicit so a run cannot silently hit the network"
                )

            engine = EngineConfig(
                model_path=model_path,
                model_id=model_key,
                model_revision=model_info.get("revision"),
                tensor_parallel_size=int(
                    combo.get("tensor_parallel_size", self.engine_defaults.get("tensor_parallel_size", 1))
                ),
                pipeline_parallel_size=int(
                    combo.get("pipeline_parallel_size", self.engine_defaults.get("pipeline_parallel_size", 1))
                ),
                max_model_len=combo.get("max_model_len", self.engine_defaults.get("max_model_len")),
                max_num_seqs=int(combo.get("max_num_seqs", self.engine_defaults.get("max_num_seqs", 256))),
                gpu_memory_utilization=float(
                    combo.get(
                        "gpu_memory_utilization",
                        self.engine_defaults.get("gpu_memory_utilization", 0.90),
                    )
                ),
                block_size=combo.get("block_size", self.engine_defaults.get("block_size")),
                kv_cache_dtype=str(
                    combo.get("kv_cache_dtype", self.engine_defaults.get("kv_cache_dtype", "auto"))
                ),
                dtype=str(self.engine_defaults.get("dtype", "bfloat16")),
                enable_prefix_caching=bool(
                    combo.get(
                        "enable_prefix_caching",
                        self.engine_defaults.get("enable_prefix_caching", False),
                    )
                ),
                spec_decode=_parse_spec_decode(
                    combo.get("spec_decode", self.engine_defaults.get("spec_decode"))
                ),
                seed=int(self.engine_defaults.get("seed", 0)),
            )

            label_bits = [backend, model_key.split("/")[-1], f"tp{engine.tensor_parallel_size}"]
            if engine.enable_prefix_caching:
                label_bits.append("apc")
            if engine.spec_decode.mode is not SpecDecodeMode.OFF:
                label_bits.append(f"spec-{engine.spec_decode.mode}")
            if engine.block_size:
                label_bits.append(f"blk{engine.block_size}")
            if engine.kv_cache_dtype != "auto":
                label_bits.append(f"kv{engine.kv_cache_dtype}")

            groups.append(
                ServerGroup(
                    backend=backend,
                    engine=engine,
                    phases=phases,
                    model_repo=model_key,
                    model_revision=model_info.get("revision"),
                    label="_".join(label_bits),
                )
            )

        log.info(
            "sweep_expanded",
            name=self.name,
            server_groups=len(groups),
            phases_per_group=len(phases),
            repetitions=self.repetitions,
            warmup=self.warmup_repetitions,
            total_measurements=len(groups) * len(phases) * self.repetitions,
        )
        return groups

    def _expand_phases(self) -> list[Phase]:
        loads = [_parse_load(x) for x in self.phase_axes.get("load", [{}])]
        workload_variants = self.phase_axes.get("workload", [{}])
        phases: list[Phase] = []
        for wl_raw in workload_variants:
            workload = _parse_workload(wl_raw, self.workload_defaults)
            for load in loads:
                bits = [load.label()]
                if workload.shared_prefix_ratio:
                    bits.append(f"pfx{workload.shared_prefix_ratio:g}")
                if isinstance(wl_raw, dict) and "label" in wl_raw:
                    bits.append(str(wl_raw["label"]))
                phases.append(Phase(load=load, workload=workload, label="_".join(bits)))
        return phases


def assign_ports(groups: list[ServerGroup], base_port: int = 8100) -> list[ServerGroup]:
    """Give each group a distinct port.

    Matters when several servers run concurrently on one node, which is how TP=1
    configurations are packed 4-per-node to avoid wasting three quarters of a
    billed 4-GPU node.
    """
    return [
        replace_group_port(g, base_port + i) for i, g in enumerate(groups)
    ]


def replace_group_port(group: ServerGroup, port: int) -> ServerGroup:
    group.engine = replace(group.engine, port=port)
    return group


def iter_measurements(
    groups: list[ServerGroup], repetitions: int, warmup_repetitions: int
) -> Iterator[tuple[ServerGroup, Phase, int, bool]]:
    """Yield every (group, phase, repetition_index, is_warmup) tuple.

    Warmup repetitions come first within a phase and are yielded with
    ``is_warmup=True``. They are recorded as artifacts and excluded in analysis,
    per the methodology: the first run of any config pays CUDA graph capture,
    autotuning and cache population, and folding that into the reported number
    would understate steady-state performance.
    """
    for group in groups:
        for phase in group.phases:
            for rep in range(warmup_repetitions + repetitions):
                yield group, phase, rep, rep < warmup_repetitions
