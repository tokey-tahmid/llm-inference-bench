#!/usr/bin/env python3
"""Reconcile declared adapter capabilities against the pinned container images.

Run by ``probe_capabilities.sh`` after it captures ``--help`` from each image.
Writes ``docs/capabilities_<backend>.json`` and records backend versions in the
provisioning lockfile.

The point is to make the sweep matrix honest before it runs. Three outcomes:

* a flag the adapter emits is **missing** from the image -> that configuration
  would fail at launch, so it is reported loudly and must be fixed in the adapter
* a capability is **declared but unsupported** by the image -> the adapter is
  overclaiming and the sweep would record misleading ``unsupported`` artifacts
* a capability is **undeclared but available** -> the adapter is underclaiming
  and the sweep is leaving real coverage on the table

None of this guesses. Everything is read out of the image's own help text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from llm_inference_bench.backends import Capability, EngineConfig, get_adapter
from llm_inference_bench.backends.base import SpecDecodeConfig, SpecDecodeMode

# Flags whose presence in the help text evidences a capability. Kept as a table
# rather than scattered through the adapters so that adding a backend means
# adding a column here, not editing probe logic.
CAPABILITY_EVIDENCE: dict[str, dict[Capability, tuple[str, ...]]] = {
    "vllm": {
        Capability.PREFIX_CACHING: ("--enable-prefix-caching", "--no-enable-prefix-caching"),
        Capability.BLOCK_SIZE: ("--block-size",),
        Capability.KV_CACHE_DTYPE_FP8: ("--kv-cache-dtype",),
        Capability.TENSOR_PARALLEL: ("--tensor-parallel-size",),
        Capability.SPEC_DECODE_NGRAM: ("--speculative-config", "--speculative-model"),
        Capability.SPEC_DECODE_DRAFT_MODEL: ("--speculative-config", "--speculative-model"),
        Capability.MULTI_NODE: ("--distributed-executor-backend",),
    },
    "sglang": {
        Capability.PREFIX_CACHING: ("--disable-radix-cache",),
        Capability.TENSOR_PARALLEL: ("--tp-size", "--tp"),
        Capability.KV_CACHE_DTYPE_FP8: ("--kv-cache-dtype",),
        Capability.SPEC_DECODE_DRAFT_MODEL: ("--speculative-draft-model-path",),
        Capability.SPEC_DECODE_NGRAM: ("--speculative-algorithm",),
        Capability.PROMETHEUS_METRICS: ("--enable-metrics",),
        Capability.MULTI_NODE: ("--dist-init-addr", "--nnodes"),
    },
}

# Configurations whose launch flags we want verified. Chosen to exercise every
# axis the sweep will actually use, so that a rename anywhere in the matrix is
# caught by this one job rather than by a failed allocation.
def probe_configs() -> dict[str, EngineConfig]:
    base = {
        "model_path": "/models/placeholder",
        "model_id": "placeholder",
        "max_model_len": 4096,
    }
    return {
        "baseline": EngineConfig(**base),
        "prefix_on": EngineConfig(**base, enable_prefix_caching=True),
        "tp4": EngineConfig(**base, tensor_parallel_size=4),
        "block_size": EngineConfig(**base, block_size=16),
        "fp8_kv": EngineConfig(**base, kv_cache_dtype="fp8"),
        "spec_ngram": EngineConfig(
            **base,
            spec_decode=SpecDecodeConfig(mode=SpecDecodeMode.NGRAM, num_speculative_tokens=4),
        ),
    }


def flag_present(help_text: str, flag: str) -> bool:
    """Whether an argparse help text advertises ``flag``.

    Matched on a word boundary so that ``--tp`` does not spuriously match
    ``--tp-size``, which would report a capability the image does not have.
    """
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def help_text_is_usable(help_text: str) -> tuple[bool, str]:
    """Whether the captured text is real help output rather than a failure.

    Worth a guard of its own. When the capture is a traceback, *every* flag looks
    absent, including universally present ones like ``--host``, and the report
    fills with confident-looking OVERCLAIM findings that are pure noise. Job
    5146938 produced exactly that: the real cause was torch failing to create a
    cache directory on read-only Lustre at import time, nothing to do with flags.

    The sanity check is that a genuine ``--help`` must advertise the few flags
    every server has.
    """
    if help_text.lstrip().startswith("Traceback"):
        last = [ln for ln in help_text.strip().splitlines() if ln.strip()][-1]
        return False, f"capture is a traceback, not help text: {last.strip()}"
    if "usage:" not in help_text.lower() and "--help" not in help_text:
        return False, "capture contains neither a usage line nor --help"
    universal = ["--host", "--port"]
    missing = [f for f in universal if not flag_present(help_text, f)]
    if missing:
        return False, (
            f"capture is missing universally present flags {missing}, "
            "so it is almost certainly truncated or an error page"
        )
    return True, ""


def reconcile(backend: str, help_text: str) -> dict[str, Any]:
    adapter = get_adapter(backend, sif_path=Path("unused"))
    report: dict[str, Any] = {
        "backend": backend,
        "help_lines": help_text.count("\n"),
        "declared_capabilities": sorted(str(c) for c in adapter.capabilities),
        "config_flag_check": {},
        "capability_evidence": {},
        "problems": [],
    }

    usable, why = help_text_is_usable(help_text)
    report["help_usable"] = usable
    if not usable:
        report["help_unusable_reason"] = why
        report["problems"].append(
            f"CANNOT RECONCILE: {why}. Fix the capture before trusting any "
            "capability result for this backend."
        )
        return report

    # 1. Every flag the adapter would actually emit must exist in the image.
    for label, cfg in probe_configs().items():
        try:
            argv = adapter.server_argv(cfg)
        except (NotImplementedError, ValueError) as exc:
            report["config_flag_check"][label] = {"skipped": str(exc)}
            continue
        flags = [a for a in argv if a.startswith("--")]
        missing = [f for f in flags if not flag_present(help_text, f)]
        report["config_flag_check"][label] = {"emitted": flags, "missing": missing}
        if missing:
            report["problems"].append(
                f"config {label!r} emits flags absent from the image: {missing}"
            )

    # 2. Declared capabilities must be evidenced by the image.
    evidence = CAPABILITY_EVIDENCE.get(backend, {})
    for cap, flags in evidence.items():
        found = [f for f in flags if flag_present(help_text, f)]
        supported = bool(found)
        declared = cap in adapter.capabilities
        report["capability_evidence"][str(cap)] = {
            "declared": declared,
            "evidenced": supported,
            "matching_flags": found,
        }
        if declared and not supported:
            report["problems"].append(
                f"OVERCLAIM: adapter declares {cap} but image shows none of {list(flags)}"
            )
        elif supported and not declared:
            report["problems"].append(
                f"UNDERCLAIM: image supports {cap} (via {found}) but adapter does not declare it"
            )

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--lockfile", type=Path, required=True)
    args = ap.parse_args()

    lock: dict[str, Any] = {}
    if args.lockfile.exists():
        lock = json.loads(args.lockfile.read_text())

    exit_code = 0
    for backend in ("vllm", "sglang"):
        help_path = args.docs / f"help_{backend}.txt"
        if not help_path.exists():
            print(f"[reconcile] {backend}: no help text captured, skipping")
            continue
        help_text = help_path.read_text(errors="replace")
        report = reconcile(backend, help_text)

        version_path = args.docs / f"version_{backend}.txt"
        if version_path.exists():
            version = version_path.read_text().strip()
            report["version"] = version
            # Never record a placeholder as provenance. Every raw artifact copies
            # backend_version out of this lockfile, so writing "UNKNOWN" here
            # would stamp a fake value onto real measurements. An absent key is
            # honest; a placeholder is not.
            if version and version != "UNKNOWN" and "Error" not in version:
                lock.setdefault("images", {}).setdefault(backend, {})["version"] = version
            else:
                print(
                    f"[reconcile] {backend}: version unreadable, leaving it out of the "
                    "lockfile rather than recording a placeholder"
                )
                lock.get("images", {}).get(backend, {}).pop("version", None)
        python_path = args.docs / f"python_{backend}.txt"
        if python_path.exists():
            interp = python_path.read_text().strip()
            report["interpreter"] = interp
            lock.setdefault("images", {}).setdefault(backend, {})["interpreter"] = interp

        out = args.docs / f"capabilities_{backend}.json"
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        print(f"\n[reconcile] === {backend} (version {report.get('version', 'UNKNOWN')}) ===")
        if report["problems"]:
            exit_code = 1
            for p in report["problems"]:
                print(f"[reconcile]   PROBLEM: {p}")
        else:
            print("[reconcile]   all declared flags and capabilities verified against the image")

    args.lockfile.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(f"\n[reconcile] lockfile updated: {args.lockfile}")
    if exit_code:
        print("[reconcile] adapters need fixing before the sweep runs")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
