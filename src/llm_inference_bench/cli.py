"""Command-line entry point.

``expand`` is deliberately separate from ``run`` and costs nothing. A sweep is
planned and budget-checked on the cpu partition before any GPU allocation is
requested, because the monthly GPU allocation is 117 node-hours and a
mis-specified matrix is an expensive way to find that out.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import structlog

from .runner import RunnerConfig, SweepRunner
from .sweep import SweepDefinition, assign_ports


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(message)s", stream=sys.stdout, level=logging.DEBUG if verbose else logging.INFO
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=True),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def _repo_root() -> Path:
    return Path(os.environ.get("LIB_REPO", Path(__file__).resolve().parents[2]))


def _load_lockfile(repo: Path) -> dict:
    path = Path(os.environ.get("LIB_LOCKFILE", repo / "provision.lock.json"))
    if not path.exists():
        raise SystemExit(
            f"provisioning lockfile not found at {path}; run provision.sh first. "
            "Results must trace to pinned image digests and model revisions."
        )
    return json.loads(path.read_text())


def _parse_notes(pairs: list[str]) -> dict[str, str]:
    notes: dict[str, str] = {}
    for item in pairs:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--note expects KEY=VALUE, got {item!r}")
        notes[key.strip()] = value.strip()
    return notes


def cmd_expand(args: argparse.Namespace) -> int:
    definition = SweepDefinition.from_yaml(args.config)
    groups = assign_ports(definition.expand())

    total_seconds = sum(g.estimated_gpu_seconds() for g in groups)
    reps = definition.repetitions + definition.warmup_repetitions
    total_seconds *= reps

    print(f"\nsweep: {definition.name}")
    print(f"  server groups:      {len(groups)}")
    print(f"  phases per group:   {len(groups[0].phases) if groups else 0}")
    print(
        f"  repetitions:        {definition.repetitions} "
        f"(+{definition.warmup_repetitions} warmup)"
    )
    print(f"  total measurements: {sum(len(g.phases) for g in groups) * reps}")
    print("\ngroups:")
    for i, g in enumerate(groups):
        print(
            f"  [{i:3d}] {g.label:<44} port={g.engine.port} "
            f"phases={len(g.phases)} id={g.group_id()}"
        )

    hours = total_seconds / 3600.0
    print(f"\n  ESTIMATED cost: {hours:.1f} GPU node-hours (crude: startup + 120s/phase)")
    print("  This is a planning estimate, not a measurement.")
    if hours > 15:
        print(
            "\n  WARNING: exceeds the ~15 node-hour single-sweep threshold in CLAUDE.md.\n"
            "  Have the design sanity-checked before submitting."
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    repo = _repo_root()
    lock = _load_lockfile(repo)
    definition = SweepDefinition.from_yaml(args.config)
    groups = assign_ports(definition.expand())

    if args.group_index is not None:
        if not 0 <= args.group_index < len(groups):
            raise SystemExit(
                f"group index {args.group_index} out of range (0..{len(groups) - 1})"
            )
        groups = [groups[args.group_index]]

    images = lock.get("images", {})
    image_paths = {name: Path(info["sif"]) for name, info in images.items() if "sif" in info}
    image_digests = {name: info.get("digest") for name, info in images.items()}

    missing = {g.backend for g in groups} - set(image_paths)
    if missing:
        raise SystemExit(f"no provisioned image for backend(s): {sorted(missing)}")

    cfg = RunnerConfig(
        repo_root=repo,
        results_raw=args.results_raw or (repo / "results" / "raw"),
        log_dir=args.log_dir or Path(os.environ.get("LIB_LOGS", repo / "logs")),
        repetitions=definition.repetitions,
        warmup_repetitions=definition.warmup_repetitions,
        gpu_ids=[int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids else None,
        sweep_name=definition.name,
        notes=_parse_notes(args.note),
    )
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    runner = SweepRunner(cfg, image_paths, image_digests)
    written: list[Path] = []
    for group in groups:
        written.extend(runner.run_group(group))

    print(f"\nwrote {len(written)} raw artifacts to {cfg.results_raw}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lib-run", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p_expand = sub.add_parser("expand", help="plan a sweep and estimate its cost, no GPU needed")
    p_expand.add_argument("config", type=Path)
    p_expand.set_defaults(func=cmd_expand)

    p_run = sub.add_parser("run", help="execute a sweep and write raw artifacts")
    p_run.add_argument("config", type=Path)
    p_run.add_argument(
        "--group-index",
        type=int,
        default=None,
        help="run only this server group; the Slurm array task id maps here",
    )
    p_run.add_argument("--gpu-ids", type=str, default=None,
                       help="comma-separated CUDA_VISIBLE_DEVICES for packed single-GPU runs")
    p_run.add_argument(
        "--note",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="tag recorded in every artifact's provenance, e.g. --note placement=packed",
    )
    p_run.add_argument("--results-raw", type=Path, default=None)
    p_run.add_argument("--log-dir", type=Path, default=None)
    p_run.set_defaults(func=cmd_run)

    args = ap.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
