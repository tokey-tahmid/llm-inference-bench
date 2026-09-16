"""Integrity-scoping tests for the phase-1 figure.

The bug being pinned: ``plot_phase1.py`` scopes its data frame to
``sweep_name == phase1-validation`` but historically passed the GLOBAL
``LoadReport`` into ``check_integrity``. So a clean 8-artifact phase-1 figure
would carry a warning like "1015 run(s) came from a dirty git tree" that
described the rest of ``results/raw/`` and nothing about the figure itself.

A figure scoped to one sweep must only accuse itself of its own problems.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from plot_phase1 import check_integrity


def _row(**overrides) -> dict:
    base = {
        "sweep_name": "phase1-validation",
        "status": "ok",
        "warmup": False,
        "git_sha": "abc1234",
        "output_length_exact": True,
        "prompt_length_exact": True,
        "cache_reset_ok": True,
        "client_may_be_bottleneck": False,
    }
    base.update(overrides)
    return base


def test_dirty_runs_from_another_sweep_do_not_taint_this_figure() -> None:
    """The scoped frame contains only its own sweep, and only its own runs
    should influence the integrity warnings."""
    scoped = pd.DataFrame(
        [
            _row(warmup=True),
            _row(),
            _row(),
        ]
    )
    problems = check_integrity(scoped)
    assert not any("dirty git tree" in p for p in problems), problems


def test_dirty_run_in_scope_is_reported() -> None:
    scoped = pd.DataFrame(
        [
            _row(warmup=True),
            _row(git_sha="abc1234-dirty"),
            _row(),
        ]
    )
    problems = check_integrity(scoped)
    assert any("1 run(s) came from a dirty git tree" in p for p in problems), problems


def test_missing_warmup_is_reported() -> None:
    scoped = pd.DataFrame([_row(), _row()])
    problems = check_integrity(scoped)
    assert any("no warmup repetitions found" in p for p in problems), problems


def test_warmup_present_does_not_trigger_warmup_warning() -> None:
    scoped = pd.DataFrame([_row(warmup=True), _row(), _row()])
    problems = check_integrity(scoped)
    assert not any("no warmup repetitions found" in p for p in problems), problems


def test_warmup_ok_row_is_not_counted_as_failure() -> None:
    """A warmup row is not a failed run; it is a run deliberately excluded."""
    scoped = pd.DataFrame([_row(warmup=True, status="ok"), _row(), _row()])
    problems = check_integrity(scoped)
    assert not any("non-ok runs present" in p for p in problems), problems


def test_output_length_mismatch_flags_uncontrolled_axis() -> None:
    scoped = pd.DataFrame(
        [
            _row(warmup=True),
            _row(output_length_exact=False),
            _row(),
        ]
    )
    problems = check_integrity(scoped)
    assert any("output lengths did not match" in p for p in problems), problems
