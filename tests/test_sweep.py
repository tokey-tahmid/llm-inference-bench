"""Sweep expansion tests.

The server-axis / phase-axis split is a budget decision with a correctness
hazard attached, so both halves are pinned: phases must group under one server,
and axes that genuinely require a new engine must not be collapsed into one.
"""

from __future__ import annotations

import textwrap

import pytest

from llm_inference_bench.sweep import (
    PHASE_AXES,
    SERVER_AXES,
    SweepDefinition,
    assign_ports,
)

MINIMAL = """
name: t
repetitions: 3
warmup_repetitions: 1
models:
  m:
    path: /models/m
    revision: abc123
model: m
engine:
  max_model_len: 4096
workload:
  num_requests: 32
  input_len: 128
  output_len: 16
axes:
  backend: [vllm]
  tensor_parallel_size: [1, 2]
  load:
    - {mode: closed_loop, concurrency: 1}
    - {mode: closed_loop, concurrency: 8}
"""


def _write(tmp_path, text: str):
    p = tmp_path / "sweep.yaml"
    p.write_text(textwrap.dedent(text))
    return p


def test_phases_group_under_one_server(tmp_path) -> None:
    """Two concurrency points must not cost two model loads."""
    groups = SweepDefinition.from_yaml(_write(tmp_path, MINIMAL)).expand()
    assert len(groups) == 2  # one per TP degree, not per (TP x concurrency)
    assert all(len(g.phases) == 2 for g in groups)


def test_server_axes_do_not_collapse(tmp_path) -> None:
    """TP degree changes the engine and must produce separate launches."""
    groups = SweepDefinition.from_yaml(_write(tmp_path, MINIMAL)).expand()
    tps = sorted(g.engine.tensor_parallel_size for g in groups)
    assert tps == [1, 2]
    assert len({g.group_id() for g in groups}) == 2


def test_prefix_caching_is_a_server_axis() -> None:
    """Toggling it requires a relaunch; treating it as a phase axis would be wrong."""
    assert "enable_prefix_caching" in SERVER_AXES
    assert "enable_prefix_caching" not in PHASE_AXES


def test_load_and_workload_are_phase_axes() -> None:
    assert {"load", "workload"} == set(PHASE_AXES)


def test_group_id_ignores_port(tmp_path) -> None:
    """Two servers differing only in port are the same configuration."""
    groups = SweepDefinition.from_yaml(_write(tmp_path, MINIMAL)).expand()
    before = [g.group_id() for g in groups]
    after = [g.group_id() for g in assign_ports(groups)]
    assert before == after


def test_assign_ports_gives_distinct_ports(tmp_path) -> None:
    """Needed when TP=1 groups are packed several to a billed node."""
    groups = assign_ports(SweepDefinition.from_yaml(_write(tmp_path, MINIMAL)).expand())
    ports = [g.engine.port for g in groups]
    assert len(set(ports)) == len(ports)


def test_unknown_axis_is_rejected(tmp_path) -> None:
    """A typo'd axis would otherwise be silently ignored and shrink the sweep."""
    bad = MINIMAL.replace("  backend: [vllm]", "  bakcend: [vllm]")
    with pytest.raises(ValueError, match="unknown sweep axes"):
        SweepDefinition.from_yaml(_write(tmp_path, bad))


def test_exclude_removes_impossible_combinations(tmp_path) -> None:
    text = MINIMAL + textwrap.dedent(
        """
        exclude:
          - tensor_parallel_size: 1
        """
    )
    groups = SweepDefinition.from_yaml(_write(tmp_path, text)).expand()
    assert [g.engine.tensor_parallel_size for g in groups] == [2]


def test_model_without_path_is_rejected(tmp_path) -> None:
    """An implicit path could resolve over the network and break offline runs."""
    text = MINIMAL.replace("    path: /models/m\n", "")
    with pytest.raises(ValueError, match="no 'path'"):
        SweepDefinition.from_yaml(_write(tmp_path, text)).expand()


def test_revision_is_carried_onto_the_group(tmp_path) -> None:
    groups = SweepDefinition.from_yaml(_write(tmp_path, MINIMAL)).expand()
    assert all(g.model_revision == "abc123" for g in groups)


def test_labels_describe_the_configuration(tmp_path) -> None:
    groups = SweepDefinition.from_yaml(_write(tmp_path, MINIMAL)).expand()
    labels = {g.label for g in groups}
    assert labels == {"vllm_m_tp1", "vllm_m_tp2"}


def test_closed_loop_requires_concurrency(tmp_path) -> None:
    bad = MINIMAL.replace("    - {mode: closed_loop, concurrency: 1}", "    - {mode: closed_loop}")
    with pytest.raises(ValueError, match="closed_loop load requires concurrency"):
        SweepDefinition.from_yaml(_write(tmp_path, bad)).expand()


def test_open_loop_requires_request_rate(tmp_path) -> None:
    bad = MINIMAL.replace("    - {mode: closed_loop, concurrency: 1}", "    - {mode: open_loop}")
    with pytest.raises(ValueError, match="open_loop load requires request_rate"):
        SweepDefinition.from_yaml(_write(tmp_path, bad)).expand()
