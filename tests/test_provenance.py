"""Provenance tests.

These encode the non-negotiable rules as executable checks. The rule that a raw
artifact is append-only and traceable is worth nothing if it is only a comment,
so the refusal paths are tested directly.
"""

from __future__ import annotations

import json

import pytest

from llm_inference_bench.provenance import (
    IncompleteProvenanceError,
    Provenance,
    write_artifact,
)


def _provenance(**overrides) -> Provenance:
    base = {
        "run_id": "abc123",
        "utc_timestamp": "2026-08-26T10:00:00+00:00",
        "git_sha": "deadbeef",
        "hostname": "mel2017",
        "slurm_job_id": "12345",
        "gpu_model": "NVIDIA A100-SXM4-40GB",
        "gpu_count": 4,
        "gpu_memory_gb": 40.0,
        "driver_version": "595.71.05",
        "cuda_version": "13.2",
        "gpu_topology": "GPU0 NV4 ...",
        "backend_name": "vllm",
        "backend_version": "0.11.0",
        "backend_image_digest": "sha256:0a51ea",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": "a09a35",
        "full_config_dict": {"tp": 1},
        "exact_command_line": "lib-run run configs/x.yaml",
        "repetition_index": 0,
        "warmup": False,
        "status": "ok",
    }
    base.update(overrides)
    return Provenance(**base)


def test_artifact_roundtrips(tmp_path) -> None:
    path = write_artifact(tmp_path, _provenance(), {"output_tokens_per_s": 1234.5})
    payload = json.loads(path.read_text())
    assert payload["provenance"]["gpu_model"] == "NVIDIA A100-SXM4-40GB"
    assert payload["measurements"]["output_tokens_per_s"] == 1234.5


def test_incomplete_provenance_is_refused(tmp_path) -> None:
    """A number with no traceable hardware is not a measurement."""
    with pytest.raises(IncompleteProvenanceError) as exc:
        write_artifact(tmp_path, _provenance(gpu_model=None), {"x": 1})
    assert "gpu_model" in str(exc.value)


def test_missing_model_revision_is_refused(tmp_path) -> None:
    with pytest.raises(IncompleteProvenanceError):
        write_artifact(tmp_path, _provenance(model_revision=None), {"x": 1})


def test_artifacts_are_append_only(tmp_path) -> None:
    prov = _provenance()
    write_artifact(tmp_path, prov, {"x": 1})
    with pytest.raises(FileExistsError):
        write_artifact(tmp_path, prov, {"x": 2})


def test_artifacts_are_written_read_only(tmp_path) -> None:
    """Raw artifacts are never edited by hand; the filesystem should agree."""
    path = write_artifact(tmp_path, _provenance(), {"x": 1})
    assert not path.stat().st_mode & 0o200


def test_failed_runs_are_recorded_not_dropped(tmp_path) -> None:
    """An OOM boundary is a real result and belongs in results/raw/."""
    path = write_artifact(
        tmp_path,
        _provenance(status="oom", error="CUDA out of memory"),
        {"engine_log_path": "/tmp/x.log"},
    )
    payload = json.loads(path.read_text())
    assert payload["provenance"]["status"] == "oom"
    assert "out of memory" in payload["provenance"]["error"]


def test_warmup_runs_are_marked(tmp_path) -> None:
    path = write_artifact(tmp_path, _provenance(warmup=True, repetition_index=0), {"x": 1})
    assert json.loads(path.read_text())["provenance"]["warmup"] is True
