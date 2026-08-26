"""Backend registry.

Adding a third engine is: write the adapter, register it here. Nothing else in
the harness may import a concrete adapter by name.
"""

from __future__ import annotations

from pathlib import Path

from .base import (
    BackendAdapter,
    BackendLaunchError,
    Capability,
    EngineConfig,
    ServerHandle,
    SpecDecodeConfig,
    SpecDecodeMode,
)
from .sglang import SGLangAdapter
from .vllm import VLLMAdapter

_REGISTRY: dict[str, type[BackendAdapter]] = {
    VLLMAdapter.name: VLLMAdapter,
    SGLangAdapter.name: SGLangAdapter,
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_adapter(name: str, sif_path: Path, image_digest: str | None = None) -> BackendAdapter:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown backend {name!r}; available: {', '.join(available_backends())}"
        ) from None
    return cls(sif_path=sif_path, image_digest=image_digest)


__all__ = [
    "BackendAdapter",
    "BackendLaunchError",
    "Capability",
    "EngineConfig",
    "SGLangAdapter",
    "ServerHandle",
    "SpecDecodeConfig",
    "SpecDecodeMode",
    "VLLMAdapter",
    "available_backends",
    "get_adapter",
]
