"""Workload generation, shared between P1 (llm-inference-bench) and P3 (inference-router).

P3 imports this package rather than reimplementing it. If the two ever diverge,
the prefix-caching numbers from P1 stop being comparable to the prefix-affinity
routing numbers from P3, which is the whole point of the pairing.
"""

from .generator import (
    GeneratedRequest,
    LengthDistribution,
    LengthSpec,
    PromptFormat,
    WorkloadGenerator,
    WorkloadSpec,
    theoretical_prefix_hit_rate,
    validate_prompt_token_fidelity,
)

__all__ = [
    "GeneratedRequest",
    "LengthDistribution",
    "LengthSpec",
    "PromptFormat",
    "WorkloadGenerator",
    "WorkloadSpec",
    "theoretical_prefix_hit_rate",
    "validate_prompt_token_fidelity",
]
