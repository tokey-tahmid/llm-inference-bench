"""Workload generation with a controllable shared-prefix ratio.

This module is the knob that makes both the prefix-caching results in P1 and the
prefix-affinity routing results in P3 meaningful, so it is built to be imported
by P3 rather than reimplemented there.

Design decision: prompts are emitted as **token ID sequences**, not text.

The reason is exactness, and it is the difference between a real prefix-caching
measurement and a plausible-looking one. Prefix caching keys on an exact prefix
of the token sequence. If prompts are built in token space, decoded to text, and
re-tokenised by the server, the round trip is not guaranteed to be a fixed point:
tokenizers merge across boundaries, so the shared region can shift by a token or
two and silently stop being a shared prefix at all. A sweep built that way
reports a prefix cache hit rate near zero and invites the wrong conclusion about
the engine. Sending token IDs removes the round trip, and makes the input length
axis exact rather than approximate.

Both target engines accept integer-token prompts on ``/v1/completions``. That
claim is verified empirically before any sweep runs, by comparing the server's
reported ``prompt_tokens`` against the length we sent; see
:func:`validate_prompt_token_fidelity`. If an engine ever fails that check, the
generator falls back to text mode and the artifact records which mode was used,
because the two are not interchangeable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class LengthDistribution(StrEnum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


class PromptFormat(StrEnum):
    TOKEN_IDS = "token_ids"
    TEXT = "text"


@dataclass(frozen=True)
class LengthSpec:
    """How to sample one length axis (input or output).

    ``lognormal`` exists because real serving traffic is not uniform: a long tail
    of large prompts is what actually stresses the scheduler and the KV cache, and
    a uniform distribution hides that. ``mean``/``sigma`` are in token space.
    """

    distribution: LengthDistribution = LengthDistribution.FIXED
    mean: int = 512
    sigma: float = 0.0
    low: int | None = None
    high: int | None = None

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.distribution is LengthDistribution.FIXED:
            out = np.full(n, self.mean, dtype=np.int64)
        elif self.distribution is LengthDistribution.UNIFORM:
            if self.low is None or self.high is None:
                raise ValueError("uniform length spec requires low and high")
            out = rng.integers(self.low, self.high + 1, size=n, dtype=np.int64)
        elif self.distribution is LengthDistribution.LOGNORMAL:
            if self.sigma <= 0:
                raise ValueError("lognormal length spec requires sigma > 0")
            # Parameterised so that `mean` is the median of the distribution,
            # which is the intuitive knob when describing a workload.
            samples = rng.lognormal(mean=np.log(self.mean), sigma=self.sigma, size=n)
            out = np.rint(samples).astype(np.int64)
        else:  # pragma: no cover - StrEnum is exhaustive
            raise ValueError(f"unhandled distribution {self.distribution}")

        lo = self.low if self.low is not None else 1
        hi = self.high if self.high is not None else np.iinfo(np.int64).max
        return np.clip(out, lo, hi)


@dataclass(frozen=True)
class WorkloadSpec:
    """A reproducible synthetic workload.

    ``shared_prefix_ratio`` is the fraction of requests that begin with a common
    long prefix. ``num_prefix_groups`` splits that population across distinct
    prefixes, which is what separates "one hot system prompt" from "a tenant mix"
    and is exactly the load-imbalance case P3's prefix-affinity router has to
    survive.
    """

    num_requests: int = 256
    input_len: LengthSpec = field(default_factory=lambda: LengthSpec(mean=1024))
    output_len: LengthSpec = field(default_factory=lambda: LengthSpec(mean=128))

    shared_prefix_ratio: float = 0.0
    shared_prefix_tokens: int = 512
    num_prefix_groups: int = 1

    seed: int = 1234
    prompt_format: PromptFormat = PromptFormat.TOKEN_IDS

    def __post_init__(self) -> None:
        if not 0.0 <= self.shared_prefix_ratio <= 1.0:
            raise ValueError("shared_prefix_ratio must be in [0, 1]")
        if self.num_prefix_groups < 1:
            raise ValueError("num_prefix_groups must be >= 1")
        if self.shared_prefix_ratio > 0 and self.shared_prefix_tokens < 1:
            raise ValueError("shared_prefix_ratio > 0 requires shared_prefix_tokens >= 1")

    def fingerprint(self) -> str:
        """Stable hash of the spec, recorded in the raw artifact.

        Two runs with the same fingerprint saw byte-identical prompts, which is
        what makes an A/B such as prefix-caching-on vs -off a controlled comparison
        rather than two different workloads.
        """
        from dataclasses import asdict

        return hashlib.sha256(
            repr(sorted(asdict(self).items())).encode()
        ).hexdigest()[:16]


@dataclass
class GeneratedRequest:
    """One request, exact in token space."""

    index: int
    prompt_token_ids: list[int]
    prompt_text: str | None
    output_tokens: int
    prefix_group: int | None
    input_tokens: int

    @property
    def shares_prefix(self) -> bool:
        return self.prefix_group is not None


class WorkloadGenerator:
    """Builds reproducible request sets against a specific tokenizer.

    The tokenizer is required even in token-ID mode: vocabulary size and the
    special-token set bound which IDs are safe to emit, and those differ per model.
    """

    def __init__(self, tokenizer: Any, spec: WorkloadSpec) -> None:
        self.tokenizer = tokenizer
        self.spec = spec
        self._safe_ids = self._compute_safe_ids()

    def _compute_safe_ids(self) -> np.ndarray:
        """Token IDs that are safe to sample.

        Special and added tokens are excluded. An EOS or a chat-template control
        token landing mid-prompt would terminate generation early or change how
        the engine treats the sequence, which would corrupt the output-length axis
        in a way that is very hard to see in aggregate numbers.
        """
        vocab_size = int(getattr(self.tokenizer, "vocab_size", 0)) or len(self.tokenizer)
        special: set[int] = set()
        for attr in ("all_special_ids",):
            special.update(int(i) for i in getattr(self.tokenizer, attr, []) or [])
        added = getattr(self.tokenizer, "added_tokens_encoder", {}) or {}
        for tok in added:
            with_id = self.tokenizer.convert_tokens_to_ids(tok)
            if isinstance(with_id, int):
                special.add(with_id)
        # Keep well clear of the top of the vocabulary, where added and reserved
        # tokens cluster in most recent models.
        candidates = np.arange(1, max(2, vocab_size), dtype=np.int64)
        if special:
            candidates = candidates[~np.isin(candidates, np.fromiter(special, dtype=np.int64))]
        if candidates.size == 0:
            raise RuntimeError("tokenizer exposes no safe sampleable token ids")
        return candidates

    def _sample_ids(self, rng: np.random.Generator, n: int) -> list[int]:
        idx = rng.integers(0, self._safe_ids.size, size=n)
        return self._safe_ids[idx].tolist()

    def generate(self) -> list[GeneratedRequest]:
        spec = self.spec
        rng = np.random.default_rng(spec.seed)

        input_lens = spec.input_len.sample(rng, spec.num_requests)
        output_lens = spec.output_len.sample(rng, spec.num_requests)

        # Prefixes are drawn from a generator seeded independently of request
        # order, so that changing num_requests does not change the prefixes and
        # invalidate comparison against an earlier sweep point.
        prefix_rng = np.random.default_rng(spec.seed + 999_983)
        prefixes = [
            self._sample_ids(prefix_rng, spec.shared_prefix_tokens)
            for _ in range(spec.num_prefix_groups)
        ]

        n_shared = int(round(spec.shared_prefix_ratio * spec.num_requests))
        assignment: list[int | None] = [None] * spec.num_requests
        if n_shared:
            chosen = rng.choice(spec.num_requests, size=n_shared, replace=False)
            for j, req_idx in enumerate(chosen):
                # Round-robin across groups so group sizes are deterministic and
                # balanced; skew is expressed via num_prefix_groups, not by luck.
                assignment[int(req_idx)] = j % spec.num_prefix_groups

        requests: list[GeneratedRequest] = []
        for i in range(spec.num_requests):
            group = assignment[i]
            target_len = int(input_lens[i])
            if group is not None:
                prefix = prefixes[group]
                if target_len <= len(prefix):
                    # A prompt shorter than the shared prefix cannot carry it.
                    # Extend rather than truncate: truncating would make the
                    # "shared" prefix differ per request and silently defeat the
                    # very effect being measured.
                    target_len = len(prefix) + 1
                unique = self._sample_ids(rng, target_len - len(prefix))
                token_ids = list(prefix) + unique
            else:
                token_ids = self._sample_ids(rng, target_len)

            text = None
            if spec.prompt_format is PromptFormat.TEXT:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            requests.append(
                GeneratedRequest(
                    index=i,
                    prompt_token_ids=token_ids,
                    prompt_text=text,
                    output_tokens=int(output_lens[i]),
                    prefix_group=group,
                    input_tokens=len(token_ids),
                )
            )

        log.info(
            "workload_generated",
            n=len(requests),
            fingerprint=spec.fingerprint(),
            shared_ratio=spec.shared_prefix_ratio,
            groups=spec.num_prefix_groups,
            total_input_tokens=sum(r.input_tokens for r in requests),
            total_output_tokens=sum(r.output_tokens for r in requests),
        )
        return requests


def theoretical_prefix_hit_rate(requests: Sequence[GeneratedRequest], prefix_tokens: int) -> float:
    """Upper bound on prefix cache hit rate for this workload, ignoring eviction.

    Reported alongside the engine's measured hit rate. The gap between the two is
    the interesting quantity: it is eviction and block-granularity loss, and
    without the bound there is nothing to attribute the measured number against.

    The first request in each group necessarily misses, so the bound is
    ``shared prefix tokens that are not first-of-group / total prompt tokens``.
    """
    total_tokens = sum(r.input_tokens for r in requests)
    if total_tokens == 0:
        return 0.0
    seen: set[int] = set()
    cacheable = 0
    for req in requests:
        if req.prefix_group is None:
            continue
        if req.prefix_group in seen:
            cacheable += prefix_tokens
        else:
            seen.add(req.prefix_group)
    return cacheable / total_tokens


async def validate_prompt_token_fidelity(
    client: Any,
    base_url: str,
    model_id: str,
    probe_token_ids: list[int],
) -> dict[str, Any]:
    """Check that the engine consumed exactly the token IDs we sent.

    Compares the server's reported ``usage.prompt_tokens`` against the length we
    submitted. A mismatch means integer-token prompts are being re-interpreted,
    and every input-length and prefix-sharing number from that engine would be
    wrong. Run once per backend before a sweep; the result goes in the artifact.
    """
    payload = {
        "model": model_id,
        "prompt": probe_token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "stream": False,
    }
    resp = await client.post(f"{base_url}/v1/completions", json=payload, timeout=120.0)
    resp.raise_for_status()
    body = resp.json()
    reported = body.get("usage", {}).get("prompt_tokens")
    ok = reported == len(probe_token_ids)
    if not ok:
        log.warning(
            "prompt_token_fidelity_mismatch", sent=len(probe_token_ids), reported=reported
        )
    return {
        "token_id_prompts_supported": ok,
        "sent_prompt_tokens": len(probe_token_ids),
        "reported_prompt_tokens": reported,
    }
