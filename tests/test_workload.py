"""Workload generator tests.

The prefix-exactness tests are the load-bearing ones. If a "shared" prefix is not
a byte-exact token prefix, the engine's prefix cache cannot hit on it, the
measured hit rate collapses toward zero, and the natural reading of that result
is "prefix caching does not help much" rather than "the benchmark is broken".
That failure is silent in every aggregate number, so it is pinned down here.
"""

from __future__ import annotations

import pytest

from llm_inference_bench.workload import (
    LengthDistribution,
    LengthSpec,
    PromptFormat,
    WorkloadGenerator,
    WorkloadSpec,
    theoretical_prefix_hit_rate,
)


class StubTokenizer:
    """Minimal stand-in. The generator only needs vocabulary bounds and decode."""

    vocab_size = 5000
    all_special_ids = [0, 1, 2, 3]
    added_tokens_encoder: dict[str, int] = {}

    def convert_tokens_to_ids(self, token: str) -> int:  # pragma: no cover - unused path
        return 0

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)

    def __len__(self) -> int:
        return self.vocab_size


@pytest.fixture
def tok() -> StubTokenizer:
    return StubTokenizer()


def test_shared_prefix_is_an_exact_token_prefix(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(
        num_requests=64,
        input_len=LengthSpec(mean=800),
        output_len=LengthSpec(mean=16),
        shared_prefix_ratio=1.0,
        shared_prefix_tokens=256,
        num_prefix_groups=1,
        seed=7,
    )
    requests = WorkloadGenerator(tok, spec).generate()

    prefixes = {tuple(r.prompt_token_ids[:256]) for r in requests}
    assert len(prefixes) == 1, "all requests in one group must share one identical prefix"

    # And the shared region must be genuinely shared, not merely equal by luck on
    # a short sample.
    reference = requests[0].prompt_token_ids[:256]
    for r in requests:
        assert r.prompt_token_ids[:256] == reference


def test_distinct_groups_have_distinct_prefixes(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(
        num_requests=60,
        input_len=LengthSpec(mean=600),
        shared_prefix_ratio=1.0,
        shared_prefix_tokens=128,
        num_prefix_groups=3,
        seed=11,
    )
    requests = WorkloadGenerator(tok, spec).generate()

    by_group: dict[int, set[tuple[int, ...]]] = {}
    for r in requests:
        assert r.prefix_group is not None
        by_group.setdefault(r.prefix_group, set()).add(tuple(r.prompt_token_ids[:128]))

    assert len(by_group) == 3
    for group, prefixes in by_group.items():
        assert len(prefixes) == 1, f"group {group} has inconsistent prefixes"

    all_prefixes = {next(iter(p)) for p in by_group.values()}
    assert len(all_prefixes) == 3, "different groups must not collide on the same prefix"


def test_shared_prefix_ratio_is_respected(tok: StubTokenizer) -> None:
    for ratio in (0.0, 0.25, 0.5, 0.9, 1.0):
        spec = WorkloadSpec(
            num_requests=200,
            input_len=LengthSpec(mean=400),
            shared_prefix_ratio=ratio,
            shared_prefix_tokens=64,
            seed=3,
        )
        requests = WorkloadGenerator(tok, spec).generate()
        shared = sum(1 for r in requests if r.shares_prefix)
        assert shared == round(ratio * 200)


def test_input_length_is_exact_in_token_space(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(
        num_requests=32, input_len=LengthSpec(mean=777), output_len=LengthSpec(mean=8), seed=5
    )
    requests = WorkloadGenerator(tok, spec).generate()
    assert all(r.input_tokens == 777 for r in requests)
    assert all(len(r.prompt_token_ids) == 777 for r in requests)


def test_prompt_shorter_than_prefix_is_extended_not_truncated(tok: StubTokenizer) -> None:
    """A truncated prefix would differ per request and defeat the measurement.

    This is the subtle one: if a sampled input length lands below the shared
    prefix length, truncating the prefix to fit produces prompts that no longer
    share a common prefix at all, silently zeroing the effect under study.
    """
    spec = WorkloadSpec(
        num_requests=40,
        input_len=LengthSpec(
            distribution=LengthDistribution.UNIFORM, mean=0, low=50, high=150
        ),
        shared_prefix_ratio=1.0,
        shared_prefix_tokens=100,
        seed=13,
    )
    requests = WorkloadGenerator(tok, spec).generate()
    reference = requests[0].prompt_token_ids[:100]
    for r in requests:
        assert r.input_tokens > 100
        assert r.prompt_token_ids[:100] == reference


def test_no_special_tokens_leak_into_prompts(tok: StubTokenizer) -> None:
    """An EOS mid-prompt would end generation early and corrupt the length axis."""
    spec = WorkloadSpec(num_requests=50, input_len=LengthSpec(mean=500), seed=17)
    requests = WorkloadGenerator(tok, spec).generate()
    special = set(tok.all_special_ids)
    for r in requests:
        assert not special.intersection(r.prompt_token_ids)


def test_generation_is_reproducible(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(num_requests=24, input_len=LengthSpec(mean=300), seed=99)
    a = WorkloadGenerator(tok, spec).generate()
    b = WorkloadGenerator(tok, spec).generate()
    assert [r.prompt_token_ids for r in a] == [r.prompt_token_ids for r in b]
    assert [r.output_tokens for r in a] == [r.output_tokens for r in b]


def test_prefixes_are_stable_when_request_count_changes(tok: StubTokenizer) -> None:
    """Changing num_requests must not change the prefixes themselves.

    Otherwise a sweep point measured at 128 requests is not comparable to the
    same configuration measured at 256, because the workloads differ in more than
    the one variable that was supposed to change.
    """
    common = {
        "input_len": LengthSpec(mean=400),
        "shared_prefix_ratio": 1.0,
        "shared_prefix_tokens": 64,
        "seed": 21,
    }
    small = WorkloadGenerator(tok, WorkloadSpec(num_requests=16, **common)).generate()
    large = WorkloadGenerator(tok, WorkloadSpec(num_requests=64, **common)).generate()
    assert small[0].prompt_token_ids[:64] == large[0].prompt_token_ids[:64]


def test_fingerprint_distinguishes_specs(tok: StubTokenizer) -> None:
    a = WorkloadSpec(num_requests=10, shared_prefix_ratio=0.0)
    b = WorkloadSpec(num_requests=10, shared_prefix_ratio=0.5)
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == WorkloadSpec(num_requests=10, shared_prefix_ratio=0.0).fingerprint()


def test_theoretical_bound_is_zero_without_sharing(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(num_requests=20, input_len=LengthSpec(mean=200), shared_prefix_ratio=0.0)
    requests = WorkloadGenerator(tok, spec).generate()
    assert theoretical_prefix_hit_rate(requests, 0) == 0.0


def test_theoretical_bound_excludes_first_of_each_group(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(
        num_requests=10,
        input_len=LengthSpec(mean=100),
        shared_prefix_ratio=1.0,
        shared_prefix_tokens=50,
        num_prefix_groups=1,
        seed=2,
    )
    requests = WorkloadGenerator(tok, spec).generate()
    bound = theoretical_prefix_hit_rate(requests, 50)
    # 9 of 10 requests can hit; the first populates the cache.
    expected = (9 * 50) / sum(r.input_tokens for r in requests)
    assert bound == pytest.approx(expected)


def test_text_format_populates_prompt_text(tok: StubTokenizer) -> None:
    spec = WorkloadSpec(
        num_requests=4,
        input_len=LengthSpec(mean=20),
        prompt_format=PromptFormat.TEXT,
        seed=1,
    )
    requests = WorkloadGenerator(tok, spec).generate()
    assert all(r.prompt_text is not None for r in requests)


def test_invalid_specs_are_rejected() -> None:
    with pytest.raises(ValueError):
        WorkloadSpec(shared_prefix_ratio=1.5)
    with pytest.raises(ValueError):
        WorkloadSpec(num_prefix_groups=0)
    with pytest.raises(ValueError):
        LengthSpec(distribution=LengthDistribution.UNIFORM).sample(
            __import__("numpy").random.default_rng(0), 5
        )
