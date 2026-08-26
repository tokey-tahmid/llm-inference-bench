"""SGLang adapter.

Mapping notes where SGLang's vocabulary differs from vLLM's, because getting
these wrong produces a comparison that looks fair and is not:

* Prefix caching is **RadixAttention**, and it is **on by default**. There is no
  ``--enable-radix-cache``; the off arm passes ``--disable-radix-cache``. Same
  trap as vLLM V1, opposite flag name.
* ``--mem-fraction-static`` is the analogue of vLLM's ``--gpu-memory-utilization``
  but it is **not the same quantity**: vLLM's fraction is the total pool the
  engine may occupy, while SGLang's is the static (weights plus KV) share, with
  the remainder left for activations. Matched numeric values do not mean matched
  memory budgets. The sweep configs set these per backend and the raw artifact
  records both, so the writeup can state the memory budget actually used rather
  than implying the flags are interchangeable.
* Tensor parallel is ``--tp-size``, and context length is ``--context-length``.
* Prometheus metrics require ``--enable-metrics``; without it ``/metrics`` 404s.

Speculative decoding support is deliberately declared conservatively here. SGLang's
speculative algorithms have moved around across releases, so
``slurm/probe_capabilities.sh`` dumps ``--help`` from the pinned image and the
declared set is reconciled against it before any sweep runs. An unsupported
combination is recorded as an ``unsupported`` artifact, never quietly skipped.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from .base import (
    BackendAdapter,
    Capability,
    EngineConfig,
    SpecDecodeMode,
    parse_prometheus,
)


class SGLangAdapter(BackendAdapter):
    name = "sglang"
    capabilities = frozenset(
        {
            Capability.PREFIX_CACHING,
            Capability.PREFIX_CACHE_METRICS,
            Capability.SPEC_DECODE_DRAFT_MODEL,
            Capability.TENSOR_PARALLEL,
            Capability.MULTI_NODE,
            Capability.PROMETHEUS_METRICS,
            Capability.IGNORE_EOS,
            # Both confirmed present in sglang 0.5.18 by capability probe 5147030,
            # which caught them as UNDERCLAIMs against the pinned image:
            #   --speculative-algorithm accepts NGRAM
            #   --kv-cache-dtype accepts fp8_e5m2 / fp8_e4m3
            Capability.SPEC_DECODE_NGRAM,
            Capability.KV_CACHE_DTYPE_FP8,
            # SGLang calls the KV paging granularity --page-size where vLLM calls
            # it --block-size. Same axis, different vocabulary, which is exactly
            # the kind of difference the adapter layer exists to absorb. Note for
            # any cross-backend block-size comparison: the two engines have
            # different defaults and different valid value sets, so matched
            # numeric values are matched granularity but not a matched baseline.
            Capability.BLOCK_SIZE,
            # Still absent, verified rather than assumed:
            #   Capability.SPEC_DECODE_METRICS
        }
    )

    # vLLM accepts a bare "fp8" as an alias for fp8_e4m3; SGLang does not and
    # requires the format to be named. Normalising here rather than in the sweep
    # config keeps the YAML backend-neutral, which is the point of the abstraction.
    _KV_DTYPE_ALIASES: ClassVar[dict[str, str]] = {"fp8": "fp8_e4m3"}

    def health_path(self) -> str:
        # /health returns 200 once the HTTP layer is up, but the model may still
        # be loading. /health_generate runs an actual one-token generation, so a
        # 2xx here means the engine can genuinely serve, which is what the
        # startup-time measurement should be gated on.
        return "/health_generate"

    def server_argv(self, cfg: EngineConfig) -> list[str]:
        argv = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cfg.model_path,
            "--served-model-name",
            cfg.model_id,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "--tp-size",
            str(cfg.tensor_parallel_size),
            "--dtype",
            cfg.dtype,
            "--mem-fraction-static",
            f"{cfg.gpu_memory_utilization}",
            "--max-running-requests",
            str(cfg.max_num_seqs),
            "--random-seed",
            str(cfg.seed),
            "--enable-metrics",
        ]
        if cfg.max_model_len is not None:
            argv += ["--context-length", str(cfg.max_model_len)]

        if cfg.block_size is not None:
            argv += ["--page-size", str(cfg.block_size)]

        if cfg.kv_cache_dtype and cfg.kv_cache_dtype != "auto":
            argv += [
                "--kv-cache-dtype",
                self._KV_DTYPE_ALIASES.get(cfg.kv_cache_dtype, cfg.kv_cache_dtype),
            ]

        # Explicit in both directions. RadixAttention defaults to on.
        if not cfg.enable_prefix_caching:
            argv += ["--disable-radix-cache"]

        spec = cfg.spec_decode
        if spec.mode is SpecDecodeMode.DRAFT_MODEL:
            if not spec.draft_model:
                raise ValueError("draft_model speculative decoding requires a draft_model path")
            argv += [
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                spec.draft_model,
                "--speculative-num-steps",
                str(spec.num_speculative_tokens),
            ]
        elif spec.mode is SpecDecodeMode.NGRAM:
            # SGLang 0.5.18 supports NGRAM, confirmed against the pinned image.
            #
            # IMPORTANT for the backend-vs-backend writeup: the n-gram knobs are
            # NOT equivalent across engines and are deliberately not forced to
            # look equivalent. vLLM parameterises a prompt-lookup window
            # (prompt_lookup_min/max); SGLang parameterises a suffix-automaton
            # trie with BFS breadth and depth limits
            # (--speculative-ngram-max-trie-depth, --speculative-ngram-*-bfs-breadth).
            # Mapping one onto the other would invent an equivalence that does
            # not exist. Only the draft-token count, which genuinely means the
            # same thing on both, is carried across; the engine defaults govern
            # the rest, and any spec-decode comparison must say so.
            argv += [
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-num-draft-tokens",
                str(spec.num_speculative_tokens),
            ]

        argv += list(cfg.extra_args)
        return argv

    def version_probe_argv(self) -> list[str]:
        return ["python3", "-c", "import sglang; print(sglang.__version__)"]

    def reset_cache_endpoint(self) -> str | None:
        # SGLang's radix tree flush. Named differently from vLLM's, which is
        # exactly why this is adapter-level rather than hardcoded in the runner.
        return "/flush_cache"

    async def engine_telemetry(
        self, client: httpx.AsyncClient, base_url: str
    ) -> dict[str, Any]:
        """Normalise SGLang telemetry into the same keys the vLLM adapter emits.

        Analysis code must not branch per backend, so the normalised names are
        identical. As in the vLLM adapter, a concept SGLang does not report is
        omitted rather than zeroed.
        """
        out: dict[str, Any] = {}

        try:
            resp = await client.get(f"{base_url}/metrics", timeout=15.0)
            resp.raise_for_status()
            raw = parse_prometheus(resp.text)
            out["raw_prometheus"] = raw
        except httpx.HTTPError as exc:
            out["telemetry_error"] = str(exc)
            raw = {}

        def total(prefix: str) -> float | None:
            vals = [v for k, v in raw.items() if k.startswith(prefix)]
            return sum(vals) if vals else None

        hit_rate = total("sglang:cache_hit_rate")
        if hit_rate is not None:
            # SGLang has reported this as a percentage in some releases. Normalise
            # to a fraction so it is directly comparable to the vLLM hit rate.
            out["prefix_cache_hit_rate"] = hit_rate / 100.0 if hit_rate > 1.0 else hit_rate

        usage = total("sglang:token_usage")
        if usage is not None:
            out["kv_cache_usage_frac"] = usage

        # SGLang also exposes a richer snapshot outside Prometheus. Capture it
        # verbatim: it carries scheduler state that the exposition format omits.
        try:
            info = await client.get(f"{base_url}/get_server_info", timeout=15.0)
            if info.status_code < 300:
                out["server_info"] = info.json()
        except (httpx.HTTPError, ValueError):
            pass

        return out

    @staticmethod
    def verify_flags(help_text: str, cfg: EngineConfig) -> list[str]:
        """Return flags this adapter would emit that the pinned image does not accept."""
        adapter = SGLangAdapter(sif_path="unused")
        argv = adapter.server_argv(cfg)
        flags = [a for a in argv if a.startswith("--")]
        return [f for f in flags if f not in help_text]
