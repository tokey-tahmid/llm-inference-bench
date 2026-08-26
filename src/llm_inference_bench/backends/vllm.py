"""vLLM adapter.

Flag names below target the vLLM V1 engine. Two V1 behaviours are easy to get
wrong and would quietly invalidate a comparison:

1. **Prefix caching defaults to ON in V1.** The prefix-caching-off arm of the
   sweep must pass ``--no-enable-prefix-caching`` explicitly. Omitting the flag
   does not mean "off", it means "on", so an on/off comparison built by omission
   would compare on against on.
2. **Speculative decoding moved to a single ``--speculative-config`` JSON blob.**
   The older ``--speculative-model`` / ``--num-speculative-tokens`` pair is gone.

Neither is guessed at run time: ``slurm/probe_capabilities.sh`` dumps ``--help``
from the pinned image and :func:`verify_flags` checks that every flag this
adapter will emit actually exists in that image before a sweep is launched.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import (
    BackendAdapter,
    Capability,
    EngineConfig,
    SpecDecodeMode,
    parse_prometheus,
)


class VLLMAdapter(BackendAdapter):
    name = "vllm"
    capabilities = frozenset(
        {
            Capability.PREFIX_CACHING,
            Capability.PREFIX_CACHE_METRICS,
            Capability.SPEC_DECODE_NGRAM,
            Capability.SPEC_DECODE_DRAFT_MODEL,
            Capability.SPEC_DECODE_METRICS,
            Capability.BLOCK_SIZE,
            Capability.KV_CACHE_DTYPE_FP8,
            Capability.TENSOR_PARALLEL,
            Capability.MULTI_NODE,
            Capability.PROMETHEUS_METRICS,
            Capability.IGNORE_EOS,
        }
    )

    def health_path(self) -> str:
        return "/health"

    def server_argv(self, cfg: EngineConfig) -> list[str]:
        argv = [
            "vllm",
            "serve",
            cfg.model_path,
            "--served-model-name",
            cfg.model_id,
            "--host",
            cfg.host,
            "--port",
            str(cfg.port),
            "--tensor-parallel-size",
            str(cfg.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(cfg.pipeline_parallel_size),
            "--dtype",
            cfg.dtype,
            "--gpu-memory-utilization",
            f"{cfg.gpu_memory_utilization}",
            "--max-num-seqs",
            str(cfg.max_num_seqs),
            "--kv-cache-dtype",
            cfg.kv_cache_dtype,
            "--seed",
            str(cfg.seed),
            # Keep startup deterministic across the sweep: no auto-download, and
            # no surprise trust_remote_code prompt on a headless node.
            "--load-format",
            "auto",
            # vLLM 0.27.1 replaced --disable-log-requests with an
            # --enable-log-requests/--no-enable-log-requests pair and flipped the
            # default to off (caught by capability probe 5147176; the old flag
            # would have failed every launch in the sweep). Passed explicitly
            # rather than relying on the default: per-request logging costs real
            # time in the serving loop, so it must be off by intent, not by luck.
            "--no-enable-log-requests",
        ]
        if cfg.max_model_len is not None:
            argv += ["--max-model-len", str(cfg.max_model_len)]
        if cfg.block_size is not None:
            argv += ["--block-size", str(cfg.block_size)]

        # Explicit in both directions. See the module docstring: V1 defaults to on.
        argv += (
            ["--enable-prefix-caching"]
            if cfg.enable_prefix_caching
            else ["--no-enable-prefix-caching"]
        )

        spec = cfg.spec_decode
        if spec.mode is SpecDecodeMode.NGRAM:
            argv += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "ngram",
                        "num_speculative_tokens": spec.num_speculative_tokens,
                        "prompt_lookup_min": spec.ngram_prompt_lookup_min,
                        "prompt_lookup_max": spec.ngram_prompt_lookup_max,
                    }
                ),
            ]
        elif spec.mode is SpecDecodeMode.DRAFT_MODEL:
            if not spec.draft_model:
                raise ValueError("draft_model speculative decoding requires a draft_model path")
            argv += [
                "--speculative-config",
                json.dumps(
                    {
                        "model": spec.draft_model,
                        "num_speculative_tokens": spec.num_speculative_tokens,
                    }
                ),
            ]

        argv += list(cfg.extra_args)
        return argv

    def version_probe_argv(self) -> list[str]:
        return ["python3", "-c", "import vllm; print(vllm.__version__)"]

    def reset_cache_endpoint(self) -> str | None:
        # Present on the vLLM OpenAI server; its availability is confirmed at
        # run time by the recorded status code rather than assumed here.
        return "/reset_prefix_cache"

    async def engine_telemetry(
        self, client: httpx.AsyncClient, base_url: str
    ) -> dict[str, Any]:
        """Normalise vLLM's Prometheus counters into harness-neutral keys.

        Keys are omitted, never zeroed, when vLLM does not report them: a missing
        prefix cache hit rate means the engine did not expose one, which is a
        different fact from a measured hit rate of zero.
        """
        try:
            resp = await client.get(f"{base_url}/metrics", timeout=15.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return {"telemetry_error": str(exc)}

        raw = parse_prometheus(resp.text)
        out: dict[str, Any] = {"raw_prometheus": raw}

        def total(prefix: str) -> float | None:
            vals = [v for k, v in raw.items() if k.startswith(prefix)]
            return sum(vals) if vals else None

        # V1 reports prefix caching as query/hit counters; V0 reported a gauge.
        # Support both rather than assuming which engine version the image pinned.
        queries = total("vllm:gpu_prefix_cache_queries_total")
        hits = total("vllm:gpu_prefix_cache_hits_total")
        if queries and queries > 0 and hits is not None:
            out["prefix_cache_hit_rate"] = hits / queries
            out["prefix_cache_queries"] = queries
            out["prefix_cache_hits"] = hits
        else:
            gauge = total("vllm:gpu_prefix_cache_hit_rate")
            if gauge is not None:
                out["prefix_cache_hit_rate"] = gauge

        drafted = total("vllm:spec_decode_num_draft_tokens_total")
        accepted = total("vllm:spec_decode_num_accepted_tokens_total")
        if drafted and drafted > 0 and accepted is not None:
            out["spec_acceptance_rate"] = accepted / drafted
            out["spec_draft_tokens"] = drafted
            out["spec_accepted_tokens"] = accepted

        cache_usage = total("vllm:gpu_cache_usage_perc")
        if cache_usage is not None:
            out["kv_cache_usage_frac"] = cache_usage
        preemptions = total("vllm:num_preemptions_total")
        if preemptions is not None:
            out["preemptions_total"] = preemptions

        return out

    @staticmethod
    def verify_flags(help_text: str, cfg: EngineConfig) -> list[str]:
        """Return flags this adapter would emit that the pinned image does not accept.

        Run before a sweep, against ``vllm serve --help`` captured from the exact
        pinned image. Catches the case where a newer or older vLLM renamed a flag,
        which would otherwise show up as a wall of identical ``launch_failed``
        artifacts halfway through a job array.
        """
        adapter = VLLMAdapter(sif_path="unused")
        argv = adapter.server_argv(cfg)
        flags = [a for a in argv if a.startswith("--")]
        return [f for f in flags if f not in help_text]
