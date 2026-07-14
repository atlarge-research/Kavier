"""Per-request simulation: prefill/decode latency + prefix-cache hits -> one OpenDC task and its GPU-usage fragments."""

from __future__ import annotations

from typing import Any, List

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheAction, SimConfig
from kavier.sdk.inference.stages.decode import get_decode_time_s
from kavier.sdk.inference.stages.gpu_usage import get_gpu_utilization
from kavier.sdk.inference.stages.prefill import get_prefill_time_s
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec
from kavier.sdk.units import MS_PER_SECOND


def simulate_one(
    idx: int,
    session_id: Any,
    n_in_tokens: int,
    n_out_tokens: int,
    in_tokens: list[int] | None,
    llm: LLMSpec,
    gpu: GPUSpec,
    cache: PrefixCache,
    cfg: SimConfig,
    export_rate_s: float,
    t0_ms: int,
) -> tuple[dict, list[dict], float, float]:
    """One request -> ``(task, fragments, t_prefill_s, t_decode_s)``; latencies in s, emitted durations in ms."""
    t_prefill = get_prefill_time_s(n_in_tokens, llm, gpu)
    t_decode = get_decode_time_s(n_out_tokens, llm, gpu, cfg.kv_cache)

    if in_tokens and n_in_tokens >= cfg.cache.min_len:
        hit = cache.lookup(session_id, in_tokens)
        if hit and cfg.cache.action in (CacheAction.PREFILL, CacheAction.FULL):
            t_prefill = 0.0
        if hit and cfg.cache.action == CacheAction.FULL:
            t_decode = 0.0

    total_s = t_prefill + t_decode
    # Store duration in ms (fragments and kavier.sdk.energy expect ms); floor at 1 so a request that
    # rounds below 1ms still gets a non-zero duration.
    total_ms = max(1, int(round(total_s * MS_PER_SECOND)))
    gpu_capacity = float(gpu.core_max_mhz * gpu.cores)
    task = {
        "id": int(idx),
        "submission_time": t0_ms,
        "duration": total_ms,
        "cpu_count": 1,
        "cpu_capacity": 1000.0,
        "mem_capacity": int(gpu.memory_gb * 1024),
        "gpu_count": 1,
        "gpu_capacity": gpu_capacity,
        "total_tokens": int(n_in_tokens + n_out_tokens),  # for the kavier energy per-token step
    }

    fragments: List[dict] = []
    num_snaps = max(1, int(total_s / export_rate_s))
    fragment_duration_ms = max(1, int(round(export_rate_s * MS_PER_SECOND)))
    t_sec = 0.0
    for i in range(num_snaps):
        gpu_use = get_gpu_utilization(t_sec, t_prefill, t_decode)
        # Final fragment absorbs the residual so fragments sum EXACTLY to the task duration.
        if i == num_snaps - 1:
            duration_ms = max(1, total_ms - i * fragment_duration_ms)
        else:
            duration_ms = fragment_duration_ms
        fragments.append(
            {
                "id": int(idx),
                "duration": duration_ms,
                "cpu_count": 1,
                "cpu_usage": 0.0,
                "gpu_count": 1,
                "gpu_usage": gpu_use * gpu_capacity,
            }
        )
        t_sec += export_rate_s

    return task, fragments, t_prefill, t_decode
