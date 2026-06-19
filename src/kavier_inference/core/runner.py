"""Per-request simulation: prefill/decode latency + prefix-cache hits -> one OpenDC task and its GPU-usage fragments."""

from __future__ import annotations

from typing import Any, List

from kavier_inference.core.cache import PrefixCache
from kavier_inference.core.config import SimConfig
from kavier_inference.stages.decode import get_decode_time_s
from kavier_inference.stages.gpu_usage import get_gpu_utilization
from kavier_inference.stages.prefill import get_prefill_time_s
from kavier_library.specs.GPUSpec import GPUSpec
from kavier_library.specs.LLMSpec import LLMSpec


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
    """Simulate one request -> ``(task, fragments, t_prefill_s, t_decode_s)``.

    Prefill/decode latency is in seconds (zeroed on a prefix-cache hit per the cache policy);
    emitted task/fragment durations are in milliseconds (see the unit note below).
    """
    t_prefill = get_prefill_time_s(n_in_tokens, llm, gpu)
    t_decode = get_decode_time_s(n_out_tokens, llm, gpu, cfg.kv_cache)

    if in_tokens and n_in_tokens >= cfg.cache.min_len:
        hit = cache.lookup(session_id, in_tokens)
        if hit and cfg.cache.action in ("prefill", "full"):
            t_prefill = 0.0
        if hit and cfg.cache.action == "full":
            t_decode = 0.0

    total_s = t_prefill + t_decode
    # Durations are MILLISECONDS (downstream fragments + kavier_energy treat them as ms).
    # s*1000, rounded, floored at 1: raw seconds under-counted 1000x and int() truncated sub-second requests to 0.
    total_ms = max(1, int(round(total_s * 1000)))
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
        "total_tokens": int(n_in_tokens + n_out_tokens),  # for the kavier-energy per-token step
    }

    fragments: List[dict] = []
    num_snaps = max(1, int(total_s / export_rate_s))
    fragment_duration_ms = max(1, int(round(export_rate_s * 1000)))
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
