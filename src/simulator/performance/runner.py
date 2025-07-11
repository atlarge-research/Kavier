from __future__ import annotations

from typing import Any, List

from simulator.config import SimConfig
from simulator.performance.cache import PrefixCache
from simulator.performance.util.decode import get_decode_time_s
from simulator.performance.util.gpu_utilization import get_gpu_utilization
from simulator.performance.util.kv_utilization import get_kv_cache_utilization
from simulator.performance.util.prefill import get_prefill_time_s
from simulator.performance.util.specs.GPUSpec import GPUSpec
from simulator.performance.util.specs.LLMSpec import LLMSpec


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
    t_prefill = get_prefill_time_s(n_in_tokens, llm, gpu)
    t_decode = get_decode_time_s(n_out_tokens, llm, gpu, cfg.kv_cache)

    if in_tokens and n_in_tokens >= cfg.cache.min_len:
        hit = cache.lookup(session_id, in_tokens)
        if hit and cfg.cache.action in ("prefill", "full"):
            t_prefill = 0.0
        if hit and cfg.cache.action == "full":
            t_decode = 0.0

    total_ms = int((t_prefill + t_decode))
    task = {
        "id": str(idx),
        "submission_time": t0_ms,
        "duration": total_ms,
        "gpu_count": 1,
        "gpu_capacity": gpu.core_max_mhz * gpu.cores,
        "mem_capacity": 0,
        "cpu_count": 1,
        "cpu_capacity": 0,
        "model_name": llm.name,
        "gpu_name": gpu.name,
        "gpu_mem_capacity": int(gpu.memory_gb * 2 ** 30),
        "total_tokens": n_in_tokens + n_out_tokens,
    }

    fragments: List[dict] = []
    total_s = t_prefill + t_decode
    num_snaps = max(1, int(total_s / export_rate_s))
    t_sec = 0.0
    for snap in range(num_snaps):
        gpu_use = get_gpu_utilization(t_sec, t_prefill, t_decode)
        kv_use = get_kv_cache_utilization(
            llm, gpu, t_prefill, t_decode, t_sec, n_in_tokens, n_out_tokens, cfg.kv_cache
        )
        fragments.append(
            {
                "id": str(idx),
                "timestamp": t0_ms + int(t_sec * 1000),
                "duration": int(export_rate_s * 1000),
                "cpu_count": 1,
                "cpu_usage": 0,
                "gpu_count": 1,
                "gpu_usage": gpu_use * gpu.core_max_mhz * gpu.cores,
                "kv_gb": round(round(kv_use, 3) * gpu.memory_gb, 3),
            }
        )
        t_sec += export_rate_s

    return task, fragments, t_prefill, t_decode
