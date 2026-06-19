"""Per-request simulation: compute prefill/decode latency, apply prefix-cache hits, and emit one OpenDC
task plus its time-sliced GPU-usage fragments."""

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
    """Simulate one request and return ``(task, fragments, t_prefill_s, t_decode_s)``.

    Computes prefill/decode latency in seconds (zeroed on a prefix-cache hit per the
    cache policy); the emitted task ``duration`` and fragment durations are stored in
    milliseconds (s*1000, rounded, floored at 1), and fragments tile the task at
    ``export_rate_s`` with the last fragment absorbing the residual. Also carries
    ``total_tokens`` for the downstream efficiency step.
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
    # Durations are stored in MILLISECONDS (the downstream fragments + kavier_energy
    # treat `duration` as ms). Previously this stored raw seconds -> a 1000x under-count,
    # and `int()` truncated any sub-second request to 0. Convert s->ms, round, floor at 1.
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
        # carried so the efficiency step (kavier-energy) can normalise per token
        "total_tokens": int(n_in_tokens + n_out_tokens),
    }

    fragments: List[dict] = []
    num_snaps = max(1, int(total_s / export_rate_s))
    fragment_duration_ms = max(1, int(round(export_rate_s * 1000)))
    t_sec = 0.0
    for i in range(num_snaps):
        gpu_use = get_gpu_utilization(t_sec, t_prefill, t_decode)
        # The final fragment absorbs the residual so the fragments sum EXACTLY to the
        # task duration (integer truncation of num_snaps would otherwise under-cover it).
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
