"""Per-request simulation: prefill and decode latency with prefix-cache hits -> one OpenDC task and its fragments."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List

from tqdm.auto import tqdm

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheAction, SimConfig
from kavier.sdk.inference.core.metrics import Metrics
from kavier.sdk.inference.stages.decode import get_decode_time_s
from kavier.sdk.inference.stages.gpu_usage import get_gpu_utilization
from kavier.sdk.inference.stages.prefill import get_prefill_time_s
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec
from kavier.sdk.units import MS_PER_SECOND

# Submission time of exported tasks [ms since the Unix epoch]: 2026-01-01 00:00 UTC, the start the facade
# carbon path also uses. A fixed origin makes identical runs write identical OpenDC workloads.
TASK_ORIGIN_MS = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * MS_PER_SECOND)


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

    Latencies are in s; task and fragment durations in ms.
    """
    t_prefill = get_prefill_time_s(n_in_tokens, llm, gpu)
    t_decode = get_decode_time_s(n_out_tokens, llm, gpu, cfg.kv_cache)

    if in_tokens and n_in_tokens >= cfg.cache.min_len:
        hit = cache.lookup(session_id, in_tokens)
        if hit and cfg.cache.action in (CacheAction.PREFILL, CacheAction.FULL):
            t_prefill = 0.0
        if hit and cfg.cache.action == CacheAction.FULL:
            t_decode = 0.0

    total_s = t_prefill + t_decode
    # ms, as fragments and kavier.sdk.energy expect; floored at 1 so no task has zero duration.
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
    # Count fragments in whole ms, the unit of the durations they must sum to.
    fragment_duration_ms = max(1, int(round(export_rate_s * MS_PER_SECOND)))
    num_snaps = max(1, total_ms // fragment_duration_ms)
    t_sec = 0.0
    for i in range(num_snaps):
        gpu_use = get_gpu_utilization(t_sec, t_prefill, t_decode)
        # The last fragment takes the remainder so fragments sum to the task duration.
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
        t_sec += fragment_duration_ms / MS_PER_SECOND

    return task, fragments, t_prefill, t_decode


@dataclass(frozen=True)
class RequestInput:
    """Per-request inputs to the simulation loop: session id, token counts, optional prompt tokens."""

    session_id: Any
    n_in_tokens: int
    n_out_tokens: int
    in_tokens: list[int] | None


def run_request_loop(
    requests: Iterable[RequestInput],
    *,
    llm: LLMSpec,
    gpu: GPUSpec,
    cache: PrefixCache,
    cfg: SimConfig,
    metrics: Metrics,
    t0_ms: int,
    total: int | None = None,
    progress_desc: str | None = None,
) -> Iterator[tuple[int, dict, list[dict], float, float]]:
    """Drive ``simulate_one`` over ``requests``, accumulating into ``metrics`` and yielding each result.

    Yields ``(idx, task, fragments, t_prefill_s, t_decode_s)``. Used by ``core.engine.simulate``, which
    streams to parquet, and ``facade.run_inference``, which collects in memory.
    """
    seq: Iterable[RequestInput] = requests
    if progress_desc is not None:
        seq = tqdm(requests, total=total, desc=progress_desc, unit="req")
    for i, req in enumerate(seq):
        task, fragments, t_p, t_d = simulate_one(
            idx=i,
            session_id=req.session_id,
            n_in_tokens=req.n_in_tokens,
            n_out_tokens=req.n_out_tokens,
            in_tokens=req.in_tokens,
            llm=llm,
            gpu=gpu,
            cache=cache,
            cfg=cfg,
            export_rate_s=cfg.export_rate,
            t0_ms=t0_ms,
        )
        metrics.add(t_p, t_d, (t_p + t_d) * MS_PER_SECOND)
        yield i, task, fragments, t_p, t_d
