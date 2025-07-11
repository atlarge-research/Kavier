# simulator/performance/simulate.py
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from simulator.config import SimConfig
from simulator.performance.cache import PrefixCache
from simulator.performance.metrics import Metrics
from simulator.performance.runner import simulate_one
from simulator.performance.util.specs import GPUSpec, InputSpec, LLMSpec
from simulator.performance.util.stream_writer import StreamingParquetWriter


def simulate(
    trace: InputSpec,
    llm: LLMSpec,
    gpu: GPUSpec,
    cfg: SimConfig,
    flush_size: int,
    tasks_writer: StreamingParquetWriter,
    frags_writer: StreamingParquetWriter,
) -> str:
    cache = PrefixCache(cfg.cache)
    metrics = Metrics()

    num_in = trace.num_in_t.to_numpy()
    num_out = trace.num_out_t.to_numpy()
    sessions = np.asarray(trace.sessions) if trace.sessions else None
    in_tokens = trace.in_t if trace.in_t else None

    TASKS, FRAGS = [], []
    t0_ms = int(time.time_ns() / 1e6)
    total = len(num_in)

    def _flush() -> None:
        if TASKS:
            tasks_writer.write(pd.DataFrame(TASKS))
            TASKS.clear()
        if FRAGS:
            frags_writer.write(pd.DataFrame(FRAGS))
            FRAGS.clear()

    for i in tqdm(range(total), desc="Simulating", unit="req"):
        task, frags, t_p, t_d = simulate_one(
            idx=i,
            session_id=None if sessions is None else sessions[i],
            n_in_tokens=int(num_in[i]),
            n_out_tokens=int(num_out[i]),
            in_tokens=None if in_tokens is None else in_tokens[i],
            llm=llm,
            gpu=gpu,
            cache=cache,
            cfg=cfg,
            export_rate_s=cfg.export_rate,
            t0_ms=t0_ms,
        )
        TASKS.append(task)
        FRAGS.extend(frags)
        metrics.add(t_p, t_d, (t_p + t_d) * 1_000)

        if flush_size and (i + 1) % flush_size == 0:
            _flush()

    results = metrics.summary(cache, total, pd.DataFrame(FRAGS), gpu, llm, cfg)
    _flush()
    return results
