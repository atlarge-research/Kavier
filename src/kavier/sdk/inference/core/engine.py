"""Inference simulation engine: iterate a trace, simulate each request, stream tasks/fragments, aggregate metrics."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import SimConfig
from kavier.sdk.inference.core.metrics import Metrics
from kavier.sdk.inference.core.runner import RequestInput, run_request_loop
from kavier.sdk.io.input_spec import InputSpec
from kavier.sdk.io.stream_writer import StreamingParquetWriter
from kavier.sdk.library.specs.GPUSpec import GPUSpec
from kavier.sdk.library.specs.LLMSpec import LLMSpec


def simulate(
    trace: InputSpec,
    llm: LLMSpec,
    gpu: GPUSpec,
    cfg: SimConfig,
    flush_size: int,
    tasks_writer: StreamingParquetWriter,
    frags_writer: StreamingParquetWriter,
) -> str:
    """Simulate every request in ``trace``, stream OpenDC tasks/fragments to the writers, and return the summary."""
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

    requests = (
        RequestInput(
            session_id=None if sessions is None else sessions[i],
            n_in_tokens=int(num_in[i]),
            n_out_tokens=int(num_out[i]),
            in_tokens=None if in_tokens is None else in_tokens[i],
        )
        for i in range(total)
    )
    for i, task, frags, _t_p, _t_d in run_request_loop(
        requests,
        llm=llm,
        gpu=gpu,
        cache=cache,
        cfg=cfg,
        metrics=metrics,
        t0_ms=t0_ms,
        total=total,
        progress_desc="Simulating",
    ):
        TASKS.append(task)
        FRAGS.extend(frags)
        if flush_size and (i + 1) % flush_size == 0:
            _flush()

    _flush()
    return metrics.summary(cache, total, gpu, llm, cfg)
