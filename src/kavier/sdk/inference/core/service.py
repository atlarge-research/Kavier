"""Performance-run service: wire CLI args to the engine and write OpenDC + spec outputs to a timestamped folder."""

import datetime
import os
import time

import numpy as np
import pandas as pd

from kavier.sdk.inference.core.config import SimConfig
from kavier.sdk.inference.core.engine import simulate
from kavier.sdk.io.input_spec import InputSpec
from kavier.sdk.io.log import log
from kavier.sdk.io.opendc.adapter import output_kavier_specs, prepare_opendc_input
from kavier.sdk.io.stream_writer import StreamingParquetWriter
from kavier.sdk.library.lookup import get_gpu, get_llm


def run_performance(args) -> str:
    """Load specs, simulate, export OpenDC + Kavier outputs to a timestamped folder, and return the summary."""
    np.random.seed(42)

    cfg = SimConfig.from_cli(args)
    trace = InputSpec(args.trace)
    llm = get_llm(args.llm)
    gpu = get_gpu(args.gpu)

    out_dir = f"{args.output_folder}/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    os.makedirs(out_dir, exist_ok=True)

    tasks_sw = StreamingParquetWriter(f"{out_dir}/tasks.parquet")
    frags_sw = StreamingParquetWriter(f"{out_dir}/fragments.parquet")

    t0 = time.time()
    log("[green]Simulation started")

    try:
        results = simulate(
            trace,
            llm,
            gpu,
            cfg,
            flush_size=args.flush_size,
            tasks_writer=tasks_sw,
            frags_writer=frags_sw,
        )
    finally:
        # Finalise the parquet footers even when the simulation fails, so no
        # writer handles dangle and any flushed rows stay readable.
        tasks_sw.close()
        frags_sw.close()

    prepare_opendc_input(
        pd.read_parquet(f"{out_dir}/tasks.parquet"),
        pd.read_parquet(f"{out_dir}/fragments.parquet"),
        out_dir,
    )
    output_kavier_specs(out_dir, results)
    log(f"[green]Finished in {time.time() - t0:,.1f}s  →  {out_dir}")
    log(results)
    return results
