"""Performance-run service: run the engine from CLI args and write OpenDC and spec outputs to a timestamped folder."""

import datetime
import os
import time

import numpy as np

from kavier.sdk.inference.core.config import SimConfig
from kavier.sdk.inference.core.engine import simulate
from kavier.sdk.io.input_spec import InputSpec
from kavier.sdk.io.log import log
from kavier.sdk.io.opendc.adapter import output_kavier_specs, prepare_opendc_input
from kavier.sdk.io.parquet import read_parquet
from kavier.sdk.io.stream_writer import StreamingParquetWriter
from kavier.sdk.library.lookup import get_gpu, get_llm


def _new_run_dir(root: str | os.PathLike[str]) -> str:
    """Create and return ``root/<timestamp>``, adding a numeric suffix (``_1``, ``_2``) if that folder exists."""
    base = f"{root}/{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}"
    out_dir, n = base, 0
    while True:
        try:
            os.makedirs(out_dir)
            return out_dir
        except FileExistsError:
            n += 1
            out_dir = f"{base}_{n}"


def run_performance(args) -> str:
    """Simulate the trace, write OpenDC and Kavier outputs to a timestamped folder, and return the summary."""
    np.random.seed(42)

    cfg = SimConfig.from_cli(args)
    trace = InputSpec(args.trace)
    llm = get_llm(args.llm)
    gpu = get_gpu(args.gpu)

    out_dir = _new_run_dir(args.output_folder)

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
        # Close the writers on failure too, so the parquet footers are written and flushed rows stay readable.
        tasks_sw.close()
        frags_sw.close()

    prepare_opendc_input(
        read_parquet(f"{out_dir}/tasks.parquet"),
        read_parquet(f"{out_dir}/fragments.parquet"),
        out_dir,
    )
    output_kavier_specs(out_dir, results)
    log(f"[green]Finished in {time.time() - t0:,.1f}s, output in {out_dir}")
    log(results)
    return results
