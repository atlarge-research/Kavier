import datetime
import os
import time

import numpy as np
import pandas as pd

from kavier_inference.core.config import SimConfig
from kavier_inference.core.engine import simulate
from kavier_io.input_spec import InputSpec
from kavier_io.log import log
from kavier_io.stream_writer import StreamingParquetWriter
from library.lookup import get_gpu, get_llm
from opendc.adapter import output_kavier_specs, prepare_opendc_input


def run_performance(args) -> str:
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

    results = simulate(
        trace,
        llm,
        gpu,
        cfg,
        flush_size=args.flush_size,
        tasks_writer=tasks_sw,
        frags_writer=frags_sw,
    )
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
