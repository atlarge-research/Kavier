import datetime
import os
import time

import numpy as np
import pandas as pd

from library.gpu_library import GPU_SPEC_LIBRARY
from library.llm_library import LLM_SPEC_LIBRARY
from outputter.outputter import output_kavier_specs, prepare_opendc_input
from simulator.config import SimConfig
from simulator.log import log
from simulator.performance.simulate import simulate
from simulator.performance.util.specs.InputSpec import InputSpec
from simulator.performance.util.stream_writer import StreamingParquetWriter


def run_performance(args) -> str:
    np.random.seed(42)

    cfg = SimConfig.from_cli(args)
    trace = InputSpec(args.trace)
    llm = LLM_SPEC_LIBRARY[args.llm]
    gpu = GPU_SPEC_LIBRARY[args.gpu]

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
