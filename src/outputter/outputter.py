from __future__ import annotations

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from outputter.opendc_schema import FRAGMENTS_SCHEMA, TASKS_SCHEMA


def write_tasks_opendc(df: pd.DataFrame, path: str) -> None:
    pq.write_table(
        pa.Table.from_pandas(df, schema=TASKS_SCHEMA, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=["id"],
    )


def write_fragments_opendc(df: pd.DataFrame, path: str) -> None:
    pq.write_table(
        pa.Table.from_pandas(df, schema=FRAGMENTS_SCHEMA, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=["id"],
    )


def prepare_opendc_input(tasks: pd.DataFrame, fragments: pd.DataFrame, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    write_tasks_opendc(tasks, f"{dst_dir}/tasks.parquet")
    write_fragments_opendc(fragments, f"{dst_dir}/fragments.parquet")


def _spec_to_dict(spec) -> dict:
    return {k: v for k, v in spec.__dict__.items() if not k.startswith("_") and not callable(v)}


def output_kavier_specs(dst_dir: str, results: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    with open(f"{dst_dir}/_sim_results.txt", "w") as f:
        f.write(results)
