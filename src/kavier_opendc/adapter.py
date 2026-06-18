from __future__ import annotations

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kavier_opendc.schema import FRAGMENTS_SCHEMA, TASKS_SCHEMA


def _coerce_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    tasks = df.loc[:, TASKS_SCHEMA.names].copy()
    tasks["id"] = tasks["id"].astype("int32")
    tasks["submission_time"] = pd.to_datetime(tasks["submission_time"], unit="ms", utc=True)
    tasks["duration"] = tasks["duration"].astype("int64")
    tasks["cpu_count"] = tasks["cpu_count"].astype("int32")
    tasks["cpu_capacity"] = tasks["cpu_capacity"].astype("float64")
    tasks["mem_capacity"] = tasks["mem_capacity"].astype("int64")
    if "gpu_count" in tasks:
        tasks["gpu_count"] = tasks["gpu_count"].astype("Int32")
    if "gpu_capacity" in tasks:
        tasks["gpu_capacity"] = tasks["gpu_capacity"].astype("float64")
    return tasks


def _coerce_fragments_df(df: pd.DataFrame) -> pd.DataFrame:
    fragments = df.loc[:, FRAGMENTS_SCHEMA.names].copy()
    fragments["id"] = fragments["id"].astype("int32")
    fragments["duration"] = pd.to_timedelta(fragments["duration"], unit="ms")
    fragments["cpu_count"] = fragments["cpu_count"].astype("int32")
    fragments["cpu_usage"] = fragments["cpu_usage"].astype("float64")
    if "gpu_count" in fragments:
        fragments["gpu_count"] = fragments["gpu_count"].astype("Int32")
    if "gpu_usage" in fragments:
        fragments["gpu_usage"] = fragments["gpu_usage"].astype("float64")
    return fragments


def write_tasks_opendc(df: pd.DataFrame, path: str) -> None:
    pq.write_table(
        pa.Table.from_pandas(_coerce_tasks_df(df), schema=TASKS_SCHEMA, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def write_fragments_opendc(df: pd.DataFrame, path: str) -> None:
    pq.write_table(
        pa.Table.from_pandas(_coerce_fragments_df(df), schema=FRAGMENTS_SCHEMA, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def prepare_opendc_input(tasks: pd.DataFrame, fragments: pd.DataFrame, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    write_tasks_opendc(tasks, f"{dst_dir}/tasks.parquet")
    write_fragments_opendc(fragments, f"{dst_dir}/fragments.parquet")


def output_kavier_specs(dst_dir: str, results: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)
    with open(f"{dst_dir}/_sim_results.txt", "w") as f:
        f.write(results)
