"""Export Kavier task/fragment frames to OpenDC workload parquet (coerce dtypes to the
OpenDC schema, then write tasks.parquet + fragments.parquet)."""

from __future__ import annotations

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kavier_opendc.schema import FRAGMENTS_SCHEMA, TASKS_SCHEMA


def _coerce_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    # total_tokens is inference-only (for the kavier-energy per-token efficiency step);
    # keep it when present but don't require it (training tasks don't carry it).
    cols = list(TASKS_SCHEMA.names) + (["total_tokens"] if "total_tokens" in df.columns else [])
    tasks = df.loc[:, cols].copy()
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
    if "total_tokens" in tasks:
        tasks["total_tokens"] = tasks["total_tokens"].astype("int64")
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
    """Write ``df`` as an OpenDC tasks.parquet (TASKS_SCHEMA, zstd). The inference-only
    ``total_tokens`` column is appended to the schema and carried through when present."""
    coerced = _coerce_tasks_df(df)
    schema = TASKS_SCHEMA
    if "total_tokens" in coerced.columns:  # inference path: carry it for the efficiency step
        schema = schema.append(pa.field("total_tokens", pa.int64(), True))
    pq.write_table(
        pa.Table.from_pandas(coerced, schema=schema, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def write_fragments_opendc(df: pd.DataFrame, path: str) -> None:
    """Write ``df`` as an OpenDC fragments.parquet (FRAGMENTS_SCHEMA, zstd)."""
    pq.write_table(
        pa.Table.from_pandas(_coerce_fragments_df(df), schema=FRAGMENTS_SCHEMA, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=True,
    )


def prepare_opendc_input(tasks: pd.DataFrame, fragments: pd.DataFrame, dst_dir: str) -> None:
    """Write both tasks.parquet and fragments.parquet into ``dst_dir`` (created if needed),
    forming a complete OpenDC workload directory."""
    os.makedirs(dst_dir, exist_ok=True)
    write_tasks_opendc(tasks, f"{dst_dir}/tasks.parquet")
    write_fragments_opendc(fragments, f"{dst_dir}/fragments.parquet")


def output_kavier_specs(dst_dir: str, results: str) -> None:
    """Dump the Kavier sim ``results`` text alongside the workload as ``_sim_results.txt``."""
    os.makedirs(dst_dir, exist_ok=True)
    with open(f"{dst_dir}/_sim_results.txt", "w") as f:
        f.write(results)
