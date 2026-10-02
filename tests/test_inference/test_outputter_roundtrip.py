"""Tests for the OpenDC parquet exporter (kavier.sdk.io.opendc.adapter).

Covers the ms-epoch -> timestamp and ms -> timedelta conversions, the subset to schema columns,
the optional total_tokens column, and the OpenDC column types.
"""

from datetime import datetime, timedelta
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kavier.sdk.io.opendc.adapter import (
    output_kavier_specs,
    prepare_opendc_input,
    write_fragments_opendc,
    write_tasks_opendc,
)


def _tasks_df(**overrides):
    """Return one tasks row with a distinct value in every TASKS_SCHEMA column."""
    row = {
        "id": 7,
        "submission_time": 0,  # ms since epoch
        "duration": 1234,
        "cpu_count": 3,
        "cpu_capacity": 1500.0,
        "mem_capacity": 2048,
        "gpu_count": 2,
        "gpu_capacity": 4.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _fragments_df(**overrides):
    row = {
        "id": 9,
        "duration": 1500,  # ms
        "cpu_count": 3,
        "cpu_usage": 0.25,
        "gpu_count": 2,
        "gpu_usage": 0.75,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_tasks_submission_time_ms_epoch_to_utc_timestamp():
    # 86_400_000 ms = one day -> 1970-01-02 00:00:00. The schema is tz-naive timestamp[ms].
    df = _tasks_df(submission_time=86_400_000)
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(df, path)
        ts = pq.read_table(path).column("submission_time").to_pylist()
    # unit="s" would give a year near 4700.
    assert ts == [datetime(1970, 1, 2, 0, 0)]


def test_tasks_scalar_values_roundtrip_and_extra_columns_dropped():
    # Distinct values catch a swapped column; extra_col checks the subset to schema columns.
    df = _tasks_df(extra_col=999)
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(df, path)
        table = pq.read_table(path)
    assert "extra_col" not in table.column_names
    assert table.column("id").to_pylist() == [7]
    assert table.column("duration").to_pylist() == [1234]
    assert table.column("cpu_count").to_pylist() == [3]
    assert table.column("cpu_capacity").to_pylist() == [1500.0]
    assert table.column("mem_capacity").to_pylist() == [2048]
    assert table.column("gpu_count").to_pylist() == [2]
    assert table.column("gpu_capacity").to_pylist() == [4.0]


def test_tasks_written_dtypes_match_opendc_contract():
    # Physical types required by the OpenDC workload reader, listed apart from schema.py.
    # id enters as int64 and must be narrowed to int32.
    df = _tasks_df(id=7)
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(df, path)
        schema = pq.read_table(path).schema
    assert schema.field("id").type == pa.int32()
    assert schema.field("submission_time").type == pa.timestamp("ms")
    assert schema.field("duration").type == pa.int64()
    assert schema.field("cpu_count").type == pa.int32()
    assert schema.field("cpu_capacity").type == pa.float64()
    assert schema.field("mem_capacity").type == pa.int64()
    assert schema.field("gpu_count").type == pa.int32()
    assert schema.field("gpu_capacity").type == pa.float64()


def test_tasks_total_tokens_absent_by_default():
    # total_tokens is inference-only; a training frame has no such column.
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(_tasks_df(), path)
        cols = pq.read_table(path).column_names
    assert "total_tokens" not in cols


def test_tasks_total_tokens_preserved_when_present():
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(_tasks_df(total_tokens=42), path)
        table = pq.read_table(path)
    assert table.column("total_tokens").to_pylist() == [42]
    assert table.schema.field("total_tokens").type == pa.int64()


def test_fragments_duration_ms_to_timedelta():
    # 1500 ms read back as a duration[ms] timedelta.
    df = _fragments_df(duration=1500)
    with TemporaryDirectory() as td:
        path = f"{td}/fragments.parquet"
        write_fragments_opendc(df, path)
        dur = pq.read_table(path).column("duration").to_pylist()
    # unit="s" would give 1500 s.
    assert dur == [timedelta(milliseconds=1500)]


def test_fragments_values_roundtrip_and_extra_columns_dropped():
    df = _fragments_df(extra="drop me")
    with TemporaryDirectory() as td:
        path = f"{td}/fragments.parquet"
        write_fragments_opendc(df, path)
        table = pq.read_table(path)
    assert "extra" not in table.column_names
    assert table.column("id").to_pylist() == [9]
    assert table.column("cpu_usage").to_pylist() == [0.25]
    assert table.column("gpu_usage").to_pylist() == [0.75]


def test_fragments_written_dtypes_match_opendc_contract():
    # OpenDC fragments: int32 id, duration[ms], float64 usage columns.
    with TemporaryDirectory() as td:
        path = f"{td}/fragments.parquet"
        write_fragments_opendc(_fragments_df(), path)
        schema = pq.read_table(path).schema
    assert schema.field("id").type == pa.int32()
    assert schema.field("duration").type == pa.duration("ms")
    assert schema.field("cpu_count").type == pa.int32()
    assert schema.field("cpu_usage").type == pa.float64()
    assert schema.field("gpu_count").type == pa.int32()
    assert schema.field("gpu_usage").type == pa.float64()


def test_prepare_opendc_input_writes_both_workload_files():
    # A workload is tasks.parquet + fragments.parquet in a newly created directory.
    tasks = pd.concat([_tasks_df(id=1), _tasks_df(id=2)], ignore_index=True)
    frags = _fragments_df()
    with TemporaryDirectory() as td:
        dst = f"{td}/nested/workload"  # does not exist yet
        prepare_opendc_input(tasks, frags, dst)
        t = pq.read_table(f"{dst}/tasks.parquet")
        f = pq.read_table(f"{dst}/fragments.parquet")
    assert t.num_rows == 2
    assert t.column("id").to_pylist() == [1, 2]
    assert f.num_rows == 1


def test_output_kavier_specs_roundtrip():
    # The text is written unchanged to _sim_results.txt.
    payload = "latency=1.5ms\nthroughput=42tok/s\n"
    with TemporaryDirectory() as td:
        dst = f"{td}/out"
        output_kavier_specs(dst, payload)
        with open(f"{dst}/_sim_results.txt") as fh:
            written = fh.read()
    assert written == payload
