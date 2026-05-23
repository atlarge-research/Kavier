from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow.parquet as pq

from opendc.adapter import write_fragments_opendc, write_tasks_opendc


def test_tasks_parquet_roundtrip():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "submission_time": 0,
                "duration": 1000,
                "cpu_count": 1,
                "cpu_capacity": 1000.0,
                "mem_capacity": 1024,
                "gpu_count": 1,
                "gpu_capacity": 1.0,
            }
        ]
    )
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(df, path)
        table = pq.read_table(path)
        assert table.column_names == list(df.columns)
        assert table.num_rows == 1
        assert table.column("id").to_pylist() == [1]
        assert table.column("duration").to_pylist() == [1000]


def test_fragments_parquet_roundtrip():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "duration": 100,
                "cpu_count": 1,
                "cpu_usage": 0.0,
                "gpu_count": 1,
                "gpu_usage": 1.0,
            }
        ]
    )
    with TemporaryDirectory() as td:
        path = f"{td}/fragments.parquet"
        write_fragments_opendc(df, path)
        table = pq.read_table(path)
        assert table.column_names == list(df.columns)
        assert table.num_rows == 1
        assert table.column("id").to_pylist() == [1]
