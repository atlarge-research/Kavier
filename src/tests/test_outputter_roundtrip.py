from tempfile import TemporaryDirectory

import pandas as pd

from outputter.outputter import write_tasks_opendc


def test_parquet_roundtrip():
    df = pd.DataFrame(
        {
            "id": ["1"],
            "submission_time": [0],
            "duration": [1000],
            "cpu_count": [1],
            "cpu_capacity": [0.0],
            "mem_capacity": [0],
            "gpu_count": [1],
            "gpu_capacity": [1.0],
            "gpu_mem_capacity": [1],
            "model_name": ["x"],
            "gpu_name": ["y"],
            "total_tokens": [10],
        }
    )
    with TemporaryDirectory() as td:
        path = f"{td}/tasks.parquet"
        write_tasks_opendc(df, path)
        out = pd.read_parquet(path)
        pd.testing.assert_frame_equal(out, df, check_dtype=False)
