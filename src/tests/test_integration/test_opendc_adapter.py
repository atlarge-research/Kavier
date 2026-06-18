"""The OpenDC task coercion must preserve total_tokens, or the kavier-perf -> kavier-energy
pipeline breaks (efficiency can't normalise per token). Regression for that drop."""
import pandas as pd

from kavier_opendc.adapter import _coerce_tasks_df


def test_coerce_keeps_total_tokens():
    df = pd.DataFrame({
        "id": [0], "submission_time": [0], "duration": [100],
        "cpu_count": [1], "cpu_capacity": [1000.0], "mem_capacity": [1024],
        "gpu_count": [1], "gpu_capacity": [1.0], "total_tokens": [42],
    })
    out = _coerce_tasks_df(df)
    assert "total_tokens" in out.columns
    assert int(out["total_tokens"].iloc[0]) == 42
