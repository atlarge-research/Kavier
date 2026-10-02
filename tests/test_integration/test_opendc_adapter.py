"""OpenDC export adapter: schema projection, dtype and unit coercion, and parquet round-trips.

Expected values do not come from the adapter: epoch arithmetic is done by hand, column sets come
from the pyarrow schemas, and round-trips read the file back with pyarrow.
"""

import os
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq

from kavier.sdk.io.opendc.adapter import (
    _coerce_fragments_df,
    _coerce_tasks_df,
    output_kavier_specs,
    prepare_opendc_input,
    write_tasks_opendc,
)
from kavier.sdk.io.opendc.schema import FRAGMENTS_SCHEMA, TASKS_SCHEMA


def _tasks_df(**overrides):
    """Return a one-row tasks frame with every required schema column, updated with ``overrides``."""
    base = {
        "id": [0],
        "submission_time": [0],
        "duration": [100],
        "cpu_count": [1],
        "cpu_capacity": [1000.0],
        "mem_capacity": [1024],
        "gpu_count": [1],
        "gpu_capacity": [1.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _fragments_df(**overrides):
    base = {
        "id": [0],
        "duration": [100],
        "cpu_count": [1],
        "cpu_usage": [1000.0],
        "gpu_count": [1],
        "gpu_usage": [1.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_coerce_keeps_total_tokens_value():
    # total_tokens is inference-only; when present its value passes through coercion unchanged.
    out = _coerce_tasks_df(_tasks_df(total_tokens=[42]))
    assert "total_tokens" in out.columns
    assert int(out["total_tokens"].iloc[0]) == 42


def test_coerce_drops_total_tokens_when_absent():
    # Training tasks have no token count; the adapter does not add the column.
    out = _coerce_tasks_df(_tasks_df())
    assert "total_tokens" not in out.columns


def test_coerce_tasks_projects_to_schema_columns_only():
    # Output columns equal the tasks schema; df.copy() in place of .loc[:, cols] would leak "junk".
    out = _coerce_tasks_df(_tasks_df(junk=["leak"]))
    assert list(out.columns) == list(TASKS_SCHEMA.names)


def test_coerce_fragments_projects_to_schema_columns_only():
    out = _coerce_fragments_df(_fragments_df(junk=["leak"]))
    assert list(out.columns) == list(FRAGMENTS_SCHEMA.names)


def test_submission_time_is_read_as_milliseconds_since_epoch():
    # 1500 ms after the Unix epoch = 1970-01-01 00:00:01.500 UTC. Read as seconds it would be 00:25:00.
    out = _coerce_tasks_df(_tasks_df(submission_time=[1500]))
    assert out["submission_time"].iloc[0] == pd.Timestamp("1970-01-01 00:00:01.500", tz="UTC")


def test_fragment_duration_is_read_as_milliseconds():
    # 1500 ms -> a 1.5 s timedelta; unit="s" would give 1500 s (25 min).
    out = _coerce_fragments_df(_fragments_df(duration=[1500]))
    assert out["duration"].iloc[0] == pd.Timedelta(seconds=1.5)


def test_write_tasks_roundtrip_appends_total_tokens_field(tmp_path):
    # With total_tokens present, the written schema is the base tasks schema plus a total_tokens field,
    # and the value reads back unchanged.
    path = tmp_path / "tasks.parquet"
    write_tasks_opendc(_tasks_df(total_tokens=[42]), str(path))
    table = pq.read_table(str(path))
    assert table.schema.names == list(TASKS_SCHEMA.names) + ["total_tokens"]
    assert table.column("total_tokens").to_pylist() == [42]


def test_write_tasks_roundtrip_without_total_tokens(tmp_path):
    # Training-style tasks: written schema is exactly the base tasks schema, no extra field.
    path = tmp_path / "tasks.parquet"
    write_tasks_opendc(_tasks_df(), str(path))
    assert pq.read_table(str(path)).schema.names == list(TASKS_SCHEMA.names)


def test_prepare_opendc_input_writes_both_workload_files(tmp_path):
    # An OpenDC workload is tasks.parquet + fragments.parquet in the destination dir, each readable
    # and matching its schema.
    dst = tmp_path / "workload"
    prepare_opendc_input(_tasks_df(), _fragments_df(), str(dst))
    assert (dst / "tasks.parquet").exists()
    assert (dst / "fragments.parquet").exists()
    assert pq.read_table(str(dst / "tasks.parquet")).schema.names == list(TASKS_SCHEMA.names)
    assert pq.read_table(str(dst / "fragments.parquet")).schema.names == list(FRAGMENTS_SCHEMA.names)


def test_output_kavier_specs_writes_verbatim_sim_results(tmp_path):
    # The sidecar file holds the results text byte for byte under a fixed name.
    payload = "prefill=1.5ms\ndecode=42ms\n"
    output_kavier_specs(str(tmp_path), payload)
    assert (tmp_path / "_sim_results.txt").read_text() == payload


def test_output_kavier_specs_writes_utf8_under_an_ascii_locale(tmp_path):
    # Windows defaults to cp1252 and a C locale to ASCII; neither can encode U+2265.
    code = (
        "import sys\n"
        "from kavier.sdk.io.opendc.adapter import output_kavier_specs\n"
        "output_kavier_specs(sys.argv[1], 'p95 \\u2265 1 ms')\n"
    )
    env = {**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert (tmp_path / "_sim_results.txt").read_text(encoding="utf-8") == "p95 \u2265 1 ms"
