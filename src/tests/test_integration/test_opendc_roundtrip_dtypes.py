"""OpenDC export round-trip: schema, DTYPES and VALUES must survive write->read.

The existing test_outputter_roundtrip.py only checks column NAMES and a couple
of cell values. OpenDC's Java reader is dtype-strict: submission_time must be a
ms timestamp, task duration an int64 (ms), fragment duration a duration[ms], the
count columns int32/Int32. These assert the coercion in kavier_opendc.adapter
actually produces those Arrow types on disk and that values round-trip exactly.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kavier_opendc.adapter import (
    prepare_opendc_input,
    write_fragments_opendc,
    write_tasks_opendc,
)


def _raw_tasks(with_tokens: bool = False) -> pd.DataFrame:
    row = {
        "id": 5, "submission_time": 1234, "duration": 100000,
        "cpu_count": 2, "cpu_capacity": 1000.0, "mem_capacity": 4096,
        "gpu_count": 2, "gpu_capacity": 1410.0,
    }
    if with_tokens:
        row["total_tokens"] = 4242
    return pd.DataFrame([row])


def _raw_fragments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 5, "duration": 30000, "cpu_count": 2, "cpu_usage": 1500.0, "gpu_count": 2, "gpu_usage": 800.0},
            {"id": 5, "duration": 70000, "cpu_count": 2, "cpu_usage": 1500.0, "gpu_count": 2, "gpu_usage": 800.0},
        ]
    )


def test_tasks_arrow_dtypes_and_timestamp_conversion(tmp_path) -> None:
    path = str(tmp_path / "tasks.parquet")
    write_tasks_opendc(_raw_tasks(), path)
    table = pq.read_table(path)
    s = table.schema

    assert s.field("id").type == pa.int32()
    # submission_time goes in as ms-since-epoch int -> must come back a ms timestamp.
    assert pa.types.is_timestamp(s.field("submission_time").type)
    assert s.field("duration").type == pa.int64()
    assert s.field("cpu_count").type == pa.int32()
    assert s.field("cpu_capacity").type == pa.float64()
    assert s.field("mem_capacity").type == pa.int64()

    # The submission_time int (1234 ms) must decode to the same instant
    # (1970-01-01T00:00:01.234). Compare the millisecond count to be tz-robust.
    ts = pd.Timestamp(table.column("submission_time").to_pylist()[0])
    epoch_ms = ts.value // 1_000_000  # ns -> ms
    assert epoch_ms == 1234
    assert table.column("duration").to_pylist() == [100000]


def test_fragments_duration_is_arrow_duration_ms(tmp_path) -> None:
    path = str(tmp_path / "fragments.parquet")
    write_fragments_opendc(_raw_fragments(), path)
    table = pq.read_table(path)
    assert table.schema.field("duration").type == pa.duration("ms")
    # Values survive as ms-resolution timedeltas.
    durs = pd.to_timedelta(table.column("duration").to_pylist())
    assert list(durs / pd.Timedelta(milliseconds=1)) == [30000.0, 70000.0]


def test_total_tokens_carried_only_when_present(tmp_path) -> None:
    with_path = str(tmp_path / "with.parquet")
    without_path = str(tmp_path / "without.parquet")
    write_tasks_opendc(_raw_tasks(with_tokens=True), with_path)
    write_tasks_opendc(_raw_tasks(with_tokens=False), without_path)

    with_tbl = pq.read_table(with_path)
    without_tbl = pq.read_table(without_path)
    assert "total_tokens" in with_tbl.column_names
    assert with_tbl.column("total_tokens").to_pylist() == [4242]
    assert with_tbl.schema.field("total_tokens").type == pa.int64()
    # Training tasks (no token count) must NOT gain a spurious column.
    assert "total_tokens" not in without_tbl.column_names


def test_prepare_opendc_input_writes_both_files_readable(tmp_path) -> None:
    prepare_opendc_input(_raw_tasks(), _raw_fragments(), str(tmp_path))
    tasks = pq.read_table(tmp_path / "tasks.parquet")
    frags = pq.read_table(tmp_path / "fragments.parquet")
    assert tasks.num_rows == 1
    assert frags.num_rows == 2
    # Fragment durations must tile the task duration (30000 + 70000 == 100000 ms).
    total_frag_ms = sum(
        pd.Timedelta(d) / pd.Timedelta(milliseconds=1) for d in frags.column("duration").to_pylist()
    )
    assert total_frag_ms == pytest.approx(tasks.column("duration").to_pylist()[0])
