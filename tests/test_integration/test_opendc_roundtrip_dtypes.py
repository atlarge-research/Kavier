"""OpenDC export round-trip schema / DTYPES / VALUES for ``kavier.sdk.io.opendc.adapter``.

The oracle for every dtype assertion is EXTERNAL: OpenDC's Java workload reader is dtype-strict and
demands exactly these Arrow types on disk (submission_time = ms timestamp, fragment durations =
duration[ms], task duration/mem = int64, counts = int32). The adapter's job is to coerce arbitrary
pandas dtypes to that contract; these tests pin the contract, not the adapter's own choices. Value
oracles are the literal inputs fed in — a faithful coercion must return them unchanged.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from kavier.sdk.io.opendc.adapter import (
    prepare_opendc_input,
    write_fragments_opendc,
    write_tasks_opendc,
)


def _raw_tasks(with_tokens: bool = False, extra: bool = False) -> pd.DataFrame:
    # Deliberately hand floats where the schema wants ints (mem_capacity, id) so a dropped
    # ``.astype`` shows up as the wrong on-disk type rather than silently passing through.
    row = {
        "id": 5.0,
        "submission_time": 1234,
        "duration": 100000.0,
        "cpu_count": 2.0,
        "cpu_capacity": 1000.0,
        "mem_capacity": 4096.0,
        "gpu_count": 2,
        "gpu_capacity": 1410.0,
    }
    if with_tokens:
        row["total_tokens"] = 4242
    if extra:
        row["scheduling_class"] = 99  # not an OpenDC tasks column -> must be dropped
    return pd.DataFrame([row])


def _raw_fragments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 5, "duration": 30000, "cpu_count": 2, "cpu_usage": 1500.0, "gpu_count": 2, "gpu_usage": 800.0},
            {"id": 5, "duration": 70000, "cpu_count": 2, "cpu_usage": 1500.0, "gpu_count": 2, "gpu_usage": 800.0},
        ]
    )


def test_tasks_coerced_to_opendc_required_dtypes(tmp_path) -> None:
    # Oracle: the OpenDC reader's required column types (independent of the adapter's astype calls).
    path = str(tmp_path / "tasks.parquet")
    write_tasks_opendc(_raw_tasks(), path)
    schema = pq.read_table(path).schema

    expected = {
        "id": pa.int32(),
        "duration": pa.int64(),
        "cpu_count": pa.int32(),
        "cpu_capacity": pa.float64(),
        "mem_capacity": pa.int64(),
        "gpu_count": pa.int32(),
        "gpu_capacity": pa.float64(),
    }
    for name, typ in expected.items():
        assert schema.field(name).type == typ, name
    # submission_time is a millisecond timestamp, not an int/second timestamp.
    assert schema.field("submission_time").type == pa.timestamp("ms")


def test_tasks_numeric_values_round_trip(tmp_path) -> None:
    # Oracle: the literal inputs; coercion must not rescale or reorder values.
    path = str(tmp_path / "tasks.parquet")
    write_tasks_opendc(_raw_tasks(), path)
    tbl = pq.read_table(path)
    assert tbl.column("id").to_pylist() == [5]
    assert tbl.column("duration").to_pylist() == [100000]
    assert tbl.column("mem_capacity").to_pylist() == [4096]
    assert tbl.column("cpu_count").to_pylist() == [2]
    assert tbl.column("cpu_capacity").to_pylist() == [1000.0]
    assert tbl.column("gpu_count").to_pylist() == [2]
    assert tbl.column("gpu_capacity").to_pylist() == [1410.0]


def test_tasks_submission_time_is_ms_since_epoch(tmp_path) -> None:
    # Oracle: 1234 is interpreted as MILLISECONDS since epoch -> the instant 1234 ms after 1970-01-01.
    # If the adapter used unit="s" this would decode to 1_234_000 ms; unit="us"/"ns" would give <1234.
    path = str(tmp_path / "tasks.parquet")
    write_tasks_opendc(_raw_tasks(), path)
    tbl = pq.read_table(path)
    ts = pd.Timestamp(tbl.column("submission_time").to_pylist()[0])
    epoch_ms = ts.value // 1_000_000  # pandas stores ns; ns -> ms
    assert epoch_ms == 1234


def test_tasks_drop_columns_absent_from_schema(tmp_path) -> None:
    # Oracle: _coerce_tasks_df selects df.loc[:, schema cols] -> an unknown column is dropped.
    # A regression to df.copy() (carry all columns) would leak "scheduling_class" onto disk.
    path = str(tmp_path / "tasks.parquet")
    write_tasks_opendc(_raw_tasks(extra=True), path)
    names = pq.read_table(path).column_names
    assert "scheduling_class" not in names


def test_total_tokens_carried_only_when_present(tmp_path) -> None:
    # Oracle: total_tokens is inference-only; the schema gains it iff the input has it.
    with_path = str(tmp_path / "with.parquet")
    without_path = str(tmp_path / "without.parquet")
    write_tasks_opendc(_raw_tasks(with_tokens=True), with_path)
    write_tasks_opendc(_raw_tasks(with_tokens=False), without_path)

    with_tbl = pq.read_table(with_path)
    without_tbl = pq.read_table(without_path)
    assert with_tbl.column("total_tokens").to_pylist() == [4242]
    assert with_tbl.schema.field("total_tokens").type == pa.int64()
    # Training tasks (no token count) must NOT gain a spurious column.
    assert "total_tokens" not in without_tbl.column_names


def test_fragments_coerced_to_opendc_required_dtypes(tmp_path) -> None:
    # Oracle: OpenDC fragment schema types; duration is a duration[ms], counts int32, usages float64.
    path = str(tmp_path / "fragments.parquet")
    write_fragments_opendc(_raw_fragments(), path)
    schema = pq.read_table(path).schema
    expected = {
        "id": pa.int32(),
        "duration": pa.duration("ms"),
        "cpu_count": pa.int32(),
        "cpu_usage": pa.float64(),
        "gpu_count": pa.int32(),
        "gpu_usage": pa.float64(),
    }
    for name, typ in expected.items():
        assert schema.field(name).type == typ, name


def test_fragments_duration_values_are_milliseconds(tmp_path) -> None:
    # Oracle: input 30000/70000 are MILLISECOND durations -> 30000 ms / 70000 ms timedeltas.
    # unit="s" would blow these up 1000x; unit="us" would shrink them.
    path = str(tmp_path / "fragments.parquet")
    write_fragments_opendc(_raw_fragments(), path)
    durs = pq.read_table(path).column("duration").to_pylist()
    ms = [d.total_seconds() * 1000 for d in durs]
    assert ms == [30000.0, 70000.0]


def test_prepare_writes_both_named_files_into_new_dir(tmp_path) -> None:
    # prepare_opendc_input must os.makedirs a missing dir and emit the two OpenDC files,
    # routing tasks vs fragments to the right names (not swapped).
    dst = tmp_path / "nested" / "workload"  # does not exist yet
    prepare_opendc_input(_raw_tasks(), _raw_fragments(), str(dst))

    tasks = pq.read_table(dst / "tasks.parquet")
    frags = pq.read_table(dst / "fragments.parquet")
    # 1 task row, 2 fragment rows -> confirms the frames were not swapped between the two files.
    assert tasks.num_rows == 1
    assert frags.num_rows == 2
    # Content marker: only the tasks file carries submission_time; only fragments carry cpu_usage.
    assert "submission_time" in tasks.column_names
    assert "cpu_usage" in frags.column_names
