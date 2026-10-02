"""``fragments_from_powersource`` and ``kavier carbon --powersource`` on OpenDC's own powerSource schema.

The fixture follows OpenDC's DfltPowerSourceExportColumns: ``timestamp`` (ms since simulation start) and
``timestamp_absolute`` (epoch ms) are plain int64, energy and power are float32. Each row's
``energy_usage`` is the energy since the previous row, so it covers the interval ending at its timestamp.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kavier.cli.carbon import main
from kavier.sdk.co2.fragments import fragments_from_powersource

SIM_START = pd.Timestamp("2025-06-01 00:00")

OPENDC_POWERSOURCE_SCHEMA = pa.schema(
    [
        pa.field("timestamp", pa.int64(), nullable=False),
        pa.field("timestamp_absolute", pa.int64(), nullable=False),
        pa.field("source_name", pa.string(), nullable=False),
        pa.field("cluster_name", pa.string(), nullable=False),
        pa.field("power_draw", pa.float32(), nullable=False),
        pa.field("energy_usage", pa.float32(), nullable=False),
        pa.field("carbon_intensity", pa.float32(), nullable=False),
        pa.field("carbon_emission", pa.float32(), nullable=False),
    ]
)


def _write_opendc_powersource(
    path: Path, rel_ms: list[int], energy_ws: list[float], sources: list[str] | None = None
) -> Path:
    n = len(rel_ms)
    start_ms = SIM_START.value // 1_000_000
    table = pa.table(
        {
            "timestamp": rel_ms,
            "timestamp_absolute": [start_ms + t for t in rel_ms],
            "source_name": sources or ["powerSource0"] * n,
            "cluster_name": ["C01"] * n,
            "power_draw": [0.0] * n,
            "energy_usage": energy_ws,
            "carbon_intensity": [0.0] * n,
            "carbon_emission": [0.0] * n,
        },
        schema=OPENDC_POWERSOURCE_SCHEMA,
    )
    pq.write_table(table, path)
    return path


def test_opendc_int64_timestamps_are_read_as_epoch_milliseconds(tmp_path: Path) -> None:
    # Three 5-minute export intervals of 300,000 Ws each, i.e. 1000 W.
    path = _write_opendc_powersource(tmp_path / "powerSource.parquet", [300_000, 600_000, 900_000], [3e5] * 3)
    frags = fragments_from_powersource(pd.read_parquet(path))

    assert [f.start_time for f in frags] == [
        pd.Timestamp("2025-06-01 00:00"),
        pd.Timestamp("2025-06-01 00:05"),
        pd.Timestamp("2025-06-01 00:10"),
    ]
    assert [f.duration_s for f in frags] == pytest.approx([300.0, 300.0, 300.0])
    assert [f.power_w for f in frags] == pytest.approx([1000.0, 1000.0, 1000.0])


def test_relative_int64_timestamp_alone_is_read_as_milliseconds(tmp_path: Path) -> None:
    path = _write_opendc_powersource(tmp_path / "powerSource.parquet", [300_000, 600_000], [3e5, 3e5])
    df = pd.read_parquet(path).drop(columns=["timestamp_absolute"])
    frags = fragments_from_powersource(df)

    assert [f.duration_s for f in frags] == pytest.approx([300.0, 300.0])
    assert [f.power_w for f in frags] == pytest.approx([1000.0, 1000.0])


def test_short_final_export_interval_keeps_its_own_width(tmp_path: Path) -> None:
    # OpenDC writes a last row when the simulation ends, here 150 s after the previous one.
    path = _write_opendc_powersource(tmp_path / "powerSource.parquet", [300_000, 600_000, 750_000], [3e5, 3e5, 1.5e5])
    frags = fragments_from_powersource(pd.read_parquet(path))

    assert [f.duration_s for f in frags] == pytest.approx([300.0, 300.0, 150.0])
    assert [f.power_w for f in frags] == pytest.approx([1000.0, 1000.0, 1000.0])
    assert frags[2].start_time == pd.Timestamp("2025-06-01 00:10")


def test_rows_from_several_power_sources_are_summed_per_timestamp(tmp_path: Path) -> None:
    path = _write_opendc_powersource(
        tmp_path / "powerSource.parquet",
        [300_000, 300_000, 600_000, 600_000],
        [1e5, 2e5, 1e5, 2e5],
        sources=["powerSource0", "powerSource1", "powerSource0", "powerSource1"],
    )
    frags = fragments_from_powersource(pd.read_parquet(path))

    assert len(frags) == 2
    assert [f.power_w for f in frags] == pytest.approx([1000.0, 1000.0])
    assert sum(f.power_w * f.duration_s for f in frags) == pytest.approx(6e5)


def test_carbon_cli_bills_opendc_powersource_in_the_window_before_each_row(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 30-minute trace windows at 100, 200, 300, ... gCO2/kWh from the simulation start.
    ts = pd.date_range(SIM_START, periods=48, freq="30min")
    trace_path = tmp_path / "carbon.parquet"
    pd.DataFrame({"timestamp": ts, "carbon_intensity": [100.0 * (i + 1) for i in range(len(ts))]}).to_parquet(
        trace_path
    )
    # 1 kWh (3.6e6 Ws) in each of the first three 30-minute windows.
    ps_path = _write_opendc_powersource(
        tmp_path / "powerSource.parquet", [1_800_000, 3_600_000, 5_400_000], [3.6e6] * 3
    )

    main(["--powersource", str(ps_path), "--carbon_trace", str(trace_path)])
    out = capsys.readouterr().out

    energy = re.search(r"Total energy:\s+([\d,.]+) kWh", out)
    co2 = re.search(r"Total CO2:\s+([\d,.]+) g", out)
    assert energy is not None and co2 is not None, out
    assert float(energy.group(1).replace(",", "")) == pytest.approx(3.0, abs=1e-4)
    # 1 kWh x (100 + 200 + 300) gCO2/kWh = 600 g.
    assert float(co2.group(1).replace(",", "")) == pytest.approx(600.0, abs=1e-2)
