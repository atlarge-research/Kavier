"""Tests for kavier_co2 carbon-emission joins.

The carbon trace is a step function: each row covers a window
``[timestamp, timestamp + step)`` and supplies a constant carbon intensity
(gCO2/kWh) for that window. A power fragment ``(start_time, duration_s,
power_w)`` is split at window boundaries; each sub-interval's energy is
multiplied by its window's intensity. These tests use tiny synthetic traces
with hand-computed answers.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from kavier_co2.emissions import (
    CarbonTrace,
    Fragment,
    compute_emissions,
    load_carbon_trace,
)


def _trace(rows: list[tuple[str, float]], step_minutes: int = 30) -> CarbonTrace:
    df = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(t) for t, _ in rows],
            "carbon_intensity": [ci for _, ci in rows],
        }
    )
    return CarbonTrace.from_dataframe(df, step=dt.timedelta(minutes=step_minutes))


# --------------------------------------------------------------------------- #
#  Single-window fragments: energy_kWh * intensity                            #
# --------------------------------------------------------------------------- #
def test_single_window_known_answer() -> None:
    """One fragment fully inside one window: g = kWh * intensity."""
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    # 3600 W for 1800 s = 1.8 kWh, inside the first (100 gCO2/kWh) window.
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 1800.0, 3600.0)
    result = compute_emissions([frag], trace)
    assert result.total_energy_kwh == pytest.approx(1.8)
    assert result.total_co2_g == pytest.approx(180.0)  # 1.8 * 100
    assert result.average_intensity == pytest.approx(100.0)


def test_fragment_uses_covering_window_not_first_row() -> None:
    """A fragment in the second window uses that window's intensity (200)."""
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:30"), 1800.0, 1000.0)  # 0.5 kWh
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(0.5 * 200.0)


# --------------------------------------------------------------------------- #
#  Window-spanning fragment: split + weight by time                           #
# --------------------------------------------------------------------------- #
def test_fragment_spanning_two_windows_is_split() -> None:
    """A fragment crossing the boundary splits energy across both intensities."""
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 300.0)])
    # Start 15 min into window 1, run 30 min: 15 min in w1, 15 min in w2.
    # 1000 W for 900 s each half => 0.25 kWh per half.
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    expected = 0.25 * 100.0 + 0.25 * 300.0  # 25 + 75 = 100 g
    assert result.total_energy_kwh == pytest.approx(0.5)
    assert result.total_co2_g == pytest.approx(expected)
    # Average intensity is energy-weighted: 100 g / 0.5 kWh = 200.
    assert result.average_intensity == pytest.approx(200.0)


def test_fragment_spanning_three_windows() -> None:
    trace = _trace(
        [
            ("2025-01-01 00:00", 100.0),
            ("2025-01-01 00:30", 200.0),
            ("2025-01-01 01:00", 400.0),
        ]
    )
    # Start at 00:00, run 75 min: 30 in w1, 30 in w2, 15 in w3.
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 75 * 60.0, 2000.0)
    result = compute_emissions([frag], trace)
    # energy per minute = 2000 W * 60 s = 120000 Ws = 120000/3.6e6 kWh
    kwh_per_min = 2000.0 * 60.0 / 3.6e6
    expected = (
        30 * kwh_per_min * 100.0
        + 30 * kwh_per_min * 200.0
        + 15 * kwh_per_min * 400.0
    )
    assert result.total_co2_g == pytest.approx(expected)


# --------------------------------------------------------------------------- #
#  Per-window breakdown                                                        #
# --------------------------------------------------------------------------- #
def test_per_window_breakdown_groups_by_window() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 300.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    bd = result.breakdown
    assert len(bd) == 2
    assert bd[0]["window_start"] == pd.Timestamp("2025-01-01 00:00")
    assert bd[0]["carbon_intensity"] == pytest.approx(100.0)
    assert bd[0]["co2_g"] == pytest.approx(25.0)
    assert bd[1]["window_start"] == pd.Timestamp("2025-01-01 00:30")
    assert bd[1]["co2_g"] == pytest.approx(75.0)


def test_multiple_fragments_same_window_accumulate() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    f1 = Fragment(pd.Timestamp("2025-01-01 00:00"), 600.0, 3600.0)  # 0.6 kWh
    f2 = Fragment(pd.Timestamp("2025-01-01 00:10"), 600.0, 3600.0)  # 0.6 kWh
    result = compute_emissions([f1, f2], trace)
    assert result.total_energy_kwh == pytest.approx(1.2)
    assert result.total_co2_g == pytest.approx(1.2 * 100.0)
    assert len(result.breakdown) == 1  # both land in window 1


# --------------------------------------------------------------------------- #
#  Out-of-range errors name the coverage                                       #
# --------------------------------------------------------------------------- #
def test_fragment_before_trace_raises_with_coverage() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2024-12-31 23:00"), 60.0, 1000.0)
    with pytest.raises(ValueError) as exc:
        compute_emissions([frag], trace)
    msg = str(exc.value)
    assert "2025-01-01 00:00" in msg  # coverage start named
    assert "01:00" in msg  # coverage end (last window end) named


def test_fragment_after_trace_raises_with_coverage() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    # Last window ends at 01:00; a fragment ending past that is out of range.
    frag = Fragment(pd.Timestamp("2025-01-01 00:45"), 30 * 60.0, 1000.0)
    with pytest.raises(ValueError) as exc:
        compute_emissions([frag], trace)
    assert "01:00" in str(exc.value)


def test_fragment_exactly_to_last_window_end_ok() -> None:
    """A fragment ending exactly at the last window's end is in range."""
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:30"), 30 * 60.0, 1000.0)
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(0.5 * 200.0)


# --------------------------------------------------------------------------- #
#  load_carbon_trace from a real parquet (round-trip)                          #
# --------------------------------------------------------------------------- #
def test_load_carbon_trace_round_trip(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"]
            ),
            "carbon_intensity": [10.0, 20.0, 30.0],
        }
    )
    p = tmp_path / "ci.parquet"
    df.to_parquet(p)
    trace = load_carbon_trace(str(p))
    assert trace.step == dt.timedelta(minutes=30)
    assert trace.coverage_start == pd.Timestamp("2025-01-01 00:00")
    assert trace.coverage_end == pd.Timestamp("2025-01-01 01:30")


def test_load_carbon_trace_missing_column(tmp_path) -> None:
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 00:00"])})
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError):
        load_carbon_trace(str(p))


# --------------------------------------------------------------------------- #
#  Timezone-naive consistency                                                  #
# --------------------------------------------------------------------------- #
def test_tz_aware_fragment_against_naive_trace_raises() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00", tz="UTC"), 60.0, 1000.0)
    with pytest.raises(ValueError):
        compute_emissions([frag], trace)
