"""Tests for kavier.sdk.co2 emissions on small synthetic traces.

Each sub-interval bills at min(own window, next window); the last window bills at its own value.
OpenDC uses a plain left step instead.
"""

from __future__ import annotations

import datetime as dt
import threading

import pandas as pd
import pytest

from kavier.sdk.co2.engine import (
    CarbonTrace,
    EmissionResult,
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


def test_single_window_billed_at_min_of_self_and_next() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 1800.0, 3600.0)  # 1.8 kWh, in window 1
    result = compute_emissions([frag], trace)
    assert result.total_energy_kwh == pytest.approx(1.8)
    assert result.total_co2_g == pytest.approx(1.8 * 100.0)  # min(100, 200) = 100
    assert result.average_intensity == pytest.approx(100.0)


def test_single_window_next_is_lower_uses_next() -> None:
    # min(300, 120) = 120
    trace = _trace([("2025-01-01 00:00", 300.0), ("2025-01-01 00:30", 120.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 1800.0, 3600.0)  # 1.8 kWh
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(1.8 * 120.0)
    assert result.breakdown[0]["carbon_intensity"] == pytest.approx(120.0)


def test_last_window_uses_own_value_no_successor() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:30"), 1800.0, 1000.0)  # 0.5 kWh, last window
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(0.5 * 200.0)


def test_fragment_spanning_two_windows_next_higher() -> None:
    # w1=100,w2=300,w3=300: w1 piece min(100,300)=100; w2 piece min(300,300)=300.
    trace = _trace(
        [
            ("2025-01-01 00:00", 100.0),
            ("2025-01-01 00:30", 300.0),
            ("2025-01-01 01:00", 300.0),
        ]
    )
    # 15 min in w1, 15 min in w2; 0.25 kWh per half.
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    expected = 0.25 * 100.0 + 0.25 * 300.0  # min(100,300)=100 ; min(300,300)=300
    assert result.total_energy_kwh == pytest.approx(0.5)
    assert result.total_co2_g == pytest.approx(expected)  # 25 + 75 = 100 g
    assert result.average_intensity == pytest.approx(200.0)


def test_fragment_spanning_two_windows_next_lower_underestimates() -> None:
    # w1=300,w2=100,w3=100: both pieces bill at 100.
    trace = _trace(
        [
            ("2025-01-01 00:00", 300.0),
            ("2025-01-01 00:30", 100.0),
            ("2025-01-01 01:00", 100.0),
        ]
    )
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)  # 0.25 kWh each
    result = compute_emissions([frag], trace)
    expected = 0.25 * 100.0 + 0.25 * 100.0  # both pieces -> 100
    assert result.total_co2_g == pytest.approx(expected)  # 50 g
    # w1 bucket records the billed intensity 100.
    assert result.breakdown[0]["carbon_intensity"] == pytest.approx(100.0)


def test_fragment_spanning_three_windows_min_rule() -> None:
    trace = _trace(
        [
            ("2025-01-01 00:00", 100.0),
            ("2025-01-01 00:30", 200.0),
            ("2025-01-01 01:00", 400.0),
        ]
    )
    # 75 min run: 30 in w1, 30 in w2, 15 in w3 (last).
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 75 * 60.0, 2000.0)
    result = compute_emissions([frag], trace)
    kwh_per_min = 2000.0 * 60.0 / 3.6e6
    # w1 min(100,200)=100 ; w2 min(200,400)=200 ; w3 (last)=400.
    expected = 30 * kwh_per_min * 100.0 + 30 * kwh_per_min * 200.0 + 15 * kwh_per_min * 400.0
    assert result.total_co2_g == pytest.approx(expected)


def test_min_rule_never_exceeds_left_step() -> None:
    # min-rule total <= left-step total
    trace = _trace(
        [
            ("2025-01-01 00:00", 250.0),
            ("2025-01-01 00:30", 90.0),
            ("2025-01-01 01:00", 400.0),
            ("2025-01-01 01:30", 120.0),
        ]
    )
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 2 * 3600.0, 1000.0)
    result = compute_emissions([frag], trace)
    kwh_per_window = 1000.0 * 1800.0 / 3.6e6  # 0.5 kWh per 30-min window
    # min-rule: w1 min(250,90)=90 ; w2 min(90,400)=90 ; w3 min(400,120)=120 ; w4 last=120.
    min_total = kwh_per_window * (90 + 90 + 120 + 120)
    left_step_total = kwh_per_window * (250 + 90 + 400 + 120)
    assert result.total_co2_g == pytest.approx(min_total)
    assert result.total_co2_g <= left_step_total


def test_per_window_breakdown_groups_by_window() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 300.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    bd = result.breakdown
    assert len(bd) == 2
    # w1: min(100,300)=100 -> 0.25 kWh * 100 = 25 g
    assert bd[0]["window_start"] == pd.Timestamp("2025-01-01 00:00")
    assert bd[0]["carbon_intensity"] == pytest.approx(100.0)
    assert bd[0]["co2_g"] == pytest.approx(25.0)
    # w2 last -> own value 300 -> 0.25 kWh * 300 = 75 g
    assert bd[1]["window_start"] == pd.Timestamp("2025-01-01 00:30")
    assert bd[1]["carbon_intensity"] == pytest.approx(300.0)
    assert bd[1]["co2_g"] == pytest.approx(75.0)


def _compute_with_hang_guard(fragments: list[Fragment], trace: CarbonTrace) -> EmissionResult:
    """Run compute_emissions on a daemon thread: a non-terminating regression fails instead of hanging pytest."""
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["result"] = compute_emissions(fragments, trace)
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout=10.0)
    assert not worker.is_alive(), "compute_emissions did not terminate (irregular-step hang regression)"
    error = box.get("error")
    if isinstance(error, BaseException):
        raise error
    result = box["result"]
    assert isinstance(result, EmissionResult)
    return result


def test_irregular_step_gap_fragment_terminates_and_bills_gap() -> None:
    # Regression, issue #4: trace 00:00, 01:00, 03:00 infers step=1h from the first interval;
    # a fragment inside the 01:00->03:00 gap looped forever at 02:00.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 03:00"]),
            "carbon_intensity": [100.0, 200.0, 400.0],
        }
    )
    trace = CarbonTrace.from_dataframe(df)  # step inferred: 1h
    frag = Fragment(pd.Timestamp("2025-01-01 01:00"), 2 * 3600.0, 1000.0)  # exactly spans the gap
    result = _compute_with_hang_guard([frag], trace)
    assert result.total_energy_kwh == pytest.approx(2.0)
    # The gap belongs to the extended 01:00 window: min(200, 400) = 200.
    assert result.total_co2_g == pytest.approx(2.0 * 200.0)
    assert len(result.breakdown) == 1
    assert result.breakdown[0]["window_start"] == pd.Timestamp("2025-01-01 01:00")


def test_irregular_step_fragment_crossing_gap_into_last_window() -> None:
    # 02:30 -> 03:30 straddles the gap edge: 30 min in the extended 01:00 window at
    # min(200, 400) = 200, then 30 min in the last window at its own 400.
    trace = _trace(
        [("2025-01-01 00:00", 100.0), ("2025-01-01 01:00", 200.0), ("2025-01-01 03:00", 400.0)],
        step_minutes=60,
    )
    frag = Fragment(pd.Timestamp("2025-01-01 02:30"), 3600.0, 2000.0)  # 1.0 kWh per half-hour
    result = _compute_with_hang_guard([frag], trace)
    assert result.total_energy_kwh == pytest.approx(2.0)
    assert result.total_co2_g == pytest.approx(1.0 * 200.0 + 1.0 * 400.0)
    assert [b["window_start"] for b in result.breakdown] == [
        pd.Timestamp("2025-01-01 01:00"),
        pd.Timestamp("2025-01-01 03:00"),
    ]


def test_multiple_fragments_same_window_accumulate() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    f1 = Fragment(pd.Timestamp("2025-01-01 00:00"), 600.0, 3600.0)  # 0.6 kWh
    f2 = Fragment(pd.Timestamp("2025-01-01 00:10"), 600.0, 3600.0)  # 0.6 kWh
    result = compute_emissions([f1, f2], trace)
    assert result.total_energy_kwh == pytest.approx(1.2)
    assert result.total_co2_g == pytest.approx(1.2 * 100.0)
    assert len(result.breakdown) == 1  # both land in window 1


def test_fragment_before_trace_raises_with_coverage() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2024-12-31 23:00"), 60.0, 1000.0)
    with pytest.raises(ValueError) as exc:
        compute_emissions([frag], trace)
    msg = str(exc.value)
    assert "2025-01-01 00:00" in msg  # coverage start named
    assert "01:00" in msg  # coverage end named


def test_fragment_after_trace_raises_with_coverage() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:45"), 30 * 60.0, 1000.0)  # ends past last window
    with pytest.raises(ValueError) as exc:
        compute_emissions([frag], trace)
    assert "01:00" in str(exc.value)


def test_fragment_exactly_to_last_window_end_ok() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:30"), 30 * 60.0, 1000.0)
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(0.5 * 200.0)


def test_load_carbon_trace_round_trip(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"]),
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


def test_tz_aware_fragment_against_naive_trace_raises() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00", tz="UTC"), 60.0, 1000.0)
    with pytest.raises(ValueError):
        compute_emissions([frag], trace)


def test_negative_duration_raises() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), -60.0, 1000.0)  # inside coverage, but negative
    with pytest.raises(ValueError, match="duration must be >= 0"):
        compute_emissions([frag], trace)


def test_no_energy_gives_zero_totals_and_zero_average() -> None:
    # average_intensity returns 0.0 for zero energy instead of dividing 0/0.
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    result = compute_emissions([], trace)
    assert result.total_energy_kwh == 0.0
    assert result.total_co2_g == 0.0
    assert result.average_intensity == 0.0
    assert result.breakdown == []


def test_zero_duration_fragment_contributes_nothing() -> None:
    # Loop guard is `cursor < end`; with cursor == end no bucket opens. `<=` would loop or add a bucket.
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:10"), 0.0, 1000.0)
    result = compute_emissions([frag], trace)
    assert result.total_energy_kwh == 0.0
    assert result.breakdown == []


def test_total_co2_kg_is_grams_over_thousand() -> None:
    # 1.8 kWh at 100 gCO2/kWh = 180 g = 0.180 kg
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 1800.0, 3600.0)  # 1.8 kWh, min(100,200)=100
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(180.0)
    assert result.total_co2_kg == pytest.approx(0.180)
    assert result.total_co2_kg != pytest.approx(180.0)


def test_from_dataframe_single_row_without_step_raises() -> None:
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01 00:00")], "carbon_intensity": [100.0]})
    with pytest.raises(ValueError, match="infer step"):
        CarbonTrace.from_dataframe(df)


def test_from_dataframe_rejects_timezone_aware_trace() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30"]).tz_localize("UTC"),
            "carbon_intensity": [100.0, 200.0],
        }
    )
    with pytest.raises(ValueError, match="timezone-naive"):
        CarbonTrace.from_dataframe(df)


def test_load_carbon_trace_step_minutes_overrides_inferred_step(tmp_path) -> None:
    # Rows are 30 min apart; step_minutes=60 gives coverage_end = 01:00 + 60 min = 02:00.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"]),
            "carbon_intensity": [10.0, 20.0, 30.0],
        }
    )
    p = tmp_path / "ci.parquet"
    df.to_parquet(p)
    trace = load_carbon_trace(str(p), step_minutes=60)
    assert trace.step == dt.timedelta(minutes=60)
    assert trace.coverage_end == pd.Timestamp("2025-01-01 02:00")
