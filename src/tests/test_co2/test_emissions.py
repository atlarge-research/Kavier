"""Tests for kavier_co2 carbon-emission joins.

The carbon trace is a step function: each row covers a window
``[timestamp, timestamp + step)`` and supplies a constant carbon intensity
(gCO2/kWh) for that window. A power fragment ``(start_time, duration_s,
power_w)`` is split at window boundaries.

Billing rule (conservative DOWN-ESTIMATION): each split sub-interval is billed
at ``min(intensity of its own window, intensity of the NEXT window)``. The
final trace window has no successor, so it bills at its own value. This
deliberately under-estimates at in-between times: a moment that sits between two
trace points takes the LOWER of the two bounding intensities. It deviates from
OpenDC, which holds the earlier (left) point's intensity until the next point
(a pure left-step, no min). These tests use tiny synthetic traces with
hand-computed answers.
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
def test_single_window_billed_at_min_of_self_and_next() -> None:
    """A fragment fully inside window 1 bills at min(window1, window2).

    window1=100, window2=200 -> min is 100. Here the own value already wins, so
    the result equals kWh * 100.
    """
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    # 3600 W for 1800 s = 1.8 kWh, inside the first window.
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 1800.0, 3600.0)
    result = compute_emissions([frag], trace)
    assert result.total_energy_kwh == pytest.approx(1.8)
    assert result.total_co2_g == pytest.approx(1.8 * 100.0)  # min(100, 200) = 100
    assert result.average_intensity == pytest.approx(100.0)


def test_single_window_next_is_lower_uses_next() -> None:
    """When the next window is lower, the down-estimate uses the next value."""
    # window1=300, window2=120 -> min is 120 (the next window).
    trace = _trace([("2025-01-01 00:00", 300.0), ("2025-01-01 00:30", 120.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 1800.0, 3600.0)  # 1.8 kWh
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(1.8 * 120.0)
    assert result.breakdown[0]["carbon_intensity"] == pytest.approx(120.0)


def test_last_window_uses_own_value_no_successor() -> None:
    """The final trace window has no successor, so it bills at its own value."""
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    # Fragment sits entirely in the last window (00:30..01:00).
    frag = Fragment(pd.Timestamp("2025-01-01 00:30"), 1800.0, 1000.0)  # 0.5 kWh
    result = compute_emissions([frag], trace)
    assert result.total_co2_g == pytest.approx(0.5 * 200.0)  # own value, no min


# --------------------------------------------------------------------------- #
#  Window-spanning fragment: split + weight by time                           #
# --------------------------------------------------------------------------- #
def test_fragment_spanning_two_windows_next_higher() -> None:
    """Crossing into a higher window: each piece bills at min(self, next).

    w1=100,w2=300,w3=300 (w2's successor w3 keeps it at 300). The w1 piece
    bills min(100,300)=100; the w2 piece bills min(300,300)=300.
    """
    trace = _trace(
        [
            ("2025-01-01 00:00", 100.0),
            ("2025-01-01 00:30", 300.0),
            ("2025-01-01 01:00", 300.0),
        ]
    )
    # Start 15 min into window 1, run 30 min: 15 min in w1, 15 min in w2.
    # 1000 W for 900 s each half => 0.25 kWh per half.
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    expected = 0.25 * 100.0 + 0.25 * 300.0  # min(100,300)=100 ; min(300,300)=300
    assert result.total_energy_kwh == pytest.approx(0.5)
    assert result.total_co2_g == pytest.approx(expected)  # 25 + 75 = 100 g
    assert result.average_intensity == pytest.approx(200.0)


def test_fragment_spanning_two_windows_next_lower_underestimates() -> None:
    """Crossing into a lower window: the w1 piece is down-estimated to w2.

    w1=300, w2=100, w3=100. The w1 piece bills min(300,100)=100; the w2 piece
    bills min(100,100)=100. The whole fragment ends up at 100 (well below the
    naive left-step 300/100 split), demonstrating the conservative estimate.
    """
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
    # window-1 bucket records the down-estimated intensity (100, not 300).
    assert result.breakdown[0]["carbon_intensity"] == pytest.approx(100.0)


def test_fragment_spanning_three_windows_min_rule() -> None:
    trace = _trace(
        [
            ("2025-01-01 00:00", 100.0),
            ("2025-01-01 00:30", 200.0),
            ("2025-01-01 01:00", 400.0),
        ]
    )
    # Start at 00:00, run 75 min: 30 in w1, 30 in w2, 15 in w3 (last window).
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 75 * 60.0, 2000.0)
    result = compute_emissions([frag], trace)
    # energy per minute = 2000 W * 60 s = 120000 Ws = 120000/3.6e6 kWh
    kwh_per_min = 2000.0 * 60.0 / 3.6e6
    # w1 piece: min(100,200)=100 ; w2 piece: min(200,400)=200 ; w3 (last): 400.
    expected = 30 * kwh_per_min * 100.0 + 30 * kwh_per_min * 200.0 + 15 * kwh_per_min * 400.0
    assert result.total_co2_g == pytest.approx(expected)


def test_min_rule_never_exceeds_left_step() -> None:
    """For any trace, the min-rule total is <= the old left-step total."""
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
    # min-rule: w1 min(250,90)=90 ; w2 min(90,400)=90 ; w3 min(400,120)=120 ;
    #           w4 last -> 120.
    min_total = kwh_per_window * (90 + 90 + 120 + 120)
    # old left-step: 250, 90, 400, 120 at their own values.
    left_step_total = kwh_per_window * (250 + 90 + 400 + 120)
    assert result.total_co2_g == pytest.approx(min_total)
    assert result.total_co2_g <= left_step_total


# --------------------------------------------------------------------------- #
#  Per-window breakdown                                                        #
# --------------------------------------------------------------------------- #
def test_per_window_breakdown_groups_by_window() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 300.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:15"), 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    bd = result.breakdown
    assert len(bd) == 2
    # window 1: min(100, 300) = 100 -> 0.25 kWh * 100 = 25 g
    assert bd[0]["window_start"] == pd.Timestamp("2025-01-01 00:00")
    assert bd[0]["carbon_intensity"] == pytest.approx(100.0)
    assert bd[0]["co2_g"] == pytest.approx(25.0)
    # window 2 is the last window -> own value 300 -> 0.25 kWh * 300 = 75 g
    assert bd[1]["window_start"] == pd.Timestamp("2025-01-01 00:30")
    assert bd[1]["carbon_intensity"] == pytest.approx(300.0)
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


# --------------------------------------------------------------------------- #
#  Timezone-naive consistency                                                  #
# --------------------------------------------------------------------------- #
def test_tz_aware_fragment_against_naive_trace_raises() -> None:
    trace = _trace([("2025-01-01 00:00", 100.0), ("2025-01-01 00:30", 200.0)])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00", tz="UTC"), 60.0, 1000.0)
    with pytest.raises(ValueError):
        compute_emissions([frag], trace)
