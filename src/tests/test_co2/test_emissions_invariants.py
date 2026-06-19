"""Property-based invariants for kavier_co2 emissions.

test_emissions.py pins many hand-computed cases. This file adds the GENERAL
laws that must hold for arbitrary traces/fragments:
  * energy = power * time / 3.6e6 kWh, and total CO2 = sum over windows;
  * all outputs (energy, CO2) are non-negative;
  * the conservative down-estimate NEVER exceeds the naive left-step total, for
    any trace shape (the documented guarantee);
  * total_co2_kg == total_co2_g / 1000 and average_intensity is in
    [min_used, max_used] intensity;
  * billed intensity is always one of the trace's own values (never invented).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kavier_co2.emissions import CarbonTrace, Fragment, compute_emissions

_STEP = dt.timedelta(minutes=30)


def _trace(intensities: list[float]) -> CarbonTrace:
    base = pd.Timestamp("2025-01-01 00:00")
    ts = [base + i * _STEP for i in range(len(intensities))]
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": intensities})
    return CarbonTrace.from_dataframe(df, step=_STEP)


def _left_step_total(intensities, frag, trace) -> float:
    """Reference 'hold-the-left-point' total (no min rule) for the same fragment."""
    total = 0.0
    cursor = frag.start_time
    end = frag.start_time + pd.Timedelta(seconds=frag.duration_s)
    while cursor < end:
        wi = int(trace.timestamps.searchsorted(cursor, side="right")) - 1
        window_end = trace.timestamps[wi] + trace.step
        seg_end = min(end, window_end)
        seg_s = (seg_end - cursor).total_seconds()
        kwh = frag.power_w * seg_s / 3.6e6
        total += kwh * float(trace.intensities.iloc[wi])
        cursor = seg_end
    return total


@given(
    intensities=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=8),
    power_w=st.floats(min_value=1.0, max_value=5000.0),
    minutes=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=80, deadline=None)
def test_outputs_nonnegative_and_energy_formula(intensities, power_w, minutes) -> None:
    trace = _trace(intensities)
    # Keep the fragment safely inside coverage: start at window 0, run <= 1 window.
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), minutes * 60.0, power_w)
    result = compute_emissions([frag], trace)

    assert result.total_energy_kwh >= 0.0
    assert result.total_co2_g >= 0.0
    # Energy is exactly power * time / 3.6e6 regardless of intensities.
    expected_kwh = power_w * (minutes * 60.0) / 3.6e6
    assert result.total_energy_kwh == pytest.approx(expected_kwh)
    # Sum of per-window CO2 equals the reported total.
    assert sum(b["co2_g"] for b in result.breakdown) == pytest.approx(result.total_co2_g)


@given(
    intensities=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=10),
    power_w=st.floats(min_value=1.0, max_value=5000.0),
)
@settings(max_examples=100, deadline=None)
def test_down_estimate_never_exceeds_left_step(intensities, power_w) -> None:
    trace = _trace(intensities)
    # Span the whole trace so every min(self, next) decision is exercised.
    total_seconds = (len(intensities)) * _STEP.total_seconds()
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), total_seconds, power_w)
    result = compute_emissions([frag], trace)
    left = _left_step_total(intensities, frag, trace)
    assert result.total_co2_g <= left + 1e-6


@given(intensities=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=6))
@settings(max_examples=60, deadline=None)
def test_billed_intensity_is_a_real_trace_value(intensities) -> None:
    trace = _trace(intensities)
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), len(intensities) * 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    real = set(intensities)
    for b in result.breakdown:
        assert any(abs(b["carbon_intensity"] - v) < 1e-9 for v in real)


def test_co2_kg_is_grams_over_1000_and_avg_intensity_in_range() -> None:
    trace = _trace([100.0, 300.0, 50.0])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 3 * 1800.0, 2000.0)
    result = compute_emissions([frag], trace)
    assert result.total_co2_kg == pytest.approx(result.total_co2_g / 1000.0)
    used = [b["carbon_intensity"] for b in result.breakdown]
    assert min(used) - 1e-9 <= result.average_intensity <= max(used) + 1e-9


def test_zero_duration_fragment_contributes_nothing() -> None:
    trace = _trace([100.0, 200.0])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 0.0, 5000.0)
    result = compute_emissions([frag], trace)
    assert result.total_energy_kwh == 0.0
    assert result.total_co2_g == 0.0
