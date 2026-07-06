"""Property-based invariants for kavier.sdk.co2 emissions (general laws complementing
test_emissions.py's pinned cases): energy=power*time/3.6e6, CO2 non-negative, down-estimate
never exceeds left-step, billed intensity is always a real trace value, co2_kg==co2_g/1000,
weighted-mean intensity stays inside the billed range, zero/negative-duration edges."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kavier.sdk.co2.engine import CarbonTrace, Fragment, compute_emissions

_STEP = dt.timedelta(minutes=30)


def _trace(intensities: list[float]) -> CarbonTrace:
    base = pd.Timestamp("2025-01-01 00:00")
    ts = [base + i * _STEP for i in range(len(intensities))]
    df = pd.DataFrame({"timestamp": ts, "carbon_intensity": intensities})
    return CarbonTrace.from_dataframe(df, step=_STEP)


def _left_step_total(frag, trace) -> float:
    # Reference 'hold-the-left-point' total: bill every segment at its OWN window intensity,
    # with no min(self, next) down-estimation. An independent second method; because the code
    # bills min(own, next) <= own, its total must never exceed this reference.
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
def test_energy_is_power_times_time_conserved_across_window_splits(intensities, power_w, minutes) -> None:
    trace = _trace(intensities)
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), minutes * 60.0, power_w)
    result = compute_emissions([frag], trace)

    # 1 kWh = 1000 W * 3600 s = 3.6e6 W*s (unit-definition oracle, independent of the impl).
    # Total must equal power*duration/3.6e6 no matter how the fragment is split across windows,
    # so this also asserts segmentation conserves energy (no double-count / dropped segment).
    expected_kwh = power_w * (minutes * 60.0) / 3.6e6
    assert result.total_energy_kwh == pytest.approx(expected_kwh)
    # Physical floor: energy and emissions are never negative.
    assert result.total_energy_kwh >= 0.0
    assert result.total_co2_g >= 0.0
    # Per-window breakdown must partition the total (conservation, not a snapshot).
    assert sum(b["co2_g"] for b in result.breakdown) == pytest.approx(result.total_co2_g)
    assert sum(b["energy_kwh"] for b in result.breakdown) == pytest.approx(result.total_energy_kwh)


@given(
    intensities=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=10),
    power_w=st.floats(min_value=1.0, max_value=5000.0),
)
@settings(max_examples=100, deadline=None)
def test_down_estimate_never_exceeds_left_step(intensities, power_w) -> None:
    trace = _trace(intensities)
    # Span the whole trace so every min(self, next) decision fires.
    total_seconds = len(intensities) * _STEP.total_seconds()
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), total_seconds, power_w)
    result = compute_emissions([frag], trace)
    left = _left_step_total(frag, trace)
    # min(own, next) <= own, so the down-estimated total can only be <= the hold-left total.
    # A mutation billing max(own, next) (or plain own on a falling window) would exceed `left`.
    assert result.total_co2_g <= left + 1e-6


@given(intensities=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=6))
@settings(max_examples=60, deadline=None)
def test_billed_intensity_is_always_a_real_trace_value(intensities) -> None:
    trace = _trace(intensities)
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), len(intensities) * 1800.0, 1000.0)
    result = compute_emissions([frag], trace)
    real = set(intensities)
    # min(own, next) picks one of the two adjacent trace values (never an average/blend), so
    # every billed intensity must appear verbatim in the trace. Mutating to (own+next)/2 fails.
    for b in result.breakdown:
        assert any(abs(b["carbon_intensity"] - v) < 1e-9 for v in real)


@given(
    intensities=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=2, max_size=8),
    power_w=st.floats(min_value=1.0, max_value=5000.0),
)
@settings(max_examples=80, deadline=None)
def test_average_intensity_within_billed_range(intensities, power_w) -> None:
    trace = _trace(intensities)
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), len(intensities) * 1800.0, power_w)
    result = compute_emissions([frag], trace)
    billed = [b["carbon_intensity"] for b in result.breakdown]
    # An energy-weighted mean is bounded by the extremes of the values averaged. A mutation
    # computing co2*energy (or an unweighted sum) instead of co2/energy would escape [min,max].
    assert min(billed) - 1e-9 <= result.average_intensity <= max(billed) + 1e-9


def test_co2_kg_is_grams_over_1000() -> None:
    trace = _trace([100.0, 300.0, 50.0])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 3 * 1800.0, 2000.0)
    result = compute_emissions([frag], trace)
    # 1 kg == 1000 g by definition (unit oracle). Mutating the divisor to /100 makes this fail.
    assert result.total_co2_kg == pytest.approx(result.total_co2_g / 1000.0)


def test_zero_duration_fragment_contributes_nothing() -> None:
    trace = _trace([100.0, 200.0])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), 0.0, 5000.0)
    result = compute_emissions([frag], trace)
    assert result.total_energy_kwh == 0.0
    assert result.total_co2_g == 0.0
    assert result.breakdown == []
    # No-energy branch must return 0.0, not attempt total_co2_g/total_energy_kwh (0/0 -> nan/raise).
    assert result.average_intensity == 0.0


def test_negative_duration_fragment_raises() -> None:
    trace = _trace([100.0, 200.0])
    frag = Fragment(pd.Timestamp("2025-01-01 00:00"), -60.0, 1000.0)
    # The duration guard must fire *specifically* (message mentions duration). Without the guard,
    # end = start - 60s < coverage_start, which would instead raise the "outside coverage" error,
    # so matching on "duration" pins that this branch — not the coverage check — is responsible.
    with pytest.raises(ValueError, match="duration"):
        compute_emissions([frag], trace)
