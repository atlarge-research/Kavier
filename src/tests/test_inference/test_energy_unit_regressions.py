"""Unit-correctness REGRESSION tests for kavier_energy.metrics.

These pin the three unit bugs that recent git history fixed, deriving the
expected value BY HAND so the test (not a copied "golden number") is the
authority. If any of these conversions regress, these fail.

Fixed bugs being guarded:
  * Joules -> Wh must be /3600 (1 Wh = 3600 J). A prior version used /1000,
    over-counting energy by exactly 3.6x (commit 641802c).
  * Efficiency is reported per MILLION tokens (commit e88c43a): energy/token
    scaled by 1e6.
  * Carbon is grams already (gCO2/kWh x kWh), summed as-is, no conversion
    (commit c21596f).
  * Energy efficiency has NO latency term (energy/token is dimensionless in
    time); the old version multiplied by latency -> 100x errors.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kavier_energy.metrics import (
    _extract_co2_emission_g,
    _extract_energy_wh,
    _total_gpu_hours,
    financial_efficiency,
    sustainability_efficiency,
    sustainability_efficiency_CO2,
)


def test_joules_to_wh_is_divide_by_3600_not_1000() -> None:
    # 7200 J is, by definition, exactly 2 Wh (7200 / 3600). The old /1000 bug
    # would have returned 7.2 (3.6x too high).
    power = pd.DataFrame({"energy_usage": [3600.0, 3600.0]})  # 7200 J total
    wh = _extract_energy_wh(power)
    assert wh == pytest.approx(2.0)
    # Explicitly exclude the regressed value.
    assert wh != pytest.approx(7.2)


def test_energy_efficiency_per_million_tokens_hand_derived() -> None:
    # 100 Wh over 250_000 tokens. By hand: 100 / 250_000 * 1e6 = 400 Wh/Mtoken.
    power = pd.DataFrame({"energy_usage": [100.0 * 3600.0]})  # 100 Wh in Joules
    tasks = pd.DataFrame({"duration": [5_000]})  # latency present but must not matter
    eff = sustainability_efficiency(power, tasks, total_tokens=250_000)
    assert eff == pytest.approx(400.0)


def test_carbon_grams_summed_as_is_no_conversion() -> None:
    # OpenDC carbon_emission is already grams. Summed verbatim, scaled per 1M tokens.
    # By hand: (12 + 8) g over 4e6 tokens * 1e6 = 20 / 4e6 * 1e6 = 5 gCO2/Mtoken.
    power = pd.DataFrame({"carbon_emission": [12.0, 8.0]})
    tasks = pd.DataFrame({"duration": [1_000]})
    eff = sustainability_efficiency_CO2(power, tasks, total_tokens=4_000_000)
    assert _extract_co2_emission_g(power) == pytest.approx(20.0)
    assert eff == pytest.approx(5.0)


def test_total_gpu_hours_ms_to_hours() -> None:
    # duration is per-task latency in MILLISECONDS; summed -> GPU-time.
    # 7_200_000 ms = 7.2e6 / 1000 / 3600 = 2 GPU-hours, by hand.
    tasks = pd.DataFrame({"duration": [3_600_000, 3_600_000]})
    assert _total_gpu_hours(tasks) == pytest.approx(2.0)


def test_financial_efficiency_hand_derived_dollars_per_million_tokens() -> None:
    # 2 GPU-hours at $10/h = $20, over 8e6 tokens -> 20/8e6*1e6 = 2.5 $/Mtoken.
    tasks = pd.DataFrame({"duration": [3_600_000, 3_600_000]})  # 2 GPU-h
    eff = financial_efficiency(tasks, total_tokens=8_000_000, gpu_hour_price=10.0)
    assert eff == pytest.approx(2.5)
