"""Unit-correctness regression guards for kavier.sdk.energy.metrics (expected values hand-derived):
Joules->Wh is /3600, not /1000 (commit 641802c, was 3.6x high); efficiency per MILLION tokens
(commit e88c43a); carbon is grams summed as-is (commit c21596f); energy/token has no latency term."""

from __future__ import annotations

import pandas as pd
import pytest

from kavier.sdk.energy.metrics import (
    _extract_co2_emission_g,
    _extract_energy_wh,
    _total_gpu_hours,
    efficiency_summary,
    financial_efficiency,
    sustainability_efficiency,
    sustainability_efficiency_CO2,
)


def test_joules_to_wh_is_divide_by_3600_not_1000() -> None:
    # 7200 J = 2 Wh (/3600); old /1000 bug returned 7.2 (3.6x high).
    power = pd.DataFrame({"energy_usage": [3600.0, 3600.0]})  # 7200 J total
    wh = _extract_energy_wh(power)
    assert wh == pytest.approx(2.0)
    assert wh != pytest.approx(7.2)  # exclude regressed value


def test_extract_energy_wh_raises_when_column_absent() -> None:
    # No ``energy_usage`` column -> must raise, not silently return 0 / a wrong column.
    with pytest.raises(ValueError, match="energy_usage"):
        _extract_energy_wh(pd.DataFrame({"carbon_emission": [1.0]}))


def test_energy_efficiency_per_million_tokens_hand_derived() -> None:
    # 100 Wh over 250_000 tokens. By hand: 100 / 250_000 * 1e6 = 400 Wh/Mtoken.
    power = pd.DataFrame({"energy_usage": [100.0 * 3600.0]})  # 100 Wh in Joules
    tasks = pd.DataFrame({"duration": [5_000]})  # latency present but must not matter
    eff = sustainability_efficiency(power, tasks, total_tokens=250_000)
    assert eff == pytest.approx(400.0)


def test_energy_efficiency_ignores_tasks_no_latency_term() -> None:
    # Documented contract: energy/token has no latency term, so ``tasks`` is inert.
    # Same power + tokens, wildly different durations -> identical result (invariant).
    power = pd.DataFrame({"energy_usage": [3600.0]})
    fast = sustainability_efficiency(power, pd.DataFrame({"duration": [1]}), total_tokens=1_000)
    slow = sustainability_efficiency(power, pd.DataFrame({"duration": [10_000_000]}), total_tokens=1_000)
    assert fast == pytest.approx(slow)


def test_carbon_grams_summed_as_is_no_conversion() -> None:
    # OpenDC carbon_emission is already grams. Summed verbatim, scaled per 1M tokens.
    # By hand: (12 + 8) g over 4e6 tokens * 1e6 = 20 / 4e6 * 1e6 = 5 gCO2/Mtoken.
    power = pd.DataFrame({"carbon_emission": [12.0, 8.0]})
    tasks = pd.DataFrame({"duration": [1_000]})
    eff = sustainability_efficiency_CO2(power, tasks, total_tokens=4_000_000)
    assert _extract_co2_emission_g(power) == pytest.approx(20.0)
    assert eff == pytest.approx(5.0)


def test_extract_co2_raises_when_column_absent() -> None:
    # No ``carbon_emission`` column -> must raise, not fall back to another column.
    with pytest.raises(ValueError, match="carbon_emission"):
        _extract_co2_emission_g(pd.DataFrame({"energy_usage": [1.0]}))


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


def test_efficiency_summary_financial_is_none_when_price_unset() -> None:
    # 1 Wh (3600 J), 10 g CO2, 1 GPU-h, 1M tokens -> each per-Mtoken value is unscaled.
    # By hand: energy=1, carbon=10, tokens int-cast; financial omitted when price unset.
    power = pd.DataFrame({"energy_usage": [3600.0], "carbon_emission": [10.0]})
    tasks = pd.DataFrame({"duration": [3_600_000]})  # 1 GPU-h
    out = efficiency_summary(tasks, power, total_tokens=1_000_000)
    assert out["energy_efficiency (Wh/Mtoken)"] == pytest.approx(1.0)
    assert out["carbon_efficiency (gCO2/Mtoken)"] == pytest.approx(10.0)
    assert out["financial_efficiency ($/Mtoken)"] is None
    assert out["total_tokens"] == 1_000_000


def test_efficiency_summary_financial_present_when_price_set() -> None:
    # Same 1 GPU-h / 1M tokens; $2/h -> 2/1e6*1e6 = $2/Mtoken (hand-derived).
    power = pd.DataFrame({"energy_usage": [3600.0], "carbon_emission": [10.0]})
    tasks = pd.DataFrame({"duration": [3_600_000]})  # 1 GPU-h
    out = efficiency_summary(tasks, power, total_tokens=1_000_000, gpu_hour_price=2.0)
    assert out["financial_efficiency ($/Mtoken)"] == pytest.approx(2.0)
