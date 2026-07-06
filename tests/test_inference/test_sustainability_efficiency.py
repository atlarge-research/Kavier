import pandas as pd
import pytest

from kavier.sdk.energy.metrics import (
    efficiency_summary,
    financial_efficiency,
    sustainability_efficiency,
    sustainability_efficiency_CO2,
)


def joules_from_wh(*wh_values):
    # OpenDC stores energy_usage in Joules; 1 Wh = 3600 J.
    return pd.DataFrame({"energy_usage": [wh * 3600 for wh in wh_values]})


def tasks_from_seconds(*seconds_values):
    # Kavier tasks store per-task ``duration`` in ms; 1 s = 1000 ms.
    return pd.DataFrame({"duration": [s * 1_000 for s in seconds_values]})


def test_energy_wh_conversion_and_per_mtoken_scaling():
    # 100 Wh of energy over 500_000 tokens.
    # J->Wh: 360_000 J / 3600 = 100 Wh; per-Mtoken: 100 / 500_000 * 1e6 = 200.
    power = joules_from_wh(100)
    eff = sustainability_efficiency(power, tasks_from_seconds(10), total_tokens=500_000)
    assert eff == pytest.approx(200.0)
    # cross-check vs the old /1000 bug: that path would give 360 Wh -> 720, not 200.
    assert eff != pytest.approx(720.0)


def test_energy_sums_all_powersource_rows():
    # Two power-source rows must be summed before conversion:
    # (40 + 60) Wh = 100 Wh over 1e6 tokens -> 100 Wh/Mtoken.
    # A first-row-only (.iloc[0]) impl would give 40, not 100.
    power = joules_from_wh(40, 60)
    eff = sustainability_efficiency(power, tasks_from_seconds(1), total_tokens=1_000_000)
    assert eff == pytest.approx(100.0)


def test_energy_efficiency_ignores_task_duration():
    # ``tasks`` is unused by the energy metric; a 100x slower run yields the
    # same Wh/Mtoken. The old bug scaled energy by duration and broke this.
    power = joules_from_wh(100)
    fast = sustainability_efficiency(power, tasks_from_seconds(1), total_tokens=1_000_000)
    slow = sustainability_efficiency(power, tasks_from_seconds(100), total_tokens=1_000_000)
    assert fast == slow


def test_energy_missing_column_raises():
    # No energy_usage column -> explicit ValueError, not a silent 0 or KeyError.
    bad = pd.DataFrame({"carbon_emission": [1.0]})
    with pytest.raises(ValueError, match="energy_usage"):
        sustainability_efficiency(bad, tasks_from_seconds(1), total_tokens=1_000_000)


def test_carbon_grams_passthrough_sum_and_scaling():
    # carbon_emission is already grams (no /3600). Two rows sum: 30 + 20 = 50 g.
    # Per-Mtoken over 250_000 tokens: 50 / 250_000 * 1e6 = 200 gCO2/Mtoken.
    # If it wrongly divided by 3600 (like energy) this would be ~0.056, not 200.
    power = pd.DataFrame({"carbon_emission": [30.0, 20.0]})
    eff = sustainability_efficiency_CO2(power, tasks_from_seconds(10), total_tokens=250_000)
    assert eff == pytest.approx(200.0)


def test_carbon_missing_column_raises():
    bad = pd.DataFrame({"energy_usage": [3600.0]})
    with pytest.raises(ValueError, match="carbon_emission"):
        sustainability_efficiency_CO2(bad, tasks_from_seconds(1), total_tokens=1_000_000)


def test_financial_gpu_hours_times_rate_per_mtoken():
    # duration ms -> GPU-hours: (3_600_000 + 3_600_000) ms = 7200 s = 2 GPU-h.
    # Cost: 2 h * $3/h = $6; per-Mtoken over 3e6 tokens: 6 / 3e6 * 1e6 = $2.
    tasks = tasks_from_seconds(3600, 3600)
    cost = financial_efficiency(tasks, total_tokens=3_000_000, gpu_hour_price=3.0)
    assert cost == pytest.approx(2.0)


def test_summary_financial_none_without_price():
    # gpu_hour_price omitted -> financial entry is None; energy/carbon still computed.
    power = pd.DataFrame({"energy_usage": [360_000.0], "carbon_emission": [50.0]})
    summary = efficiency_summary(tasks_from_seconds(1), power, total_tokens=1_000_000)
    assert summary["financial_efficiency ($/Mtoken)"] is None
    # 360_000 J / 3600 = 100 Wh over 1e6 tokens -> 100 Wh/Mtoken (independent oracle).
    assert summary["energy_efficiency (Wh/Mtoken)"] == pytest.approx(100.0)
    # 50 g over 1e6 tokens -> 50 gCO2/Mtoken.
    assert summary["carbon_efficiency (gCO2/Mtoken)"] == pytest.approx(50.0)


def test_summary_total_tokens_cast_to_int():
    # A float token count is cast to a Python int in the summary.
    power = pd.DataFrame({"energy_usage": [3600.0], "carbon_emission": [1.0]})
    summary = efficiency_summary(tasks_from_seconds(1), power, total_tokens=1_000_000.0)
    assert summary["total_tokens"] == 1_000_000
    assert isinstance(summary["total_tokens"], int)


def test_summary_financial_present_with_price():
    # With a price, financial matches the hand-derived $/Mtoken.
    # 2 GPU-h * $5/h = $10 over 2e6 tokens -> $5/Mtoken.
    power = pd.DataFrame({"energy_usage": [3600.0], "carbon_emission": [1.0]})
    tasks = tasks_from_seconds(3600, 3600)  # 2 GPU-hours
    summary = efficiency_summary(tasks, power, total_tokens=2_000_000, gpu_hour_price=5.0)
    assert summary["financial_efficiency ($/Mtoken)"] == pytest.approx(5.0)
