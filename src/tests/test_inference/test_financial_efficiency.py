import pandas as pd

from kavier_energy.metrics import efficiency_summary, financial_efficiency


def test_financial_efficiency_scales_with_price():
    df = pd.DataFrame({"duration": [1_000, 1_000]})
    tokens = 2_000
    cheap = financial_efficiency(df, tokens, gpu_hour_price=5)
    pricey = financial_efficiency(df, tokens, gpu_hour_price=15)
    assert pricey / cheap == 3


def test_financial_efficiency_units():
    df = pd.DataFrame({"duration": [3_600_000]})  # 1 GPU-hour total
    eff = financial_efficiency(df, total_tokens=1_000_000, gpu_hour_price=36)
    # 1 GPU-h at 36/h over 1M tokens -> 36 $/Mtoken (seconds cancel; not per-second)
    assert eff == 36


def test_financial_is_none_without_price():
    # the $/Mtoken rate is user-supplied; with no price it must not be invented
    tasks = pd.DataFrame({"duration": [1_000]})
    power = pd.DataFrame({"energy_usage": [1_000], "carbon_emission": [1.0]})
    summary = efficiency_summary(tasks, power, total_tokens=1_000)
    assert summary["financial_efficiency ($/Mtoken)"] is None
