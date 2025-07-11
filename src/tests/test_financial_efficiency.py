import pandas as pd

from simulator.efficiency.metrics import financial_efficiency


def test_financial_efficiency_scales_with_price():
    df = pd.DataFrame({"duration": [1_000, 1_000]})
    tokens = 2_000
    cheap = financial_efficiency(df, tokens, gpu_hour_price=5)
    pricey = financial_efficiency(df, tokens, gpu_hour_price=15)
    assert pricey / cheap == 3


def test_financial_efficiency_units():
    df = pd.DataFrame({"duration": [3_600_000]})  # 1 hour total
    eff = financial_efficiency(df, total_tokens=1_000_000, gpu_hour_price=36)
    # 1 M tok, 1 h, 36 €/h  →  36 €/Mtok/s
    assert round(eff, 2) == 36.00
