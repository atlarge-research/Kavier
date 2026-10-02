"""Tests for the cost efficiency [$/Mtoken] in ``kavier.sdk.energy.metrics`` and its summary.

2 GPU-h x $15 / 3M tokens = $10/Mtoken. The inputs are unequal and distinct, so a sum -> mean
change or an argument swap changes the result.
"""

from __future__ import annotations

import pandas as pd

from kavier.sdk.energy.metrics import efficiency_summary, financial_efficiency


def _power_df() -> pd.DataFrame:
    # efficiency_summary also computes energy and carbon, so both OpenDC columns are required.
    return pd.DataFrame({"energy_usage": [3_600.0], "carbon_emission": [1.0]})


def test_financial_efficiency_dollars_per_mtoken():
    # duration is per-task ms, one GPU per task, summed -> total GPU-time.
    # 2_400_000 + 4_800_000 = 7_200_000 ms = 7200 s = 2 GPU-hours.
    # 2 GPU-h * $15/h = $30 spent over 3M tokens = $10 per Mtoken.
    # .mean() would give 1 h and .first() 0.667 h. hours (2), price (15), tokens/1e6 (3) and
    # result (10) are distinct, so swapped arguments change the result.
    df = pd.DataFrame({"duration": [2_400_000, 4_800_000]})
    eff = financial_efficiency(df, total_tokens=3_000_000, gpu_hour_price=15)
    assert eff == 10.0


def test_summary_financial_none_without_price():
    # No default price: financial is None.
    tasks = pd.DataFrame({"duration": [1_000]})
    summary = efficiency_summary(tasks, _power_df(), total_tokens=1_000)
    assert summary["financial_efficiency ($/Mtoken)"] is None


def test_summary_computes_financial_when_priced():
    # efficiency_summary passes (tasks, total_tokens, price) to financial_efficiency.
    # 7_200_000 ms = 2 GPU-h; 2h * $15/h = $30 over 3M tokens = $10/Mtoken.
    tasks = pd.DataFrame({"duration": [2_400_000, 4_800_000]})
    summary = efficiency_summary(tasks, _power_df(), total_tokens=3_000_000, gpu_hour_price=15)
    assert summary["financial_efficiency ($/Mtoken)"] == 10.0


def test_summary_total_tokens_cast_to_int():
    # A float total_tokens is returned as int.
    tasks = pd.DataFrame({"duration": [1_000]})
    summary = efficiency_summary(tasks, _power_df(), total_tokens=1_000.0)
    tok = summary["total_tokens"]
    assert isinstance(tok, int) and tok == 1_000
