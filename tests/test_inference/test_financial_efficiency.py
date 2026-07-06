"""``kavier.sdk.energy.metrics`` cost efficiency ($/Mtoken) and the summary that forwards it.

Every number is hand-derived (2 GPU-h × $15 ÷ 3M tokens = $10/Mtoken) from unequal, mutually
distinct inputs, so a sum→mean/first collapse or an argument swap is falsifiable — never the
function's own output. The None-without-price and int-cast behaviors are pinned as contracts.
"""

from __future__ import annotations

import pandas as pd

from kavier.sdk.energy.metrics import efficiency_summary, financial_efficiency


def _power_df() -> pd.DataFrame:
    # efficiency_summary always evaluates the energy/carbon paths too, so the
    # powerSource frame needs both OpenDC columns present (values irrelevant here).
    return pd.DataFrame({"energy_usage": [3_600.0], "carbon_emission": [1.0]})


def test_financial_efficiency_dollars_per_mtoken():
    # duration is per-task ms, one GPU per task, summed -> total GPU-time.
    # 2_400_000 + 4_800_000 = 7_200_000 ms = 7200 s = 2 GPU-hours.
    # 2 GPU-h * $15/h = $30 spent over 3M tokens = $10 per Mtoken.
    # Unequal rows so a .sum()->.mean()/first() mutation (would give 1h/0.667h) goes red;
    # hours(2), price(15), tokens/1e6(3), result(10) are all distinct so an
    # arg-swap (price<->hours<->tokens) also goes red.
    df = pd.DataFrame({"duration": [2_400_000, 4_800_000]})
    eff = financial_efficiency(df, total_tokens=3_000_000, gpu_hour_price=15)
    assert eff == 10.0


def test_summary_financial_none_without_price():
    # The $/Mtoken rate is user-supplied; with no price it must not be invented.
    tasks = pd.DataFrame({"duration": [1_000]})
    summary = efficiency_summary(tasks, _power_df(), total_tokens=1_000)
    assert summary["financial_efficiency ($/Mtoken)"] is None


def test_summary_computes_financial_when_priced():
    # Same derivation as above, routed through efficiency_summary to pin that it
    # forwards (tasks, total_tokens, price) into financial_efficiency correctly.
    # 7_200_000 ms = 2 GPU-h; 2h * $15/h = $30 over 3M tokens = $10/Mtoken.
    tasks = pd.DataFrame({"duration": [2_400_000, 4_800_000]})
    summary = efficiency_summary(tasks, _power_df(), total_tokens=3_000_000, gpu_hour_price=15)
    assert summary["financial_efficiency ($/Mtoken)"] == 10.0


def test_summary_total_tokens_cast_to_int():
    # total_tokens is echoed back as a plain int, not the caller's float.
    tasks = pd.DataFrame({"duration": [1_000]})
    summary = efficiency_summary(tasks, _power_df(), total_tokens=1_000.0)
    tok = summary["total_tokens"]
    assert isinstance(tok, int) and tok == 1_000
