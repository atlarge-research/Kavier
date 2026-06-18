import pandas as pd

from kavier_energy.metrics import (
    sustainability_efficiency,
    sustainability_efficiency_CO2,
)


def make_power_wh(wh):
    # OpenDC stores energy_usage in Joules; 1 Wh = 3600 J
    return pd.DataFrame({"energy_usage": [wh * 3600]})


def make_tasks(seconds):
    return pd.DataFrame({"duration": [seconds * 1_000]})  # s -> ms


def test_energy_per_million_tokens():
    power = make_power_wh(100)  # 100 Wh
    tasks = make_tasks(10)
    eff = sustainability_efficiency(power, tasks, total_tokens=1_000_000)
    assert eff == 100  # Wh per 1M tokens, no latency term


def test_energy_efficiency_independent_of_latency():
    power = make_power_wh(100)
    fast = sustainability_efficiency(power, make_tasks(1), total_tokens=1_000_000)
    slow = sustainability_efficiency(power, make_tasks(100), total_tokens=1_000_000)
    assert fast == slow  # the old bug made these differ 100x


def test_carbon_per_million_tokens():
    power = pd.DataFrame({"carbon_emission": [50.0]})  # grams
    tasks = make_tasks(10)
    eff = sustainability_efficiency_CO2(power, tasks, total_tokens=1_000_000)
    assert eff == 50.0  # gCO2 per 1M tokens
