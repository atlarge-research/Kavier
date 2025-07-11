import pandas as pd

from simulator.efficiency.metrics import sustainability_efficiency


def make_power(wh):
    return pd.DataFrame({"energy_usage": [wh * 1_000]})


def make_tasks(seconds):
    return pd.DataFrame({"duration": [seconds * 1_000]})


def test_energy_per_token_second():
    power = make_power(100)
    tasks = make_tasks(10)
    eff = sustainability_efficiency(power, tasks, total_tokens=1_000_000)
    assert eff == 1
