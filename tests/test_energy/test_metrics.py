"""``kavier.sdk.energy.metrics`` through a real parquet round-trip (de-slop Phase 0 Deliverable B).

The function-level arithmetic of this module (Joules->Wh, grams passthrough, ms->GPU-hours, the
None-without-price contract, the int-cast of total_tokens) is already hand-derived and pinned in
tests/test_inference/test_energy_unit_regressions.py, test_sustainability_efficiency.py and
test_financial_efficiency.py -- all three call the functions on DataFrames built directly in memory.
None of them go through an actual parquet FILE, which is how these functions are really fed in
production (OpenDC's powerSource.parquet + Kavier's tasks.parquet, read via ``kavier energy``'s
``pd.read_parquet``, see kavier/cli/energy.py). This module closes that gap: it writes real
``powerSource.parquet`` / ``tasks.parquet`` files to ``tmp_path``, reads them back, and pins
``efficiency_summary`` against hand-derived numbers -- distinct from the numbers used in the
in-memory tests above, so this isn't a copy-paste of an existing assertion under a new name.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kavier.sdk.energy.metrics import efficiency_summary, financial_efficiency, sustainability_efficiency

# 3 power-source rows: energy_usage (Joules) sums to 180_000 J = 50 Wh (180_000 / 3600).
# carbon_emission (already grams, no conversion) sums to 15 g.
_ENERGY_J = [72_000.0, 90_000.0, 18_000.0]
_CARBON_G = [7.0, 5.0, 3.0]
# 2 task rows: duration (ms) sums to 3_600_000 ms = 1 GPU-hour (3_600_000 / 1000 / 3600).
# total_tokens sums to 100_000.
_DURATION_MS = [1_800_000, 1_800_000]
_TOTAL_TOKENS = [60_000, 40_000]


def _write_energy_parquet(tmp_path: Path) -> tuple[Path, Path]:
    """Write a tiny synthetic powerSource.parquet + tasks.parquet to ``tmp_path``; return their paths."""
    power_path = tmp_path / "powerSource.parquet"
    tasks_path = tmp_path / "tasks.parquet"
    pd.DataFrame({"energy_usage": _ENERGY_J, "carbon_emission": _CARBON_G}).to_parquet(power_path)
    pd.DataFrame({"duration": _DURATION_MS, "total_tokens": _TOTAL_TOKENS}).to_parquet(tasks_path)
    return power_path, tasks_path


def test_parquet_roundtrip_preserves_the_columns_the_metrics_need(tmp_path: Path) -> None:
    # Wiring precondition for every test below: writing then reading back via pyarrow must not rename,
    # drop, or reorder the columns _extract_energy_wh/_extract_co2_emission_g/_total_gpu_hours read.
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    power = pd.read_parquet(power_path)
    tasks = pd.read_parquet(tasks_path)
    assert list(power.columns) == ["energy_usage", "carbon_emission"]
    assert list(tasks.columns) == ["duration", "total_tokens"]
    assert len(power) == 3 and len(tasks) == 2


def test_sustainability_efficiency_from_a_real_parquet_file(tmp_path: Path) -> None:
    power_path, _ = _write_energy_parquet(tmp_path)
    power = pd.read_parquet(power_path)
    # By hand: 180_000 J / 3600 = 50 Wh, over 100_000 tokens -> 50 / 100_000 * 1e6 = 500 Wh/Mtoken.
    eff = sustainability_efficiency(power, tasks=pd.DataFrame({"duration": [1]}), total_tokens=100_000)
    assert eff == pytest.approx(500.0)


def test_financial_efficiency_from_a_real_parquet_file(tmp_path: Path) -> None:
    _, tasks_path = _write_energy_parquet(tmp_path)
    tasks = pd.read_parquet(tasks_path)
    # By hand: 3_600_000 ms = 1 GPU-hour; 1 h * $4/h = $4, over 100_000 tokens -> 4 / 100_000 * 1e6 = $40/Mtoken.
    cost = financial_efficiency(tasks, total_tokens=100_000, gpu_hour_price=4.0)
    assert cost == pytest.approx(40.0)


def test_efficiency_summary_from_real_parquet_files(tmp_path: Path) -> None:
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    power = pd.read_parquet(power_path)
    tasks = pd.read_parquet(tasks_path)

    summary = efficiency_summary(tasks, power, total_tokens=int(tasks["total_tokens"].sum()), gpu_hour_price=4.0)

    # Same three hand-derivations as above, now routed through the summary + the file-sourced total_tokens.
    assert summary["total_tokens"] == 100_000
    assert summary["energy_efficiency (Wh/Mtoken)"] == pytest.approx(500.0)
    assert summary["carbon_efficiency (gCO2/Mtoken)"] == pytest.approx(150.0)  # 15 g / 100_000 * 1e6
    assert summary["financial_efficiency ($/Mtoken)"] == pytest.approx(40.0)


def test_efficiency_summary_financial_none_without_price_from_real_parquet_files(tmp_path: Path) -> None:
    power_path, tasks_path = _write_energy_parquet(tmp_path)
    power = pd.read_parquet(power_path)
    tasks = pd.read_parquet(tasks_path)

    summary = efficiency_summary(tasks, power, total_tokens=int(tasks["total_tokens"].sum()))
    assert summary["financial_efficiency ($/Mtoken)"] is None
    # energy/carbon are still computed even though price was omitted.
    assert summary["energy_efficiency (Wh/Mtoken)"] == pytest.approx(500.0)
