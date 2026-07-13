"""Per-Mtoken efficiency: energy (Wh), carbon (gCO2), cost ($) from OpenDC powerSource + Kavier tasks."""

from __future__ import annotations

import pandas as pd

from kavier.sdk.units import SECONDS_PER_HOUR, TOKENS_PER_MTOKEN


def _extract_energy_wh(powerSource: pd.DataFrame) -> float:
    # OpenDC energy_usage is JOULES (W·s); 1 Wh = 3600 J, so divide by 3600.
    if "energy_usage" in powerSource.columns:
        return powerSource["energy_usage"].sum() / SECONDS_PER_HOUR
    raise ValueError("energy_usage not in the powerSource.parquet file")


def _extract_co2_emission_g(powerSource: pd.DataFrame) -> float:
    # OpenDC carbon_emission is already GRAMS (gCO2/kWh * kWh); no conversion.
    if "carbon_emission" in powerSource.columns:
        return float(powerSource["carbon_emission"].sum())
    raise ValueError("carbon_emission not in the powerSource.parquet file")


def _total_gpu_hours(tasks: pd.DataFrame) -> float:
    # 1 GPU/task, ``duration`` per-task ms -> summed = total GPU-time, ms -> h.
    return tasks["duration"].sum() / 1_000 / SECONDS_PER_HOUR


def sustainability_efficiency(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    """Energy efficiency: Wh per million tokens (lower is better)."""
    # ``tasks`` unused but kept for a stable signature alongside the CO2 variant.
    return _extract_energy_wh(powerSource) / total_tokens * TOKENS_PER_MTOKEN


def sustainability_efficiency_CO2(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    """Carbon efficiency: gCO2 per million tokens (lower is better)."""
    return _extract_co2_emission_g(powerSource) / total_tokens * TOKENS_PER_MTOKEN


def financial_efficiency(tasks: pd.DataFrame, total_tokens: int, gpu_hour_price: float) -> float:
    """Cost efficiency: $/Mtoken based on GPU-hours × rate (electricity ~2-5% omitted)."""
    return _total_gpu_hours(tasks) * gpu_hour_price / total_tokens * TOKENS_PER_MTOKEN


def efficiency_summary(
    tasks_df: pd.DataFrame,
    powerSource_df: pd.DataFrame,
    total_tokens: int,
    gpu_hour_price: float | None = None,
) -> dict[str, float | None]:
    """Aggregate energy/carbon/financial efficiency metrics; financial is None when ``gpu_hour_price`` is unset."""
    return {
        "energy_efficiency (Wh/Mtoken)": sustainability_efficiency(powerSource_df, tasks_df, total_tokens),
        "carbon_efficiency (gCO2/Mtoken)": sustainability_efficiency_CO2(powerSource_df, tasks_df, total_tokens),
        "financial_efficiency ($/Mtoken)": (
            financial_efficiency(tasks_df, total_tokens, gpu_hour_price) if gpu_hour_price is not None else None
        ),
        "total_tokens": int(total_tokens),
    }
