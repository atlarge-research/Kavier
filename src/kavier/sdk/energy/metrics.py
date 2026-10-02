"""Compute per-Mtoken energy (Wh), carbon (gCO2) and cost ($) from OpenDC powerSource and Kavier tasks."""

from __future__ import annotations

import pandas as pd

from kavier.sdk.units import SECONDS_PER_HOUR, TOKENS_PER_MTOKEN


def _extract_energy_wh(powerSource: pd.DataFrame) -> float:
    # OpenDC energy_usage is in J (W s); 1 Wh = 3600 J.
    if "energy_usage" in powerSource.columns:
        return powerSource["energy_usage"].sum() / SECONDS_PER_HOUR
    raise ValueError("energy_usage not in the powerSource.parquet file")


def _extract_co2_emission_g(powerSource: pd.DataFrame) -> float:
    # OpenDC carbon_emission is already in g (gCO2/kWh x kWh).
    if "carbon_emission" in powerSource.columns:
        return float(powerSource["carbon_emission"].sum())
    raise ValueError("carbon_emission not in the powerSource.parquet file")


def _total_gpu_hours(tasks: pd.DataFrame) -> float:
    # One GPU per task and ``duration`` in ms, so the sum is total GPU-time in ms.
    return tasks["duration"].sum() / 1_000 / SECONDS_PER_HOUR


def sustainability_efficiency(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    """Return energy in Wh per million tokens."""
    # ``tasks`` is unused; the signature matches the CO2 variant.
    return _extract_energy_wh(powerSource) / total_tokens * TOKENS_PER_MTOKEN


def sustainability_efficiency_CO2(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    """Return carbon in gCO2 per million tokens."""
    return _extract_co2_emission_g(powerSource) / total_tokens * TOKENS_PER_MTOKEN


def financial_efficiency(tasks: pd.DataFrame, total_tokens: int, gpu_hour_price: float) -> float:
    """Return cost in $/Mtoken from GPU-hours x hourly price; electricity (~2-5%) is omitted."""
    return _total_gpu_hours(tasks) * gpu_hour_price / total_tokens * TOKENS_PER_MTOKEN


def efficiency_summary(
    tasks_df: pd.DataFrame,
    powerSource_df: pd.DataFrame,
    total_tokens: int,
    gpu_hour_price: float | None = None,
) -> dict[str, float | None]:
    """Return energy, carbon and cost efficiency; cost is None when ``gpu_hour_price`` is unset."""
    return {
        "energy_efficiency (Wh/Mtoken)": sustainability_efficiency(powerSource_df, tasks_df, total_tokens),
        "carbon_efficiency (gCO2/Mtoken)": sustainability_efficiency_CO2(powerSource_df, tasks_df, total_tokens),
        "financial_efficiency ($/Mtoken)": (
            financial_efficiency(tasks_df, total_tokens, gpu_hour_price) if gpu_hour_price is not None else None
        ),
        "total_tokens": int(total_tokens),
    }
