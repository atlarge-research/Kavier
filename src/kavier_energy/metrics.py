"""Per-Mtoken efficiency metrics: energy (Wh), carbon (gCO2) and cost ($), from OpenDC powerSource + Kavier tasks."""

from __future__ import annotations

import pandas as pd

# Per MILLION tokens (industry-standard unit); raw per-token values are tiny.
TOKENS_PER_UNIT = 1_000_000


def _extract_energy_wh(powerSource: pd.DataFrame) -> float:
    # OpenDC energy_usage is JOULES (W·s, per SimPowerSource.java). 1 Wh = 3600 J -> /3600 (old /1000 was 3.6x high).
    if "energy_usage" in powerSource.columns:
        return powerSource["energy_usage"].sum() / 3_600  # J (W·s) -> Wh
    raise ValueError("energy_usage not in the powerSource.parquet file")


def _extract_co2_emission_g(powerSource: pd.DataFrame) -> float:
    # OpenDC carbon_emission is already GRAMS (gCO2/kWh * kWh, per SimPowerSource.java); no conversion needed.
    if "carbon_emission" in powerSource.columns:
        return float(powerSource["carbon_emission"].sum())  # grams, as-is
    raise ValueError("carbon_emission not in the powerSource.parquet file")


def _total_gpu_hours(tasks: pd.DataFrame) -> float:
    # 1 GPU per task and ``duration`` is per-task latency in ms, so summed duration = total GPU-time. ms -> h.
    return tasks["duration"].sum() / 1_000 / 3_600


def sustainability_efficiency(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    # Wh/Mtoken (lower = better). ``tasks`` unused (stable signature): energy-per-token has no time term.
    return _extract_energy_wh(powerSource) / total_tokens * TOKENS_PER_UNIT


def sustainability_efficiency_CO2(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    return _extract_co2_emission_g(powerSource) / total_tokens * TOKENS_PER_UNIT


def financial_efficiency(tasks: pd.DataFrame, total_tokens: int, gpu_hour_price: float) -> float:
    # $/Mtoken (lower = better) = GPU-hours x user rate / tokens. GPUs dominate; electricity (~2-5%) is omitted.
    return _total_gpu_hours(tasks) * gpu_hour_price / total_tokens * TOKENS_PER_UNIT


def efficiency_summary(
    tasks_df: pd.DataFrame,
    powerSource_df: pd.DataFrame,
    total_tokens: int,
    gpu_hour_price: float | None = None,
) -> dict[str, float | None]:
    # ``financial`` is None until the caller supplies ``gpu_hour_price``.
    return {
        "energy_efficiency (Wh/Mtoken)": sustainability_efficiency(powerSource_df, tasks_df, total_tokens),
        "carbon_efficiency (gCO2/Mtoken)": sustainability_efficiency_CO2(powerSource_df, tasks_df, total_tokens),
        "financial_efficiency ($/Mtoken)": (
            financial_efficiency(tasks_df, total_tokens, gpu_hour_price) if gpu_hour_price is not None else None
        ),
        "total_tokens": int(total_tokens),
    }
