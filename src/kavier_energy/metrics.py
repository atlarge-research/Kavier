"""Per-million-token efficiency metrics: energy (Wh), carbon (gCO2) and cost ($)
per Mtoken, derived from an OpenDC powerSource parquet and Kavier tasks."""

from __future__ import annotations

import pandas as pd

# Efficiency is reported per MILLION tokens (the industry-standard unit, e.g. "$ / 1M
# tokens"); raw per-token values are tiny and hard to read. One knob sets the scale.
TOKENS_PER_UNIT = 1_000_000


def _extract_energy_wh(powerSource: pd.DataFrame) -> float:
    # OpenDC writes energy_usage in JOULES (= W·s). Confirmed in OpenDC SimPowerSource.java:
    # `energyUsage = powerSupplied * passedTime * 0.001` (W × ms × 0.001 = W·s), documented
    # "(in J)". 1 Wh = 3600 J, so J -> Wh is /3600 (the old /1000 over-counted energy 3.6x).
    if "energy_usage" in powerSource.columns:
        return powerSource["energy_usage"].sum() / 3_600  # J (W·s) -> Wh
    raise ValueError("energy_usage not in the powerSource.parquet file")


def _extract_co2_emission_g(powerSource: pd.DataFrame) -> float:
    # OpenDC writes carbon_emission in GRAMS. Confirmed in OpenDC SimPowerSource.java:
    # carbonEmission += carbonIntensity * (energyUsage_J / 3_600_000) = gCO2/kWh * kWh = g
    # (the /3.6e6 J->kWh pins carbonIntensity to gCO2/kWh). So no conversion is needed.
    if "carbon_emission" in powerSource.columns:
        return float(powerSource["carbon_emission"].sum())  # grams, as-is
    raise ValueError("carbon_emission not in the powerSource.parquet file")


def _total_gpu_hours(tasks: pd.DataFrame) -> float:
    """Total GPU-hours. Inference runs 1 GPU per task and ``duration`` is the per-task
    latency in ms, so the summed duration is the total GPU-time. ms -> h."""
    return tasks["duration"].sum() / 1_000 / 3_600


def sustainability_efficiency(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    """Energy efficiency: **Wh per million tokens** (lower = better) = energy / tokens.

    ``tasks`` is accepted for a stable call signature but unused: energy-per-token has no
    time term (the previous version multiplied by latency -- a dimensional error)."""
    return _extract_energy_wh(powerSource) / total_tokens * TOKENS_PER_UNIT


def sustainability_efficiency_CO2(powerSource: pd.DataFrame, tasks: pd.DataFrame, total_tokens: int) -> float:
    """Carbon efficiency: **gCO2 per million tokens** (lower = better) = carbon / tokens."""
    return _extract_co2_emission_g(powerSource) / total_tokens * TOKENS_PER_UNIT


def financial_efficiency(tasks: pd.DataFrame, total_tokens: int, gpu_hour_price: float) -> float:
    """Cost efficiency: **$ per million tokens** (lower = better) = GPU-hours x price / tokens.

    GPUs dominate the cost, so this is GPU-hours x the user's GPU-hour rate; electricity is
    a rounding error (~2-5%) and is left out. The rate is user-supplied -- no baked-in
    default. (The seconds cancel: ($/s) / (tokens/s) = $/token, then scaled to per-million.)"""
    return _total_gpu_hours(tasks) * gpu_hour_price / total_tokens * TOKENS_PER_UNIT


def efficiency_summary(
    tasks_df: pd.DataFrame,
    powerSource_df: pd.DataFrame,
    total_tokens: int,
    gpu_hour_price: float | None = None,
) -> dict[str, float | None]:
    """The three per-million-token efficiency metrics (lower = better). ``financial`` is
    ``None`` until the caller supplies ``gpu_hour_price`` -- the GPU cost is the user's to set."""
    return {
        "energy_efficiency (Wh/Mtoken)": sustainability_efficiency(powerSource_df, tasks_df, total_tokens),
        "carbon_efficiency (gCO2/Mtoken)": sustainability_efficiency_CO2(powerSource_df, tasks_df, total_tokens),
        "financial_efficiency ($/Mtoken)": (
            financial_efficiency(tasks_df, total_tokens, gpu_hour_price) if gpu_hour_price is not None else None
        ),
        "total_tokens": int(total_tokens),
    }
