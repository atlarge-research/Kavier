from __future__ import annotations

import pandas as pd


def financial_efficiency(
        tasks: pd.DataFrame,
        total_tokens: int,
        gpu_hour_price: float,
) -> float:
    """
    Return in euros / million tokens / second.
    """
    total_latency_s = tasks["duration"].sum() / 1_000
    throughput_tps = total_tokens / total_latency_s

    gpu_price_per_s = gpu_hour_price / 3_600

    eur_per_token_per_s = gpu_price_per_s / throughput_tps
    return eur_per_token_per_s * 1_000_000


def _extract_energy_wh(powerSource: pd.DataFrame) -> float:
    if "energy_usage" in powerSource.columns:
        return powerSource["energy_usage"].sum() / 1_000  # convert from Ws to Wh
    raise ValueError("energy_usage not in powerSource the powerSource.parquet file")


def _extract_co2_emission_kg(powerSource: pd.DataFrame) -> float:
    if "carbon_emission" in powerSource.columns:
        return powerSource["carbon_emission"].sum() / 1_000  # convert from g to kg
    raise ValueError("carbon_emission not in powerSource the powerSource.parquet file")


def sustainability_efficiency(
        powerSource: pd.DataFrame,
        tasks: pd.DataFrame,
        total_tokens: int,
) -> float:
    """
    Result in Wh / million tokens / second.
    """
    energy_kWh = _extract_energy_wh(powerSource) / 1_000
    total_latency_s = tasks["duration"].sum() / 1_000
    return (energy_kWh * total_latency_s) / (total_tokens / 1_000_000)


def sustainability_efficiency_CO2(
        powerSource: pd.DataFrame,
        tasks: pd.DataFrame,
        total_tokens: int,
) -> float:
    """
    Result in kgCO2 / million tokens / second.
    """
    co2_kg = _extract_co2_emission_kg(powerSource)
    total_latency_s = tasks["duration"].sum() / 1_000
    return (co2_kg * total_latency_s) / (total_tokens / 1_000_000)


def efficiency_summary(
        tasks_df: pd.DataFrame,
        powerSource_df: pd.DataFrame,
        total_tokens: int,
        gpu_hour_price: float = 10.0,
) -> dict[str, float]:
    e_fin = financial_efficiency(tasks_df, total_tokens, gpu_hour_price)
    e_sus_wh = sustainability_efficiency(powerSource_df, tasks_df, total_tokens)
    e_sus_co2 = sustainability_efficiency_CO2(powerSource_df, tasks_df, total_tokens)
    total_latency_s = tasks_df["duration"].sum() / 1_000
    return {
        "financial_efficiency (EUR / million token / s)": e_fin,
        "sustainability_efficiency (Wh / million token / s)": e_sus_wh,
        "sustainability_efficiency (kgCO2 / million token / s)": e_sus_co2,
        "total_tokens": total_tokens,
        "total_latency_s": total_latency_s,
    }
