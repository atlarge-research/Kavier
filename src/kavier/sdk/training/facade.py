"""Training predictors ``performance``, ``energy``, ``efficiency`` and ``carbon`` for fine-tuning jobs.

Each accepts a DataFrame, a list of dicts or a single dict, and returns the input rows plus predicted
columns as a DataFrame.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from kavier.sdk.co2.engine import Fragment, compute_emissions
from kavier.sdk.domain import RESULT_SOURCE_KEY, Domain
from kavier.sdk.inference.facade import (
    DEFAULT_GPU_HOUR_PRICE,
    DEFAULT_INTENSITY_G_KWH,
    _flat_trace,
    _index_of,
    _normalise,
    _with_columns,
)
from kavier.sdk.library.lookup import get_gpu, get_llm
from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step
from kavier.sdk.units import FLOPS_PER_TFLOP, SECONDS_PER_HOUR, WH_PER_KWH, per_mtoken

DEFAULT_NUM_NODES = 1

# Forward + backward ~3x forward FLOPs: 3 x FLOPS_PER_PARAM_PER_TOKEN (2) = 6 per parameter per token.
# The "6N" of the PaLM MFU definition (Chowdhery et al., 2022).
_TRAINING_FLOPS_PER_PARAM_PER_TOKEN = 6.0


def _train_params(row: dict[str, Any]) -> dict[str, Any]:
    """Return ``row`` with ``num_nodes`` defaulted, the training key batches most often omit."""
    return {**row, "num_nodes": row.get("num_nodes", DEFAULT_NUM_NODES)}


def _mean_flops_utilization(model_name: str, gpu_name: str, train_tokens_per_second: float, total_gpus: int) -> float:
    """Return the realized per-GPU Model FLOPs Utilization [%] implied by the predicted throughput.

    ``MFU = 6 * N_active * tokens/s / (total_gpus * peak_flops) * 100`` (Chowdhery et al., 2022), with
    the catalog's active-parameter count (MoE-aware) and its fp16/bf16 tensor-core peak.

    ``gpu_compute_utilization`` is the assumed efficiency the engine uses to derive throughput; this is
    the efficiency implied by the resulting throughput. Both use ``fp_16_tensor_core_tflops`` as the peak,
    but throughput also carries the calibration multipliers, so the two are not ordered in general.
    Realized MFU falls below assumed when comm/optimizer overhead or a method_scale < 1 dominates (e.g.
    full fine-tuning) and rises above it when a method_scale > 1 (lora ~1.11, gptq-lora ~1.22)
    outweighs the small LoRA overhead. Realized <= assumed holds only on the uncalibrated path, which the
    facade does not use (``run_training`` calls the calibrated ``simulate_full_training``).

    Catalog peaks are dense figures, except the calibrated NVIDIA-H100-PCIe entry, which keeps the
    with-sparsity figure (1513) its calibration was fitted against; its MFU reads ~2x lower than with a
    dense peak. Returns NaN if ``total_gpus`` or the peak is <= 0.
    """
    peak_flops = get_gpu(gpu_name).fp_16_tensor_core_tflops * FLOPS_PER_TFLOP
    denom = total_gpus * peak_flops
    if denom <= 0:
        return math.nan
    n_active = get_llm(model_name).active_params
    return _TRAINING_FLOPS_PER_PARAM_PER_TOKEN * n_active * train_tokens_per_second / denom * 100.0


def run_training(p: dict[str, Any]) -> dict[str, Any]:
    """Return one job's ``simulate_full_training`` and ``simulate_training_step`` outputs, merged."""
    full = simulate_full_training(
        model_name=p["model"],
        method=p["method"],
        gpu_model=p["gpu"],
        tokens_per_sample=int(p["seq_len"]),
        batch_size=int(p["batch_size"]),
        number_gpus=int(p["num_gpus"]),
        number_nodes=int(p["num_nodes"]),
        total_tokens=int(p["total_tokens"]) if p.get("total_tokens") is not None else None,
        epochs=float(p["epochs"]) if p.get("epochs") is not None else None,
        dataset_tokens=int(p["dataset_tokens"]) if p.get("dataset_tokens") is not None else None,
    )
    total_gpus = int(p["num_gpus"]) * int(p["num_nodes"])
    step = simulate_training_step(
        model_name=p["model"],
        gpu_model=p["gpu"],
        tokens_per_sample=int(p["seq_len"]),
        batch_size=int(p["batch_size"]),
        method=p["method"],
        num_gpus=total_gpus,
        num_nodes=int(p["num_nodes"]),
    )
    out: dict[str, Any] = {**full, **step, "total_gpus": total_gpus}
    out["aggregate_power_w"] = step["gpu_power_watts"] * total_gpus
    out["mean_flops_utilization"] = _mean_flops_utilization(
        p["model"], p["gpu"], out["train_tokens_per_second"], total_gpus
    )
    return out


def run_carbon_from_training(p: dict[str, Any]) -> dict[str, Any]:
    """Return the emissions of one training run billed against a flat carbon intensity."""
    tr = run_training(p)
    runtime_s = float(tr["train_runtime"])
    if runtime_s <= 0:
        raise ValueError("training runtime is 0; set a job size (total tokens or epochs) to bill carbon")
    power_w = float(tr["aggregate_power_w"])
    start = pd.Timestamp("2026-01-01 00:00:00")
    trace = _flat_trace(start, runtime_s / SECONDS_PER_HOUR, p["intensity"])
    frag = Fragment(start_time=start, duration_s=runtime_s, power_w=power_w)
    res = compute_emissions([frag], trace)
    return {
        RESULT_SOURCE_KEY: Domain.TRAINING,
        "model": tr["model_name"],
        "gpu": tr["gpu_name"],
        "intensity": float(p["intensity"]),
        "runtime_s": runtime_s,
        "power_w": power_w,
        "total_energy_kwh": res.total_energy_kwh,
        "total_co2_g": res.total_co2_g,
        "total_co2_kg": res.total_co2_kg,
        "total_tokens": tr["total_tokens"],
    }


def performance(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Return ``batch`` with per-job throughput, runtime, utilization and power columns."""
    rows = _normalise(batch)
    cols = (
        "train_tokens_per_second",
        "train_runtime",
        "train_samples_per_second",
        "gpu_compute_utilization",
        "mean_flops_utilization",
        "gpu_memory_utilization",
        "gpu_power_watts",
        "total_tokens",
    )
    predicted: list[dict[str, Any]] = []
    for row in rows:
        r = run_training(_train_params(row))
        predicted.append({k: r[k] for k in cols})
    return _with_columns(rows, predicted, _index_of(batch))


def energy(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Return ``batch`` with per-job energy_wh, energy_kwh, energy_per_mtoken_wh and aggregate_power_w.

    Power comes from the training engine's own GPU power estimate.
    """
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        c = run_carbon_from_training({**_train_params(row), "intensity": DEFAULT_INTENSITY_G_KWH})
        total_tokens = c["total_tokens"]
        energy_wh = c["total_energy_kwh"] * WH_PER_KWH
        predicted.append(
            {
                "energy_wh": energy_wh,
                "energy_kwh": c["total_energy_kwh"],
                "energy_per_mtoken_wh": per_mtoken(energy_wh, total_tokens),
                "aggregate_power_w": c["power_w"],
                "total_tokens": total_tokens,
            }
        )
    return _with_columns(rows, predicted, _index_of(batch))


def efficiency(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Return ``batch`` with per-job financial_per_mtoken [$/Mtoken] and gpu_hours.

    GPU price [$/hour] comes from a ``gpu_hour_price`` column, else DEFAULT_GPU_HOUR_PRICE (2.5).
    """
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        tr = run_training(_train_params(row))
        total_tokens = tr["total_tokens"]
        runtime_s = float(tr["train_runtime"])
        price = float(row.get("gpu_hour_price", DEFAULT_GPU_HOUR_PRICE))
        # GPU-hours = wall-clock runtime x total GPUs, the same basis as inference $/Mtoken.
        gpu_hours = runtime_s / SECONDS_PER_HOUR * int(tr["total_gpus"])
        predicted.append(
            {
                "financial_per_mtoken": per_mtoken(gpu_hours * price, total_tokens) if total_tokens else None,
                "gpu_hours": gpu_hours,
            }
        )
    return _with_columns(rows, predicted, _index_of(batch))


def carbon(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Return ``batch`` with per-job total_co2_g, total_co2_kg, carbon_per_mtoken_g and total_energy_kwh.

    Carbon intensity [gCO2/kWh] comes from an ``intensity`` column, else DEFAULT_INTENSITY_G_KWH (400).
    """
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        intensity = float(row.get("intensity", DEFAULT_INTENSITY_G_KWH))
        c = run_carbon_from_training({**_train_params(row), "intensity": intensity})
        total_tokens = c["total_tokens"]
        predicted.append(
            {
                "total_co2_g": c["total_co2_g"],
                "total_co2_kg": c["total_co2_kg"],
                "carbon_per_mtoken_g": per_mtoken(c["total_co2_g"], total_tokens),
                "total_energy_kwh": c["total_energy_kwh"],
            }
        )
    return _with_columns(rows, predicted, _index_of(batch))
