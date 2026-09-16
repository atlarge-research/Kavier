"""Training predictors: ``performance / energy / efficiency / carbon`` over a batch of fine-tuning jobs.

Each verb accepts a batch (DataFrame, list[dict], or single dict) and returns the input rows plus
predicted columns as a DataFrame.
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
    _normalise,
    _with_columns,
)
from kavier.sdk.library.lookup import get_gpu, get_llm
from kavier.sdk.training.core.engine import simulate_full_training, simulate_training_step
from kavier.sdk.units import FLOPS_PER_TFLOP, SECONDS_PER_HOUR, WH_PER_KWH, per_mtoken

DEFAULT_NUM_NODES = 1

# Training does a forward + backward pass, ~3x the forward FLOPs, so per parameter per token it is
# 3 x FLOPS_PER_PARAM_PER_TOKEN (=2) = 6 FLOPs — the canonical "6N" of the PaLM/Chowdhery MFU
# definition, and exactly the multiplier the supervisors' mfu_calculator.py uses.
_TRAINING_FLOPS_PER_PARAM_PER_TOKEN = 6.0


def _train_params(row: dict[str, Any]) -> dict[str, Any]:
    """Default ``num_nodes`` (the only training key a batch commonly omits)."""
    return {**row, "num_nodes": row.get("num_nodes", DEFAULT_NUM_NODES)}


def _mean_flops_utilization(model_name: str, gpu_name: str, train_tokens_per_second: float, total_gpus: int) -> float:
    """Realized per-GPU Model FLOPs Utilization (%), backed out of the predicted throughput.

    ``MFU = (6 · N_active · tokens/s) / (total_gpus · peak_flops) · 100`` — the standard hardware MFU
    (Chowdhery et al. 2022), computed from Kavier's predicted ``train_tokens_per_second`` using the
    catalog's active-parameter count (MoE-aware) and its fp16/bf16 tensor-core peak.

    This is DISTINCT from ``gpu_compute_utilization``: that column is the raw *assumed* efficiency the
    engine uses to *derive* throughput, whereas this is the efficiency *implied by* the resulting
    throughput. The denominator is the SAME ``fp_16_tensor_core_tflops`` the engine derives throughput
    against, so the two share one basis — but they are NOT ordered in general, because throughput also
    carries the fitted calibration multipliers that ``gpu_compute_utilization`` omits. Realized MFU
    falls BELOW assumed when comm/optimizer overhead or a method_scale < 1 dominates (e.g. full or
    qlora fine-tuning) and rises ABOVE it when a method_scale > 1 (lora ≈ 1.11, gptq-lora ≈ 1.22)
    outweighs the negligible LoRA overhead. Only on the uncalibrated path (which the facade never
    exercises — ``run_training`` always calls the calibrated ``simulate_full_training``) is realized ≤
    assumed strict.

    Caveat: for Hopper (H100/H200) the catalog figure is NVIDIA's with-sparsity peak (~2× dense bf16),
    so read Hopper MFU as a fraction of that inflated peak, not of dense bf16 — it reads ~2× lower than
    the supervisors' ``mfu_calculator.py`` (which uses a dense peak). A100/L40S carry dense figures and
    need no such caveat. Returns NaN if the denominator is degenerate (``total_gpus`` or peak ≤ 0).
    """
    peak_flops = get_gpu(gpu_name).fp_16_tensor_core_tflops * FLOPS_PER_TFLOP
    denom = total_gpus * peak_flops
    if denom <= 0:
        return math.nan
    n_active = get_llm(model_name).active_params
    return _TRAINING_FLOPS_PER_PARAM_PER_TOKEN * n_active * train_tokens_per_second / denom * 100.0


def run_training(p: dict[str, Any]) -> dict[str, Any]:
    """Aggregate throughput/runtime (``simulate_full_training``) + per-step metrics (``simulate_training_step``)."""
    full = simulate_full_training(
        model_name=p["model"],
        method=p["method"],
        gpu_model=p["gpu"],
        tokens_per_sample=int(p["seq_len"]),
        batch_size=int(p["batch_size"]),
        number_gpus=int(p["num_gpus"]),
        number_nodes=int(p["num_nodes"]),
        total_tokens=int(p["total_tokens"]) if p.get("total_tokens") else None,
        epochs=float(p["epochs"]) if p.get("epochs") else None,
        dataset_tokens=int(p["dataset_tokens"]) if p.get("dataset_tokens") else None,
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
    """Bill one training-engine power fragment against a flat carbon intensity."""
    tr = run_training(p)
    runtime_s = float(tr["train_runtime"])
    if runtime_s <= 0:
        raise ValueError("training runtime is 0 — set a job size (total tokens or epochs) to bill carbon")
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
    """Per-job throughput/util: + train_tokens_per_second, train_runtime, gpu_compute_utilization,
    mean_flops_utilization, gpu_power_watts."""
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
    return _with_columns(rows, predicted)


def energy(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-job energy (self-contained GPU-power estimate): + energy_wh, energy_per_mtoken_wh, aggregate_power_w."""
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
    return _with_columns(rows, predicted)


def efficiency(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-job cost: + financial_per_mtoken ($/Mtoken). GPU $/hour from a ``gpu_hour_price`` column else 2.5."""
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        tr = run_training(_train_params(row))
        total_tokens = tr["total_tokens"]
        runtime_s = float(tr["train_runtime"])
        price = float(row.get("gpu_hour_price", DEFAULT_GPU_HOUR_PRICE))
        # GPU-hours = wall-clock runtime x total GPUs (matches the inference $/Mtoken basis).
        gpu_hours = runtime_s / SECONDS_PER_HOUR * int(tr["total_gpus"])
        predicted.append(
            {
                "financial_per_mtoken": per_mtoken(gpu_hours * price, total_tokens) if total_tokens else None,
                "gpu_hours": gpu_hours,
            }
        )
    return _with_columns(rows, predicted)


def carbon(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-job emissions: + total_co2_g, carbon_per_mtoken_g. Intensity from an ``intensity`` column else 400."""
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
    return _with_columns(rows, predicted)
