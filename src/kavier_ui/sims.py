"""Thin adapters over Kavier's real engines: dict in, dict out, no simulation maths here.

Each function calls an engine entry point directly (simulate_one / simulate_full_training /
simulate_training_step / compute_emissions), so results match the one-shot CLIs exactly.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kavier_co2.emissions import CarbonTrace, Fragment, compute_emissions
from kavier_inference.core.cache import PrefixCache
from kavier_inference.core.config import CacheCfg, SimConfig
from kavier_inference.core.metrics import Metrics
from kavier_inference.core.runner import simulate_one
from kavier_library import GPU_SPEC_LIBRARY, LLM_SPEC_LIBRARY, get_gpu, get_llm
from kavier_training.core.engine import simulate_full_training, simulate_training_step


def model_names() -> list[str]:
    return sorted(LLM_SPEC_LIBRARY)


def gpu_names() -> list[str]:
    return sorted(GPU_SPEC_LIBRARY)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_inference(p: dict[str, Any]) -> dict[str, Any]:
    """Loop ``simulate_one`` over a homogeneous workload from the prompts (same engine as the CLI, no disk I/O)."""
    llm = get_llm(p["model"])
    gpu = get_gpu(p["gpu"])
    cfg = SimConfig(
        export_rate=0.1,
        kv_cache=bool(p["kv_cache"]),
        cache=CacheCfg(min_len=int(p["prefix_min_tokens"]), action=p["prefix_policy"], scope="session", max_entries=10),
    )

    n = int(p["num_requests"])
    n_in, n_out = int(p["input_tokens"]), int(p["output_tokens"])
    cache = PrefixCache(cfg.cache)
    metrics = Metrics()
    t0 = int(time.time_ns() / 1e6)
    ttfts: list[float] = []
    tasks: list[dict[str, Any]] = []
    for i in range(n):
        task, _frags, t_p, t_d = simulate_one(
            idx=i,
            session_id=None,
            n_in_tokens=n_in,
            n_out_tokens=n_out,
            in_tokens=None,
            llm=llm,
            gpu=gpu,
            cache=cache,
            cfg=cfg,
            export_rate_s=cfg.export_rate,
            t0_ms=t0,
        )
        metrics.add(t_p, t_d, (t_p + t_d) * 1000.0)
        ttfts.append(t_p * 1000.0)
        tasks.append(task)

    total_s = metrics.sum_prefill + metrics.sum_decode
    total_tokens = n * (n_in + n_out)
    lat = np.asarray(metrics.latencies)
    return {
        "model": llm.name,
        "gpu": gpu.name,
        "num_requests": n,
        "input_tokens": n_in,
        "output_tokens": n_out,
        "kv_cache": cfg.kv_cache,
        "prefix_policy": cfg.cache.action,
        "prefix_min_tokens": cfg.cache.min_len,
        "prefill_s": metrics.sum_prefill,
        "decode_s": metrics.sum_decode,
        "total_s": total_s,
        "mean_ttft_ms": float(np.mean(ttfts)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "throughput_req_s": n / total_s if total_s else 0.0,
        "throughput_tok_s": total_tokens / total_s if total_s else 0.0,
        "total_tokens": total_tokens,
        "cache_hits": cache.hits,
        "cache_hit_ratio": cache.hits / n if n else 0.0,
        "evictions": cache.evictions,
        "_tasks": tasks,  # internal: reused by the energy chain
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def run_training(p: dict[str, Any]) -> dict[str, Any]:
    """Aggregate throughput/runtime (``simulate_full_training``) + per-step metrics (``simulate_training_step``)."""
    total_tokens = int(p["total_tokens"]) if p.get("total_tokens") else None
    full = simulate_full_training(
        model_name=p["model"],
        method=p["method"],
        gpu_model=p["gpu"],
        tokens_per_sample=int(p["seq_len"]),
        batch_size=int(p["batch_size"]),
        number_gpus=int(p["num_gpus"]),
        number_nodes=int(p["num_nodes"]),
        total_tokens=total_tokens,
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
    out: dict[str, Any] = {**full, **step, "total_gpus": total_gpus, "total_tokens": total_tokens}
    out["aggregate_power_w"] = step["gpu_power_watts"] * total_gpus
    return out


# --------------------------------------------------------------------------- #
# Carbon — analytical chain (no external OpenDC needed)
# --------------------------------------------------------------------------- #
def _flat_trace(start: pd.Timestamp, hours: float, intensity_g_kwh: float) -> CarbonTrace:
    """Constant-intensity trace covering the run, so ``compute_emissions`` runs without an external grid trace."""
    rows = max(2, int(hours) + 2)
    df = pd.DataFrame(
        {
            "timestamp": [start + dt.timedelta(hours=h) for h in range(rows)],
            "carbon_intensity": [float(intensity_g_kwh)] * rows,
        }
    )
    return CarbonTrace.from_dataframe(df)


def run_carbon_from_training(p: dict[str, Any]) -> dict[str, Any]:
    """Bill one training-engine power fragment against a flat carbon intensity via ``compute_emissions``."""
    tr = run_training(p)
    runtime_s = float(tr["train_runtime"])
    if runtime_s <= 0:
        raise ValueError("training runtime is 0 — set a non-zero 'total tokens' to bill carbon")
    power_w = float(tr["aggregate_power_w"])
    start = pd.Timestamp("2026-01-01 00:00:00")
    trace = _flat_trace(start, runtime_s / 3600.0, p["intensity"])
    frag = Fragment(start_time=start, duration_s=runtime_s, power_w=power_w)
    res = compute_emissions([frag], trace)
    return {
        "source": "training",
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


def run_carbon_from_inference(infer: dict[str, Any], intensity_g_kwh: float) -> dict[str, Any]:
    """Bill the GPU's max power over the summed busy time against a flat intensity via ``compute_emissions``."""
    gpu = get_gpu(infer["gpu"])
    runtime_s = float(infer["total_s"])
    power_w = float(gpu.max_power_w)
    start = pd.Timestamp("2026-01-01 00:00:00")
    trace = _flat_trace(start, runtime_s / 3600.0, intensity_g_kwh)
    frag = Fragment(start_time=start, duration_s=runtime_s, power_w=power_w)
    res = compute_emissions([frag], trace)
    return {
        "source": "inference",
        "model": infer["model"],
        "gpu": infer["gpu"],
        "intensity": float(intensity_g_kwh),
        "runtime_s": runtime_s,
        "power_w": power_w,
        "total_energy_kwh": res.total_energy_kwh,
        "total_co2_g": res.total_co2_g,
        "total_co2_kg": res.total_co2_kg,
        "total_tokens": infer["total_tokens"],
    }


# --------------------------------------------------------------------------- #
# Energy / efficiency
# --------------------------------------------------------------------------- #
def energy_from_inference(infer: dict[str, Any], gpu_hour_price: float | None) -> dict[str, Any]:
    """Energy/carbon/$ efficiency per Mtoken for an inference run.

    $/Mtoken is from GPU-hours, matching ``kavier_energy.metrics.financial_efficiency``.
    """
    carbon = run_carbon_from_inference(infer, intensity_g_kwh=400.0)
    total_tokens = infer["total_tokens"]
    energy_wh = carbon["total_energy_kwh"] * 1000.0
    per_m = 1_000_000.0 / total_tokens if total_tokens else 0.0
    gpu_hours = infer["total_s"] / 3600.0
    return {
        "model": infer["model"],
        "gpu": infer["gpu"],
        "total_tokens": total_tokens,
        "energy_wh": energy_wh,
        "energy_kwh": carbon["total_energy_kwh"],
        "energy_per_mtoken_wh": energy_wh * per_m,
        "carbon_per_mtoken_g": carbon["total_co2_g"] * per_m,
        "gpu_hours": gpu_hours,
        "financial_per_mtoken": (gpu_hours * gpu_hour_price * per_m) if gpu_hour_price else None,
        "tokens_per_wh": total_tokens / energy_wh if energy_wh else 0.0,
    }


def export_opendc(infer: dict[str, Any], dst: Path) -> Path:
    """Write the inference run's tasks/fragments as OpenDC input via the real adapter."""
    from kavier_opendc.adapter import prepare_opendc_input

    tasks = pd.DataFrame(infer["_tasks"])
    # Rebuild a minimal fragments frame from task durations (1 fragment per task is
    # sufficient for OpenDC's power model; the adapter coerces the schema).
    frags = pd.DataFrame(
        [
            {
                "id": t["id"],
                "duration": t["duration"],
                "cpu_count": 1,
                "cpu_usage": 0.0,
                "gpu_count": 1,
                "gpu_usage": t["gpu_capacity"],
            }
            for t in infer["_tasks"]
        ]
    )
    dst.mkdir(parents=True, exist_ok=True)
    prepare_opendc_input(tasks, frags, str(dst))
    return dst
