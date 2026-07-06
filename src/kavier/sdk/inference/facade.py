"""Inference predictors: ``performance / energy / efficiency / carbon`` over a batch of serving workloads.

Each verb accepts a batch (DataFrame, list[dict], or single dict) and returns the input rows plus
predicted columns as a DataFrame.
"""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kavier.sdk.co2.engine import CarbonTrace, Fragment, compute_emissions
from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheCfg, SimConfig
from kavier.sdk.inference.core.metrics import Metrics
from kavier.sdk.inference.core.runner import simulate_one
from kavier.sdk.library import get_gpu, get_llm

# Defaults for workload keys a batch may omit. kv_cache / min_tokens mirror the UI prompt defaults;
# prefix_policy defaults to "none": the facade's synthetic workload shares no prompt content unless a
# policy is opted into, which keeps default outputs identical to when the cache was never consulted.
DEFAULT_KV_CACHE = True
DEFAULT_PREFIX_POLICY = "none"
DEFAULT_PREFIX_MIN_TOKENS = 1024
DEFAULT_INTENSITY_G_KWH = 400.0
DEFAULT_GPU_HOUR_PRICE = 2.5

Batch = "pd.DataFrame | list[dict[str, Any]] | dict[str, Any]"


def _drop_missing(row: dict[str, Any]) -> dict[str, Any]:
    """Drop NaN/None cells so ``.get(key)`` means 'absent' — a heterogeneous DataFrame fills gaps with NaN."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        out[k] = v
    return out


def _normalise(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce a DataFrame | list[dict] | single dict into a list of plain row dicts (NaN cells dropped)."""
    if isinstance(batch, pd.DataFrame):
        records: list[dict[str, Any]] = [{str(k): v for k, v in rec.items()} for rec in batch.to_dict(orient="records")]
    elif isinstance(batch, dict):
        records = [dict(batch)]
    else:
        records = [dict(row) for row in batch]
    return [_drop_missing(row) for row in records]


def _infer_params(row: dict[str, Any]) -> dict[str, Any]:
    """Fill the inference-engine keys, defaulting cache settings the caller may omit; reject
    unsimulatable workloads (num_requests < 1, negative token counts) with a ValueError."""
    if int(row["num_requests"]) < 1:
        raise ValueError(f"num_requests must be >= 1, got {row['num_requests']}")
    for key in ("input_tokens", "output_tokens"):
        if int(row[key]) < 0:
            raise ValueError(f"{key} must be >= 0, got {row[key]}")
    return {
        **row,
        "kv_cache": row.get("kv_cache", DEFAULT_KV_CACHE),
        "prefix_policy": row.get("prefix_policy", DEFAULT_PREFIX_POLICY),
        "prefix_min_tokens": row.get("prefix_min_tokens", DEFAULT_PREFIX_MIN_TOKENS),
    }


def run_inference(p: dict[str, Any]) -> dict[str, Any]:
    """Loop ``simulate_one`` over a homogeneous workload (same engine as the CLI, no disk I/O)."""
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
    # The workload is n IDENTICAL requests: under an active prefix policy, model them as sharing one
    # prompt (a synthetic token list, one session) so the cache can act — request 0 seeds it and the
    # rest hit. Policy "none" (the default) passes no tokens, keeping the cache inert as before.
    shared_tokens = list(range(n_in)) if cfg.cache.action != "none" else None
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
            in_tokens=shared_tokens,
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
        "_tasks": tasks,  # reused by the energy chain
    }


def _flat_trace(start: pd.Timestamp, hours: float, intensity_g_kwh: float) -> CarbonTrace:
    """Constant-intensity trace so ``compute_emissions`` runs without an external grid trace."""
    rows = max(2, int(hours) + 2)
    df = pd.DataFrame(
        {
            "timestamp": [start + dt.timedelta(hours=h) for h in range(rows)],
            "carbon_intensity": [float(intensity_g_kwh)] * rows,
        }
    )
    return CarbonTrace.from_dataframe(df)


def run_carbon_from_inference(infer: dict[str, Any], intensity_g_kwh: float) -> dict[str, Any]:
    """Bill the GPU's max power over the summed busy time against a flat intensity."""
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


def energy_from_inference(infer: dict[str, Any], gpu_hour_price: float | None) -> dict[str, Any]:
    """$/Mtoken from GPU-hours, matching kavier.sdk.energy.metrics.financial_efficiency."""
    carbon = run_carbon_from_inference(infer, intensity_g_kwh=DEFAULT_INTENSITY_G_KWH)
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
    from kavier.sdk.io.opendc.adapter import prepare_opendc_input

    tasks = pd.DataFrame(infer["_tasks"])
    # 1 fragment per task suffices for OpenDC's power model; adapter coerces the schema.
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


def _with_columns(rows: list[dict[str, Any]], predicted: list[dict[str, Any]]) -> pd.DataFrame:
    """Input rows + predicted columns, one output row per input row."""
    merged = [{**row, **pred} for row, pred in zip(rows, predicted)]
    return pd.DataFrame(merged)


def performance(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-workload latency/throughput: + p50_ms, p95_ms, mean_ttft_ms, throughput_tok_s, total_s."""
    rows = _normalise(batch)
    cols = ("p50_ms", "p95_ms", "mean_ttft_ms", "throughput_tok_s", "throughput_req_s", "total_s", "total_tokens")
    predicted: list[dict[str, Any]] = []
    for row in rows:
        r = run_inference(_infer_params(row))
        predicted.append({k: r[k] for k in cols})
    return _with_columns(rows, predicted)


def energy(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-workload energy (self-contained GPU-power estimate): + energy_wh, energy_per_mtoken_wh, tokens_per_wh."""
    rows = _normalise(batch)
    cols = ("energy_wh", "energy_kwh", "energy_per_mtoken_wh", "tokens_per_wh", "total_tokens")
    predicted: list[dict[str, Any]] = []
    for row in rows:
        e = energy_from_inference(run_inference(_infer_params(row)), gpu_hour_price=None)
        predicted.append({k: e[k] for k in cols})
    return _with_columns(rows, predicted)


def efficiency(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-workload cost: + financial_per_mtoken ($/Mtoken). GPU $/hour from a ``gpu_hour_price`` column else 2.5."""
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        price = float(row.get("gpu_hour_price", DEFAULT_GPU_HOUR_PRICE))
        e = energy_from_inference(run_inference(_infer_params(row)), gpu_hour_price=price)
        predicted.append({"financial_per_mtoken": e["financial_per_mtoken"], "gpu_hours": e["gpu_hours"]})
    return _with_columns(rows, predicted)


def carbon(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Per-workload emissions: + total_co2_g, carbon_per_mtoken_g. Intensity from an ``intensity`` column else 400."""
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        intensity = float(row.get("intensity", DEFAULT_INTENSITY_G_KWH))
        infer = run_inference(_infer_params(row))
        c = run_carbon_from_inference(infer, intensity_g_kwh=intensity)
        total_tokens = c["total_tokens"]
        per_m = 1_000_000.0 / total_tokens if total_tokens else 0.0
        predicted.append(
            {
                "total_co2_g": c["total_co2_g"],
                "total_co2_kg": c["total_co2_kg"],
                "carbon_per_mtoken_g": c["total_co2_g"] * per_m,
                "total_energy_kwh": c["total_energy_kwh"],
            }
        )
    return _with_columns(rows, predicted)
