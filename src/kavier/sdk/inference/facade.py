"""Batch inference predictors for serving workloads: performance, energy, efficiency and carbon.

Each predictor takes a DataFrame, a list of dicts, or one dict, and returns the input rows plus
predicted columns as a DataFrame.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kavier.sdk.co2.engine import TRACE_INTENSITY_COL, TRACE_TS_COL, CarbonTrace, Fragment, compute_emissions
from kavier.sdk.defaults import (
    DEFAULT_CACHE_SCOPE,
    DEFAULT_EXPORT_RATE,
    DEFAULT_FACADE_PREFIX_POLICY,
)
from kavier.sdk.defaults import (
    DEFAULT_GPU_HOUR_PRICE as DEFAULT_GPU_HOUR_PRICE,
)
from kavier.sdk.defaults import (
    DEFAULT_INTENSITY_G_KWH as DEFAULT_INTENSITY_G_KWH,
)
from kavier.sdk.defaults import (
    DEFAULT_PREFIX_MIN_TOKENS as DEFAULT_PREFIX_MIN_TOKENS,
)
from kavier.sdk.domain import RESULT_SOURCE_KEY, Domain
from kavier.sdk.inference.core.cache import PrefixCache
from kavier.sdk.inference.core.config import CacheAction, CacheCfg, SimConfig
from kavier.sdk.inference.core.metrics import Metrics
from kavier.sdk.inference.core.runner import TASK_ORIGIN_MS, RequestInput, run_request_loop
from kavier.sdk.library import get_gpu, get_llm
from kavier.sdk.units import MS_PER_SECOND, SECONDS_PER_HOUR, WH_PER_KWH, per_mtoken

# Defaults for workload keys a batch may omit. The prefix policy defaults to "none" because the
# synthetic workload shares no prompt content, so the prefix cache has nothing to reuse.
DEFAULT_KV_CACHE = True
DEFAULT_PREFIX_POLICY = DEFAULT_FACADE_PREFIX_POLICY

Batch = "pd.DataFrame | list[dict[str, Any]] | dict[str, Any]"


def _drop_missing(row: dict[str, Any]) -> dict[str, Any]:
    """Drop NaN and None cells so ``.get(key)`` treats them as absent; a mixed DataFrame fills gaps with NaN."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        out[k] = v
    return out


def _normalise(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Return the batch as a list of row dicts with NaN and None cells dropped."""
    if isinstance(batch, pd.DataFrame):
        records: list[dict[str, Any]] = [{str(k): v for k, v in rec.items()} for rec in batch.to_dict(orient="records")]
    elif isinstance(batch, dict):
        records = [dict(batch)]
    else:
        records = [dict(row) for row in batch]
    return [_drop_missing(row) for row in records]


def _infer_params(row: dict[str, Any]) -> dict[str, Any]:
    """Return the row with default cache settings filled in.

    Raises ValueError if num_requests < 1 or a token count is negative.
    """
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


_KV_CACHE_STRINGS = {"on": True, "true": True, "off": False, "false": False}


def _kv_cache_flag(value: Any) -> bool:
    """Return ``value`` as a bool; accepts bools, 0 and 1, and the strings on, off, true and false in any case."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.lower() in _KV_CACHE_STRINGS:
        return _KV_CACHE_STRINGS[value.lower()]
    raise ValueError(f"kv_cache must be a bool, 0 or 1, or one of on, off, true, false; got {value!r}")


def _prefix_policy(value: Any) -> CacheAction:
    try:
        return CacheAction(value)
    except ValueError:
        choices = ", ".join(a.value for a in CacheAction)
        raise ValueError(f"prefix_policy must be one of {choices}; got {value!r}") from None


def run_inference(p: dict[str, Any]) -> dict[str, Any]:
    """Simulate ``num_requests`` identical requests in memory with the CLI's engine.

    Raises ValueError for an unknown ``prefix_policy`` or ``kv_cache`` value.
    """
    llm = get_llm(p["model"])
    gpu = get_gpu(p["gpu"])
    cfg = SimConfig(
        export_rate=DEFAULT_EXPORT_RATE,
        kv_cache=_kv_cache_flag(p["kv_cache"]),
        cache=CacheCfg(
            min_len=int(p["prefix_min_tokens"]),
            action=_prefix_policy(p["prefix_policy"]),
            scope=DEFAULT_CACHE_SCOPE,
            max_entries=10,
        ),
    )

    n = int(p["num_requests"])
    n_in, n_out = int(p["input_tokens"]), int(p["output_tokens"])
    cache = PrefixCache(cfg.cache)
    # Under an active prefix policy the n requests share one synthetic prompt and session: request 0
    # fills the cache and the rest hit. Policy "none" passes no tokens, so the cache is not consulted.
    shared_tokens = list(range(n_in)) if cfg.cache.action != CacheAction.NONE else None
    metrics = Metrics()
    ttfts: list[float] = []
    tasks: list[dict[str, Any]] = []
    requests = (RequestInput(None, n_in, n_out, shared_tokens) for _ in range(n))
    for _i, task, _frags, t_p, _t_d in run_request_loop(
        requests, llm=llm, gpu=gpu, cache=cache, cfg=cfg, metrics=metrics, t0_ms=TASK_ORIGIN_MS
    ):
        ttfts.append(t_p * MS_PER_SECOND)
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
        "_tasks": tasks,  # read by export_opendc
    }


def _flat_trace(start: pd.Timestamp, hours: float, intensity_g_kwh: float) -> CarbonTrace:
    """Return a constant-intensity trace, so ``compute_emissions`` needs no external grid trace."""
    rows = max(2, int(hours) + 2)
    df = pd.DataFrame(
        {
            TRACE_TS_COL: [start + dt.timedelta(hours=h) for h in range(rows)],
            TRACE_INTENSITY_COL: [float(intensity_g_kwh)] * rows,
        }
    )
    return CarbonTrace.from_dataframe(df)


def run_carbon_from_inference(infer: dict[str, Any], intensity_g_kwh: float) -> dict[str, Any]:
    """Compute emissions from GPU max power over the summed busy time at a flat intensity (gCO2/kWh)."""
    gpu = get_gpu(infer["gpu"])
    runtime_s = float(infer["total_s"])
    power_w = float(gpu.max_power_w)
    start = pd.Timestamp("2026-01-01 00:00:00")
    trace = _flat_trace(start, runtime_s / SECONDS_PER_HOUR, intensity_g_kwh)
    frag = Fragment(start_time=start, duration_s=runtime_s, power_w=power_w)
    res = compute_emissions([frag], trace)
    return {
        RESULT_SOURCE_KEY: Domain.INFERENCE,
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
    """Return energy, carbon and cost per Mtoken for an inference result.

    Cost ($/Mtoken) is GPU-hours x ``gpu_hour_price``, as in kavier.sdk.energy.metrics.financial_efficiency;
    None when no price is given.
    """
    carbon = run_carbon_from_inference(infer, intensity_g_kwh=DEFAULT_INTENSITY_G_KWH)
    total_tokens = infer["total_tokens"]
    energy_wh = carbon["total_energy_kwh"] * WH_PER_KWH
    gpu_hours = infer["total_s"] / SECONDS_PER_HOUR
    return {
        "model": infer["model"],
        "gpu": infer["gpu"],
        "total_tokens": total_tokens,
        "energy_wh": energy_wh,
        "energy_kwh": carbon["total_energy_kwh"],
        "energy_per_mtoken_wh": per_mtoken(energy_wh, total_tokens),
        "carbon_per_mtoken_g": per_mtoken(carbon["total_co2_g"], total_tokens),
        "gpu_hours": gpu_hours,
        "financial_per_mtoken": (
            per_mtoken(gpu_hours * gpu_hour_price, total_tokens) if gpu_hour_price is not None else None
        ),
        "tokens_per_wh": total_tokens / energy_wh if energy_wh else 0.0,
    }


def export_opendc(infer: dict[str, Any], dst: Path) -> Path:
    """Write the run's tasks and fragments to ``dst`` as OpenDC input and return ``dst``."""
    from kavier.sdk.io.opendc.adapter import prepare_opendc_input

    tasks = pd.DataFrame(infer["_tasks"])
    # One fragment per task is enough for OpenDC's power model; the adapter coerces the schema.
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


def _with_columns(
    rows: list[dict[str, Any]], predicted: list[dict[str, Any]], index: pd.Index | None = None
) -> pd.DataFrame:
    """Return the input rows merged with their predicted columns, one output row per input row.

    ``index`` is the input DataFrame's index, so results can be assigned back to it.
    """
    merged = [{**row, **pred} for row, pred in zip(rows, predicted)]
    return pd.DataFrame(merged, index=index)


def _index_of(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.Index | None:
    return batch.index if isinstance(batch, pd.DataFrame) else None


def performance(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Predict latency and throughput per workload.

    Adds p50_ms, p95_ms, mean_ttft_ms, throughput_tok_s, throughput_req_s, total_s and total_tokens.
    """
    rows = _normalise(batch)
    cols = ("p50_ms", "p95_ms", "mean_ttft_ms", "throughput_tok_s", "throughput_req_s", "total_s", "total_tokens")
    predicted: list[dict[str, Any]] = []
    for row in rows:
        r = run_inference(_infer_params(row))
        predicted.append({k: r[k] for k in cols})
    return _with_columns(rows, predicted, _index_of(batch))


def energy(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Predict energy per workload from the GPU's max power.

    Adds energy_wh, energy_kwh, energy_per_mtoken_wh, tokens_per_wh and total_tokens.
    """
    rows = _normalise(batch)
    cols = ("energy_wh", "energy_kwh", "energy_per_mtoken_wh", "tokens_per_wh", "total_tokens")
    predicted: list[dict[str, Any]] = []
    for row in rows:
        e = energy_from_inference(run_inference(_infer_params(row)), gpu_hour_price=None)
        predicted.append({k: e[k] for k in cols})
    return _with_columns(rows, predicted, _index_of(batch))


def efficiency(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Predict cost per workload.

    Adds financial_per_mtoken ($/Mtoken) and gpu_hours. The GPU price ($/hour) comes from a
    ``gpu_hour_price`` column, else 2.5.
    """
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        price = float(row.get("gpu_hour_price", DEFAULT_GPU_HOUR_PRICE))
        e = energy_from_inference(run_inference(_infer_params(row)), gpu_hour_price=price)
        predicted.append({"financial_per_mtoken": e["financial_per_mtoken"], "gpu_hours": e["gpu_hours"]})
    return _with_columns(rows, predicted, _index_of(batch))


def carbon(batch: pd.DataFrame | list[dict[str, Any]] | dict[str, Any]) -> pd.DataFrame:
    """Predict emissions per workload.

    Adds total_co2_g, total_co2_kg, carbon_per_mtoken_g and total_energy_kwh. Carbon intensity
    (gCO2/kWh) comes from an ``intensity`` column, else 400.
    """
    rows = _normalise(batch)
    predicted: list[dict[str, Any]] = []
    for row in rows:
        intensity = float(row.get("intensity", DEFAULT_INTENSITY_G_KWH))
        infer = run_inference(_infer_params(row))
        c = run_carbon_from_inference(infer, intensity_g_kwh=intensity)
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
