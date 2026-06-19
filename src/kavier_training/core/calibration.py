"""Loader and accessors for the fitted calibration tables (data/calibration.json) applied by the engine when calibrated=True."""

from __future__ import annotations

import json
import warnings
from importlib.resources import files
from typing import Any

# Packaged location of calibration.json, resolved via importlib.resources (wheel-safe).
_CALIBRATION_PACKAGE = "kavier_training"
_CALIBRATION_RESOURCE = ("data", "calibration.json")

# The in-memory calibration tables. ``None`` until first accessed: the file is
# NOT read at import time, so a malformed/missing JSON surfaces at the first
# calibration call (via ``_active_calibration()``), not as an ``import kavier_training`` crash.
#
# This is a module global ON PURPOSE: fitting/benchmark scripts swap a candidate
# table in with ``saved = cal._CAL; cal._CAL = experimental_dict; ...; cal._CAL = saved``
# and the getters must dereference it live. ``_active_calibration()`` lazily materialises it on
# first read and otherwise returns whatever is currently installed.
_CAL: dict[str, Any] | None = None


def _read_calibration() -> dict[str, Any]:
    """Read and parse the packaged calibration.json via importlib.resources (wheel-safe)."""
    resource = files(_CALIBRATION_PACKAGE).joinpath(*_CALIBRATION_RESOURCE)
    with resource.open(encoding="utf-8") as f:
        return json.load(f)


def _active_calibration() -> dict[str, Any]:
    """Return the active calibration tables, lazily loading them on first use.

    Honours an externally-installed ``_CAL`` (the fit-script swap contract); only
    reads ``calibration.json`` when no table has been installed yet. The load is
    therefore lazy (never at import) and happens at most once per process unless
    a caller explicitly resets ``_CAL``.
    """
    global _CAL
    if _CAL is None:
        _CAL = _read_calibration()
    return _CAL


# Calibration tables in calibration.json: comm_scale, training_overhead_s, mfu_multiplier,
# multi_gpu_correction, method_scale, model_scale, interaction_scale (see the getters below).
# _WARNED_KEYS tracks keys already warned about, so each uncovered entry warns at most once.
_WARNED_KEYS: set[str] = set()


def _warn_uncovered(table_name: str, key: str, fallback: str) -> None:
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        warnings.warn(
            f"calibration: no {table_name} entry for {key!r}; {fallback}",
            stacklevel=3,
        )


def get_comm_scale() -> float:
    """Global multiplier applied to the modelled all-reduce communication time."""
    return float(_active_calibration()["comm_scale"])


def get_training_overhead_s() -> float:
    """Fixed per-forward-pass overhead (seconds) added to each modelled step."""
    return float(_active_calibration()["training_overhead_s"])


def get_mfu_batch_scale() -> tuple[float, float]:
    """(alpha, beta) of the MFU-vs-batch curve: batch_scale = min(1, alpha*log2(batch) + beta)."""
    s = _active_calibration()["mfu_batch_scale"]
    return float(s["alpha"]), float(s["beta"])


def get_mfu_multiplier(gpu_name: str) -> float:
    """Per-GPU MFU correction factor; neutral 1.0 (one-time warning) if uncalibrated."""
    # Fall back to a neutral 1.0 (with a one-time warning) for a spec'd-but-
    # uncalibrated GPU rather than KeyError-ing: most library GPUs have no fitted
    # multiplier and would otherwise crash any calibrated=True call on them.
    table = _active_calibration()["mfu_multiplier"]
    if gpu_name not in table:
        _warn_uncovered("mfu_multiplier", gpu_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[gpu_name])


def get_multi_gpu_correction(num_gpus: int) -> float:
    """Throughput-scaling divisor for ``num_gpus``; 1.0 for a single GPU, else the fitted value (snapped to the nearest fitted count, no interpolation)."""
    if num_gpus <= 1:
        return 1.0
    table = _active_calibration()["multi_gpu_correction"]["by_num_gpus"]
    key = str(num_gpus)
    if key in table:
        return float(table[key])
    # No interpolation: snap to the nearest fitted count, but warn — a requested
    # count absent from the table (e.g. 64 -> 32's correction, not near 128's)
    # can be a large, silent nearest-neighbour cliff.
    nearest = min((int(k) for k in table), key=lambda k: abs(k - num_gpus))
    _warn_uncovered(
        "multi_gpu_correction",
        f"num_gpus={num_gpus}",
        f"snapping to nearest fitted count {nearest} (no interpolation)",
    )
    return float(table[str(nearest)])


def get_method_scale(method: str) -> float:
    """Per-method (full/lora/gptq-lora) throughput scale; neutral 1.0 if uncalibrated."""
    # Neutral-1.0 fallback (one-time warning) for an uncalibrated method.
    table = _active_calibration()["method_scale"]
    if method not in table:
        _warn_uncovered("method_scale", method, "falling back to neutral 1.0")
        return 1.0
    return float(table[method])


def get_model_scale(model_name: str) -> float:
    """Per-model throughput scale; neutral 1.0 (one-time warning) if uncalibrated."""
    # Neutral-1.0 fallback (one-time warning) for an uncalibrated model.
    table = _active_calibration()["model_scale"]
    if model_name not in table:
        _warn_uncovered("model_scale", model_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[model_name])


def get_interaction_scale(model_name: str, method: str, gpu_name: str, num_gpus: int) -> float:
    """Residual scale for a specific (model|method|gpu|num_gpus) cell; 1.0 if not present."""
    table = _active_calibration().get("interaction_scale", {})
    key = f"{model_name}|{method}|{gpu_name}|{int(num_gpus)}"
    return float(table.get(key, 1.0))
