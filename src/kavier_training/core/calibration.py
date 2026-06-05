"""Empirical calibration layer.

Reads the fitted table from ``data/calibration.json`` into the module global
``_CAL`` (which Coastline swaps live for recalibration ablations) and exposes
strict per-key getters over it.

Provenance / forward-compatibility: the JSON may carry purely descriptive
top-level keys such as ``schema_version`` (the on-disk format version),
``version`` (the calibration-run version) and ``_note``. The getters below read
only the specific keys they need, so any unknown/extra top-level key is simply
ignored — the table loads and works unchanged. ``SCHEMA_VERSION`` records the
format version this loader was written against; a mismatch is tolerated, not
fatal.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.json"

# Calibration-JSON format version this loader targets. Unknown top-level keys in
# the file (including a higher schema_version) are tolerated, not rejected.
SCHEMA_VERSION = 1

with _CALIBRATION_PATH.open(encoding="utf-8") as f:
    _CAL: dict[str, Any] = json.load(f)

# Keys we have already warned about, so an uncovered GPU/model/method/gpu-count
# emits at most one stderr warning per distinct key over the process lifetime.
_WARNED_KEYS: set[str] = set()


def _warn_uncovered(table_name: str, key: str, fallback: str) -> None:
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        warnings.warn(
            f"calibration: no {table_name} entry for {key!r}; {fallback}",
            stacklevel=3,
        )


def get_comm_scale() -> float:
    return float(_CAL["comm_scale"])


def get_training_overhead_s() -> float:
    return float(_CAL["training_overhead_s"])


def get_mfu_multiplier(gpu_name: str) -> float:
    # Fall back to a neutral 1.0 (with a one-time warning) for a spec'd-but-
    # uncalibrated GPU rather than KeyError-ing: most library GPUs have no fitted
    # multiplier and would otherwise crash any calibrated=True call on them.
    table = _CAL["mfu_multiplier"]
    if gpu_name not in table:
        _warn_uncovered("mfu_multiplier", gpu_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[gpu_name])


def get_multi_gpu_correction(num_gpus: int) -> float:
    if num_gpus <= 1:
        return 1.0
    table = _CAL["multi_gpu_correction"]["by_num_gpus"]
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
    # Neutral-1.0 fallback (one-time warning) for an uncalibrated method.
    table = _CAL["method_scale"]
    if method not in table:
        _warn_uncovered("method_scale", method, "falling back to neutral 1.0")
        return 1.0
    return float(table[method])


def get_model_scale(model_name: str) -> float:
    # Neutral-1.0 fallback (one-time warning) for an uncalibrated model.
    table = _CAL["model_scale"]
    if model_name not in table:
        _warn_uncovered("model_scale", model_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[model_name])


def get_interaction_scale(model_name: str, method: str, gpu_name: str, num_gpus: int) -> float:
    """Per-(model x method x gpu x gpu-count) residual correction. Defaults to 1.0 for
    any combination not present in the fitted table (so unseen configs are unaffected).

    This silent 1.0 default is intentional and covers, among others, every
    mixtral-8x7b combination: mixtral has a fitted ``model_scale`` but zero
    ``interaction_scale`` entries, so its interaction residual is always the
    neutral 1.0. Other uncovered (model, method, gpu, gpu-count) tuples behave the
    same. Unlike the mfu/method/model/mgc getters, no warning is emitted here: the
    interaction table is sparse by design (only a fraction of fitted rows get a
    residual), so missing combos are the expected common case, not an anomaly."""
    table = _CAL.get("interaction_scale", {})
    key = f"{model_name}|{method}|{gpu_name}|{int(num_gpus)}"
    return float(table.get(key, 1.0))
