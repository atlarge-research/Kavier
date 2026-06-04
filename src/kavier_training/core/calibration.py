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
from pathlib import Path
from typing import Any

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.json"

# Calibration-JSON format version this loader targets. Unknown top-level keys in
# the file (including a higher schema_version) are tolerated, not rejected.
SCHEMA_VERSION = 1

with _CALIBRATION_PATH.open(encoding="utf-8") as f:
    _CAL: dict[str, Any] = json.load(f)


def get_comm_scale() -> float:
    return float(_CAL["comm_scale"])


def get_training_overhead_s() -> float:
    return float(_CAL["training_overhead_s"])


def get_mfu_multiplier(gpu_name: str) -> float:
    return float(_CAL["mfu_multiplier"][gpu_name])


def get_multi_gpu_correction(num_gpus: int) -> float:
    if num_gpus <= 1:
        return 1.0
    table = _CAL["multi_gpu_correction"]["by_num_gpus"]
    key = str(num_gpus)
    if key in table:
        return float(table[key])
    nearest = min((int(k) for k in table), key=lambda k: abs(k - num_gpus))
    return float(table[str(nearest)])


def get_method_scale(method: str) -> float:
    return float(_CAL["method_scale"][method])


def get_model_scale(model_name: str) -> float:
    return float(_CAL["model_scale"][model_name])


def get_interaction_scale(model_name: str, method: str, gpu_name: str, num_gpus: int) -> float:
    """Per-(model x method x gpu x gpu-count) residual correction. Defaults to 1.0 for
    any combination not present in the fitted table (so unseen configs are unaffected)."""
    table = _CAL.get("interaction_scale", {})
    key = f"{model_name}|{method}|{gpu_name}|{int(num_gpus)}"
    return float(table.get(key, 1.0))
