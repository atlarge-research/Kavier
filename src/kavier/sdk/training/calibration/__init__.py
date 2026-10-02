"""Accessors for the fitted calibration table, applied when ``calibrated=True``.

The shipped calibration.json is the 6-model default; other fits live in versions/. Select one with
use_calibration() or $KAVIER_CALIBRATION (available_calibrations() lists them). calibration_override()
in engine.py swaps a table for one with-block; engine.py also regenerates these files.

This module stays stdlib-only: importing it does not load scipy, sklearn, numpy or pandas."""

from __future__ import annotations

import json
import os
import warnings
from enum import Enum
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

# Resources are anchored on kavier.sdk.training, which imports without heavy dependencies.
# calibration.json and versions/ ship in the uv_build wheel with the rest of src/kavier/.
_CALIBRATION_PACKAGE = "kavier.sdk.training"
_CALIBRATION_RESOURCE = ("calibration", "calibration.json")
_VERSIONS_RESOURCE = ("calibration", "versions")
_ENV_VAR = "KAVIER_CALIBRATION"  # name or path loaded on first access (default: root file)

# Callers swap this global (saved = cal._CAL; cal._CAL = ...; cal._CAL = saved).
# calibration_override() in engine.py is the exception-safe form.
_CAL: dict[str, Any] | None = None


def _load_json(handle: Any) -> dict[str, Any]:
    """Read a JSON object from a Path or importlib Traversable."""
    with handle.open(encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def _read_calibration() -> dict[str, Any]:
    """Load the root calibration.json (the 6-model default)."""
    return _load_json(files(_CALIBRATION_PACKAGE).joinpath(*_CALIBRATION_RESOURCE))


def _resolve_calibration(name_or_path: str) -> dict[str, Any]:
    """Load a calibration by name or path.

    ``"default"`` is the root calibration.json; a version name such as ``"4model"`` maps to
    versions/calibration_<name>.json; a value ending in .json or containing a path separator is a file
    path. Raises ValueError for an unknown name and FileNotFoundError for a missing file."""
    s = str(name_or_path)
    if s.endswith(".json") or os.sep in s or (os.altsep and os.altsep in s):
        p = Path(s)
        if not p.is_file():
            raise FileNotFoundError(f"calibration file not found: {s}")
        return _load_json(p)
    if s == "default":
        return _read_calibration()
    resource = files(_CALIBRATION_PACKAGE).joinpath(*_VERSIONS_RESOURCE, f"calibration_{s}.json")
    if not resource.is_file():
        raise ValueError(f"unknown calibration {s!r}; available: {available_calibrations()}")
    return _load_json(resource)


def available_calibrations() -> list[str]:
    """Return the names use_calibration() and $KAVIER_CALIBRATION accept.

    ``"default"`` (the root calibration.json) plus each shipped versions/calibration_<name>.json.
    use_calibration() also accepts a path to a .json file."""
    names = ["default"]
    versions = files(_CALIBRATION_PACKAGE).joinpath(*_VERSIONS_RESOURCE)
    try:
        entries = sorted(e.name for e in versions.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        entries = []
    names += [
        n[len("calibration_") : -len(".json")] for n in entries if n.startswith("calibration_") and n.endswith(".json")
    ]
    return names


def use_calibration(name_or_path: str) -> dict[str, Any]:
    """Install a calibration as the live table the getters read and return it.

    Accepts ``"default"``, a shipped version name (see available_calibrations()) or a path to a .json
    file. For a temporary, block-scoped swap use calibration_override()."""
    global _CAL
    _CAL = _resolve_calibration(name_or_path)
    return _CAL


def _default_calibration() -> dict[str, Any]:
    """Return the table for first access: $KAVIER_CALIBRATION (name or path) if set, else the root file."""
    return _resolve_calibration(os.environ.get(_ENV_VAR) or "default")


def _active_calibration() -> dict[str, Any]:
    global _CAL
    if _CAL is None:
        _CAL = _default_calibration()
    return _CAL


_WARNED_KEYS: set[str] = set()  # each uncovered key warns at most once


def _warn_uncovered(table_name: str, key: str, fallback: str) -> None:
    if key not in _WARNED_KEYS:
        _WARNED_KEYS.add(key)
        warnings.warn(
            f"calibration: no {table_name} entry for {key!r}; {fallback}",
            stacklevel=3,
        )


def get_comm_scale() -> float:
    return float(_active_calibration()["comm_scale"])


def get_training_overhead_s() -> float:
    """Return the fixed per-forward-pass overhead [s] added to each modelled step."""
    return float(_active_calibration()["training_overhead_s"])


def get_mfu_batch_scale() -> tuple[float, float]:
    """Return (alpha, beta) of the MFU-vs-batch curve: batch_scale = min(1, alpha*log2(batch) + beta)."""
    s = _active_calibration()["mfu_batch_scale"]
    return float(s["alpha"]), float(s["beta"])


def get_mfu_multiplier(gpu_name: str) -> float:
    # Uncalibrated GPU: neutral 1.0 with a one-time warning.
    table = _active_calibration()["mfu_multiplier"]
    if gpu_name not in table:
        _warn_uncovered("mfu_multiplier", gpu_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[gpu_name])


def get_multi_gpu_correction(num_gpus: int) -> float:
    """Return the throughput divisor for ``num_gpus``: 1.0 for one GPU, else the value at the nearest fitted count."""
    if num_gpus <= 1:
        return 1.0
    table = _active_calibration()["multi_gpu_correction"]["by_num_gpus"]
    key = str(num_gpus)
    if key in table:
        return float(table[key])
    nearest = min((int(k) for k in table), key=lambda k: abs(k - num_gpus))
    _warn_uncovered(
        "multi_gpu_correction",
        f"num_gpus={num_gpus}",
        f"snapping to nearest fitted count {nearest} (no interpolation)",
    )
    return float(table[str(nearest)])


def get_calibrated_methods() -> frozenset[str]:
    """Return the method names the active calibration has a method_scale entry for."""
    return frozenset(_active_calibration().get("method_scale", {}))


def get_method_scale(method: str) -> float:
    """Return the per-method (full/lora/gptq-lora) throughput scale; 1.0 with a one-time warning if uncalibrated."""
    table = _active_calibration()["method_scale"]
    if method not in table:
        _warn_uncovered("method_scale", method, "falling back to neutral 1.0")
        return 1.0
    return float(table[method])


def get_model_scale(model_name: str) -> float:
    """Return the per-model throughput scale; 1.0 with a one-time warning if uncalibrated."""
    table = _active_calibration()["model_scale"]
    if model_name not in table:
        _warn_uncovered("model_scale", model_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[model_name])


def _key_part(value: Any) -> str:
    """Return ``value`` as written in a table key: an Enum member by its value, a str subclass as plain str."""
    if isinstance(value, Enum):
        value = value.value
    return str.__str__(value) if isinstance(value, str) else str(value)


def get_interaction_scale(model_name: str, method: str, gpu_name: str, num_gpus: int) -> float:
    """Return the residual scale for one (model|method|gpu|num_gpus) cell; 1.0 if absent."""
    table = _active_calibration().get("interaction_scale", {})
    key = f"{_key_part(model_name)}|{_key_part(method)}|{_key_part(gpu_name)}|{int(num_gpus)}"
    return float(table.get(key, 1.0))
