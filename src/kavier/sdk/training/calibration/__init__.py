"""Accessors for the fitted calibration table (calibration.json), applied when calibrated=True.

The shipped calibration.json is the 6-model default; other from-scratch fits live in versions/.
Pick one with use_calibration(...) or the $KAVIER_CALIBRATION env var (available_calibrations()
lists them); calibration_override(...) swaps a table for a single with-block. The dev-only engine
that regenerates these files lives in engine.py.

Import-light by contract: importing this module must not pull in scipy/sklearn/numpy/pandas, so keep
it stdlib-only (the heavy deps live in engine.py)."""

from __future__ import annotations

import json
import os
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

# Anchor resources on kavier.sdk.training and reach down into calibration/: importing that package
# (and its parents) stays stdlib-light, so the import-light contract holds. calibration.json and
# versions/ ship inside this sub-package via the uv_build wheel (the whole src/kavier/ tree).
_CALIBRATION_PACKAGE = "kavier.sdk.training"
_CALIBRATION_RESOURCE = ("calibration", "calibration.json")
_VERSIONS_RESOURCE = ("calibration", "versions")
_ENV_VAR = "KAVIER_CALIBRATION"  # name or path of the calibration loaded on first access (default: root file)

# Module global by design: callers swap it (saved = cal._CAL; cal._CAL = ...; cal._CAL = saved).
# calibration_override() (in engine.py) is the public, exception-safe form of that swap.
_CAL: dict[str, Any] | None = None


def _load_json(handle: Any) -> dict[str, Any]:
    """Read a JSON object from a Path or importlib Traversable (anything with ``.open(encoding=...)``)."""
    with handle.open(encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def _read_calibration() -> dict[str, Any]:
    """Load the root calibration.json (the shipped 6-model default)."""
    return _load_json(files(_CALIBRATION_PACKAGE).joinpath(*_CALIBRATION_RESOURCE))


def _resolve_calibration(name_or_path: str) -> dict[str, Any]:
    """Load a calibration by name or path: ``"default"`` -> the root calibration.json; a shipped
    version name like ``"4model"`` / ``"6model"`` -> versions/calibration_<name>.json; or a path to
    a .json file. Raises ValueError for an unknown name, FileNotFoundError for a missing path."""
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
    """Names accepted by use_calibration() / $KAVIER_CALIBRATION: ``"default"`` (the root
    calibration.json) plus every shipped versions/calibration_<name>.json (e.g. ``"4model"`` /
    ``"6model"``). A filesystem path to a .json file is also accepted by use_calibration()."""
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
    """Install a named/file calibration as the live table the getters read, returning the loaded
    dict. Accepts ``"default"``, a shipped version name (see available_calibrations()), or a path to
    a .json file. The existing _CAL-swap / calibration_override contract still applies for temporary,
    block-scoped swaps."""
    global _CAL
    _CAL = _resolve_calibration(name_or_path)
    return _CAL


def _default_calibration() -> dict[str, Any]:
    """The table loaded on first access: $KAVIER_CALIBRATION (a name or path) if set, else the root
    calibration.json (the 6-model default)."""
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
    """Fixed per-forward-pass overhead (seconds) added to each modelled step."""
    return float(_active_calibration()["training_overhead_s"])


def get_mfu_batch_scale() -> tuple[float, float]:
    """(alpha, beta) of the MFU-vs-batch curve: batch_scale = min(1, alpha*log2(batch) + beta)."""
    s = _active_calibration()["mfu_batch_scale"]
    return float(s["alpha"]), float(s["beta"])


def get_mfu_multiplier(gpu_name: str) -> float:
    # Uncalibrated GPU -> neutral 1.0 (warn once), not KeyError.
    table = _active_calibration()["mfu_multiplier"]
    if gpu_name not in table:
        _warn_uncovered("mfu_multiplier", gpu_name, "falling back to neutral 1.0")
        return 1.0
    return float(table[gpu_name])


def get_multi_gpu_correction(num_gpus: int) -> float:
    """Throughput-scaling divisor for ``num_gpus``: 1.0 for one GPU, else fitted (nearest count, no interp)."""
    if num_gpus <= 1:
        return 1.0
    table = _active_calibration()["multi_gpu_correction"]["by_num_gpus"]
    key = str(num_gpus)
    if key in table:
        return float(table[key])
    # Snap to nearest fitted count (no interpolation) and warn.
    nearest = min((int(k) for k in table), key=lambda k: abs(k - num_gpus))
    _warn_uncovered(
        "multi_gpu_correction",
        f"num_gpus={num_gpus}",
        f"snapping to nearest fitted count {nearest} (no interpolation)",
    )
    return float(table[str(nearest)])


def get_method_scale(method: str) -> float:
    """Per-method (full/lora/gptq-lora) throughput scale; neutral 1.0 (one-time warning) if uncalibrated."""
    table = _active_calibration()["method_scale"]
    if method not in table:
        _warn_uncovered("method_scale", method, "falling back to neutral 1.0")
        return 1.0
    return float(table[method])


def get_model_scale(model_name: str) -> float:
    """Per-model throughput scale; neutral 1.0 (one-time warning) if uncalibrated."""
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
