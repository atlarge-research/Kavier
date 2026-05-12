"""
Learned calibration parameters for Kavier training throughput.

Fit offline with validation/fit_kavier_calibration.py against measured throughput.
If data/calibration.json is missing, defaults preserve prior behavior (no calibration).
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.json"

# Defaults = physics-only (matches pre-calibration behavior)
_DEFAULT: Dict[str, Any] = {
    "version": 2,
    "comm_scale": 1.0,
    "training_overhead_s": 0.05,
    "mfu_multiplier": {},
    "multi_gpu_correction": {
        "default": 1.0,
        "by_num_gpus": {},
    },
    "method_scale": {},
    "model_scale": {},
}

_active: Dict[str, Any] | None = None


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_calibration_from_disk() -> Dict[str, Any]:
    if _CALIBRATION_PATH.exists():
        with open(_CALIBRATION_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return _deep_merge(_DEFAULT, raw)
    return deepcopy(_DEFAULT)


def get_active() -> Dict[str, Any]:
    global _active
    if _active is None:
        _active = load_calibration_from_disk()
    return _active


def set_active_calibration(data: Dict[str, Any] | None) -> None:
    """Replace in-memory calibration (used by the fitter and tests). None reloads from disk."""
    global _active
    if data is None:
        _active = load_calibration_from_disk()
    else:
        _active = _deep_merge(_DEFAULT, data)


def get_comm_scale() -> float:
    return float(get_active().get("comm_scale", 1.0))


def get_training_overhead_s() -> float:
    return float(get_active().get("training_overhead_s", _DEFAULT["training_overhead_s"]))


def get_mfu_multiplier(gpu_name: str) -> float:
    m = get_active().get("mfu_multiplier") or {}
    return float(m.get(gpu_name, 1.0))


def get_multi_gpu_correction(num_gpus: int) -> float:
    ng = int(num_gpus)
    if ng <= 1:
        return 1.0
    block = get_active().get("multi_gpu_correction") or {}
    by = block.get("by_num_gpus") or {}
    key = str(ng)
    if key in by:
        return float(by[key])
    return float(block.get("default", 1.0))


def get_method_scale(method: str) -> float:
    m = get_active().get("method_scale") or {}
    return float(m.get(method, 1.0))


def get_model_scale(model_name: str) -> float:
    m = get_active().get("model_scale") or {}
    return float(m.get(model_name, 1.0))


def save_calibration(data: Dict[str, Any], path: Path | None = None) -> None:
    p = path or _CALIBRATION_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = _deep_merge(_DEFAULT, data)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
