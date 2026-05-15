from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "data" / "calibration.json"

with _CALIBRATION_PATH.open(encoding="utf-8") as f:
    _CAL: dict[str, Any] = json.load(f)

_override: dict[str, Any] | None = None


def _cal() -> dict[str, Any]:
    if _override is not None:
        return _override
    return _CAL


def set_active_calibration(data: dict[str, Any] | None) -> None:
    global _override
    _override = data


def get_comm_scale() -> float:
    return float(_cal()["comm_scale"])


def get_training_overhead_s() -> float:
    return float(_cal()["training_overhead_s"])


def get_mfu_multiplier(gpu_name: str) -> float:
    return float(_cal()["mfu_multiplier"][gpu_name])


def get_multi_gpu_correction(num_gpus: int) -> float:
    if num_gpus <= 1:
        return 1.0
    return float(_cal()["multi_gpu_correction"]["by_num_gpus"][str(num_gpus)])


def get_method_scale(method: str) -> float:
    return float(_cal()["method_scale"][method])


def get_model_scale(model_name: str) -> float:
    return float(_cal()["model_scale"][model_name])


def save_calibration(data: dict[str, Any], path: Path | None = None) -> None:
    p = path or _CALIBRATION_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
