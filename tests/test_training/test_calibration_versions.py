"""Tests for the calibration version files and the use_calibration / available_calibrations /
$KAVIER_CALIBRATION selector; the accessor imports no scipy/sklearn/numpy/pandas for any version."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import kavier.sdk.training.calibration as cal

SRC = Path(__file__).resolve().parents[2] / "src"
CAL_DIR = Path(cal.__file__).resolve().parent
ROOT_JSON = CAL_DIR / "calibration.json"
VERSIONS_DIR = CAL_DIR / "versions"
V4 = VERSIONS_DIR / "calibration_4model.json"
V6 = VERSIONS_DIR / "calibration_6model.json"

# Model sets of the two from-scratch fits, from the calibration design: 4 dense models, and those
# four plus two granites.
DENSE_4 = {"mistral-7b-v0.1", "granite-3.3-8b", "granite-3-8b", "llama3.2-3b"}
ALL_6 = DENSE_4 | {"granite-3.1-2b", "granite-3.1-8b-instruct"}
_REQUIRED_KEYS = (
    "comm_scale",
    "mfu_multiplier",
    "multi_gpu_correction",
    "method_scale",
    "model_scale",
    "interaction_scale",
)
# Copy of the engine's _is_physical band (calibration/engine.py::_SCALE_LO/_SCALE_HI); not imported
# from the code under test.
_SCALE_LO, _SCALE_HI = 0.5, 2.0


@pytest.fixture(autouse=True)
def _restore_cal():
    """Snapshot and restore the module global _CAL around each test."""
    saved = cal._CAL
    yield
    cal._CAL = saved


# --- the shipped version files ------------------------------------------------------------------


def test_versions_shipped_via_importlib_resources():
    # The accessor loads versions through importlib.resources (the wheel path); without versions/
    # in the package data is_file() is False.
    from importlib.resources import files

    for name in ("calibration_4model.json", "calibration_6model.json"):
        res = files("kavier.sdk.training").joinpath("calibration", "versions", name)
        assert res.is_file(), f"{name} not discoverable as package data"


@pytest.mark.parametrize("path", [V4, V6])
def test_version_files_valid_json_with_required_keys(path):
    # Each shipped fit is a JSON object with every table the getters read.
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    assert not missing, f"{path.name} missing {missing}"


def test_root_calibration_is_byte_identical_to_6model():
    # The shipped default calibration.json is the 6-model from-scratch fit.
    assert ROOT_JSON.read_text(encoding="utf-8") == V6.read_text(encoding="utf-8")


def test_6model_covers_all_six_models():
    data = json.loads(V6.read_text(encoding="utf-8"))
    assert set(data["model_scale"]) == ALL_6


def test_4model_covers_exactly_dense_four():
    # No other model appears in model_scale or in the interaction_scale cell keys.
    data = json.loads(V4.read_text(encoding="utf-8"))
    assert set(data["model_scale"]) == DENSE_4
    assert all(k.split("|")[0] in DENSE_4 for k in data["interaction_scale"])


@pytest.mark.parametrize("path", [V4, V6])
def test_shipped_fits_are_physical(path):
    # _is_physical enforces at fit time that comm_scale and every mfu_multiplier lie in [0.5, 2.0].
    data = json.loads(path.read_text(encoding="utf-8"))
    assert _SCALE_LO <= data["comm_scale"] <= _SCALE_HI
    assert all(_SCALE_LO <= v <= _SCALE_HI for v in data["mfu_multiplier"].values())


# --- the selector API ---------------------------------------------------------------------------


def test_available_calibrations_is_default_plus_sorted_versions():
    # "default" first, then one name per versions/calibration_<name>.json, sorted.
    # 4model = thesis validation, 6model = thesis exploration, allmodels = non-thesis fit of all models.
    assert cal.available_calibrations() == ["default", "4model", "6model", "allmodels"]


def test_use_calibration_by_name_switches_live_table():
    # Switching by name installs that fit as the live table and the getters read it.
    # Expected comm_scale is read from the file.
    cal.use_calibration("4model")
    assert set(cal._active_calibration()["model_scale"]) == DENSE_4
    assert cal.get_comm_scale() == pytest.approx(json.loads(V4.read_text())["comm_scale"])
    # a cached first table would keep the 4-model set here
    cal.use_calibration("6model")
    assert set(cal._active_calibration()["model_scale"]) == ALL_6
    assert cal.get_comm_scale() == pytest.approx(json.loads(V6.read_text())["comm_scale"])


def test_use_calibration_default_alias_resolves_to_root_file():
    # "default" resolves to calibration.json, whose comm_scale differs from 4model's.
    cal.use_calibration("default")
    assert cal.get_comm_scale() == pytest.approx(json.loads(ROOT_JSON.read_text())["comm_scale"])


def test_use_calibration_by_path():
    # A filesystem path (with .json or os.sep) takes the path branch.
    loaded = cal.use_calibration(str(V4))
    assert set(loaded["model_scale"]) == DENSE_4


def test_use_calibration_unknown_name_raises_value_error():
    with pytest.raises(ValueError, match="unknown calibration"):
        cal.use_calibration("does-not-exist")


def test_use_calibration_missing_path_raises_file_not_found():
    # A path-like name that does not exist raises.
    with pytest.raises(FileNotFoundError):
        cal.use_calibration("/no/such/calibration.json")


def test_use_calibration_returns_the_installed_table_object():
    # The returned dict is the live _CAL object.
    out = cal.use_calibration("4model")
    assert out is cal._CAL
    assert set(out["model_scale"]) == DENSE_4


# --- env var and stdlib-only import (fresh interpreters) -----------------------------------------


def _subproc(code: str, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)


def test_env_var_selects_calibration_on_first_access():
    # $KAVIER_CALIBRATION selects the table loaded on first access; model counts 4 vs 6.
    code = "import kavier.sdk.training.calibration as c; print(len(c._active_calibration()['model_scale']))"
    assert _subproc(code, {"KAVIER_CALIBRATION": "4model"}).stdout.strip() == "4"
    assert _subproc(code, {"KAVIER_CALIBRATION": "6model"}).stdout.strip() == "6"
    assert _subproc(code, {"KAVIER_CALIBRATION": "default"}).stdout.strip() == "6"


def test_default_without_env_is_six_model():
    # Without the env var, first access loads the root (6-model) file via `... or "default"`.
    code = "import kavier.sdk.training.calibration as c; print(len(c._active_calibration()['model_scale']))"
    env = {k: v for k, v in os.environ.items() if k != "KAVIER_CALIBRATION"}
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "6"


@pytest.mark.parametrize("env", [{}, {"KAVIER_CALIBRATION": "4model"}])
def test_accessor_import_is_stdlib_only(env):
    # Importing and using the accessor, for any selected version, loads no scipy/sklearn/numpy/pandas.
    code = (
        "import sys, kavier.sdk.training.calibration as c; "
        "c.get_comm_scale(); "
        "print([m for m in ('scipy','sklearn','numpy','pandas') if m in sys.modules])"
    )
    proc = _subproc(code, env or None)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
